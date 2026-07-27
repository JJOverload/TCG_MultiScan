"""
main.py

Orchestrates the full pipeline: load vocab -> detect text boxes -> OCR each
box across several rotation angles -> fuzzy-match against vocab -> optionally
score against an answer file -> write the annotated output image.

Depends on box_merging.py and vocab.py in the same directory.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image

from box_merging import BoundingBox, decode, merge_close_boxes
from vocab import AutocorrectMatch, Vocabulary, autocorrect

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROTATION_ANGLES = [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]
CONFIDENT_MATCH_SIMILARITY = 0.6
STOP_AFTER_N_STALE_ROTATIONS = 3
NOISE_SIMILARITY_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and identify MTG card names in an image using an "
        "EAST text-detection network plus OCR and fuzzy vocabulary matching."
    )
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--model", default="frozen_east_text_detection.pb",
                         help="Path to EAST model weights (.pb).")
    parser.add_argument("--vocab", default="AtomicCards.json",
                         help="Path to MTGJSON AtomicCards.json.")
    parser.add_argument("--width", type=int, default=320,
                         help="Resize width for detection (multiple of 32).")
    parser.add_argument("--height", type=int, default=320,
                         help="Resize height for detection (multiple of 32).")
    parser.add_argument("--thr", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--answername", default="", help="Path to answer .txt for accuracy scoring.")
    parser.add_argument("--showtext", action="store_true", help="Draw matched names onto the output image.")
    parser.add_argument("--outdir", default="box_images", help="Directory for intermediate crop/rotation images.")
    parser.add_argument("--dist-limit", type=float, default=40, help="Box-merging distance threshold.")
    return parser.parse_args()


def load_model(model_path: str, device: str) -> cv.dnn_Net:
    net = cv.dnn.readNet(model_path)
    if device == "gpu":
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        log.info("Using GPU device")
    else:
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        log.info("Using CPU device")
    return net


def detect_text_boxes(
    net: cv.dnn_Net,
    frame: np.ndarray,
    inp_width: int,
    inp_height: int,
    conf_thresh: float,
    nms_thresh: float,
    dist_limit: float,
    expand_dist: int,
) -> List[BoundingBox]:
    """Run the EAST network on `frame` and return merged, axis-aligned
    bounding boxes in original-image coordinates."""
    height, width = frame.shape[:2]
    scale_x, scale_y = width / float(inp_width), height / float(inp_height)

    blob = cv.dnn.blobFromImage(
        frame, 1.0, (inp_width, inp_height), (123.68, 116.78, 103.94), True, False
    )
    net.setInput(blob)
    scores, geometry = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )
    detections = decode(scores, geometry, conf_thresh)

    rotated_rects = [(d.center, (d.width, d.height), d.angle_degrees) for d in detections]
    confidences = [d.confidence for d in detections]
    keep_indices = cv.dnn.NMSBoxesRotated(rotated_rects, confidences, conf_thresh, nms_thresh)

    boxes: List[BoundingBox] = []
    for i in keep_indices:
        vertices = cv.boxPoints(rotated_rects[i])
        vertices[:, 0] *= scale_x
        vertices[:, 1] *= scale_y
        xs, ys = vertices[:, 0], vertices[:, 1]
        boxes.append(BoundingBox(
            xmin=xs.min() - expand_dist, ymin=ys.min() - expand_dist,
            xmax=xs.max() + expand_dist, ymax=ys.max() + expand_dist,
        ))

    return merge_close_boxes(boxes, dist_limit)


def best_match_across_rotations(
    crop: Image.Image, vocab: Vocabulary, outdir: Path, box_index: int
) -> AutocorrectMatch:
    """Try OCR at several small rotation angles, keeping the best vocab
    match. Stops early once a confident match has held for a few rotations
    in a row — same idea as the original's countdown counter."""
    best = autocorrect("", vocab)
    best_similarity = 0.0
    stale_rotation_count = 0

    for angle in ROTATION_ANGLES:
        rotated = crop.rotate(angle)
        rotated.save(outdir / f"box{box_index}_{angle}.jpg")

        text = pytesseract.image_to_string(rotated).strip()
        match = autocorrect(text, vocab)

        if match.similarity > best_similarity:
            best_similarity = match.similarity
            best = match
            stale_rotation_count = 0
        elif best_similarity >= CONFIDENT_MATCH_SIMILARITY:
            stale_rotation_count += 1

        log.info("box%d @ %d deg: '%s' -> %s", box_index, angle, text, match)

        if stale_rotation_count >= STOP_AFTER_N_STALE_ROTATIONS:
            log.info("Confident match found, skipping remaining rotations")
            break

    return best


def score_against_answer_key(guesses: List[str], answer_path: str) -> None:
    answers = [line.strip() for line in Path(answer_path).read_text().splitlines()]
    remaining = list(answers)
    correct = incorrect = 0

    for guess in guesses:
        if guess in remaining:
            correct += 1
            remaining.remove(guess)
        else:
            incorrect += 1

    missed = len(remaining)
    total = correct + incorrect + missed
    accuracy = 100 * correct / total if total else 0.0
    log.info("Correct: %d  Incorrect: %d  Missed: %d  Accuracy: %.1f%%",
              correct, incorrect, missed, accuracy)
    log.info("Unmatched answers: %s", remaining)


def main() -> None:
    start_time = datetime.datetime.now()
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    log.info("Loading vocabulary from %s", args.vocab)
    vocab = Vocabulary.load(Path(args.vocab))

    log.info("Loading model from %s", args.model)
    net = load_model(args.model, args.device)

    frame = cv.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    expand_dist = int(args.dist_limit / 10)
    log.info("Detecting text boxes")
    boxes = detect_text_boxes(
        net, frame, args.width, args.height, args.thr, args.nms, args.dist_limit, expand_dist
    )

    matches: List[Tuple[str, AutocorrectMatch, BoundingBox]] = []
    noise: List[Tuple[str, AutocorrectMatch, BoundingBox]] = []
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    for i, box in enumerate(boxes, start=1):
        crop_region = (int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax))
        crop = Image.fromarray(rgb_frame).crop(crop_region)

        match = best_match_across_rotations(crop, vocab, outdir, i)
        log.info("Box %d best match: %s", i, match)

        if match.name in vocab.non_name_words or match.similarity <= NOISE_SIMILARITY_THRESHOLD:
            noise.append((match.name, match, box))
        else:
            matches.append((match.name, match, box))

    output_frame = frame.copy()
    if args.showtext:
        for name, _, box in matches:
            cv.putText(output_frame, name, (int(box.xmin), int(box.ymin)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        for _, _, box in noise:
            cv.putText(output_frame, "~noise~", (int(box.xmin), int(box.ymin)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    if args.answername:
        score_against_answer_key([name for name, _, _ in matches], args.answername)

    cv.imwrite("Rec3.jpg", output_frame)
    log.info("Elapsed time: %s", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
