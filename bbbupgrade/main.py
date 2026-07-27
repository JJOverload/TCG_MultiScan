"""
FastAPI backend for the MTG card scanner.

Runs on the "powerful machine" — receives card photos uploaded by the
BeagleBone Black, runs the detection/OCR pipeline in the background, and
serves both the results API and the built React frontend.

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MTG Card Scanner API")

# Wide open for development so the BBB and any LAN/Tailscale client can
# call the API. Once this only ever runs behind Tailscale, this is fine to
# leave — Tailscale is the access control. If you ever expose it more
# broadly, restrict allow_origins to real origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CardMatch(BaseModel):
    name: str
    confidence: float
    box: List[float]  # xmin, ymin, xmax, ymax


class ScanResult(BaseModel):
    id: str
    created_at: datetime
    status: str  # "processing" | "done" | "error"
    image_url: str
    cards: List[CardMatch] = []
    error: Optional[str] = None


# In-memory store — fine for a prototype running on one machine. Swap for
# SQLite (or anything durable) once scans need to survive a restart.
SCANS: Dict[str, ScanResult] = {}


@app.on_event("startup")
def on_startup() -> None:
    pipeline.load_vocab_once(Path("AtomicCards.json"))


@app.post("/api/scans", response_model=ScanResult)
async def create_scan(background_tasks: BackgroundTasks, image: UploadFile = File(...)) -> ScanResult:
    """Called by the BeagleBone Black right after it captures a photo."""
    scan_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{scan_id}.jpg"
    image_path.write_bytes(await image.read())

    result = ScanResult(
        id=scan_id,
        created_at=datetime.utcnow(),
        status="processing",
        image_url=f"/uploads/{scan_id}.jpg",
    )
    SCANS[scan_id] = result

    background_tasks.add_task(_run_pipeline, scan_id, image_path)
    return result


def _run_pipeline(scan_id: str, image_path: Path) -> None:
    try:
        cards = pipeline.scan_image(image_path)
        SCANS[scan_id].cards = [CardMatch(**c) for c in cards]
        SCANS[scan_id].status = "done"
    except Exception as exc:  # surfaced to the client via the `error` field
        SCANS[scan_id].status = "error"
        SCANS[scan_id].error = str(exc)


@app.get("/api/scans", response_model=List[ScanResult])
async def list_scans() -> List[ScanResult]:
    return sorted(SCANS.values(), key=lambda s: s.created_at, reverse=True)


@app.get("/api/scans/{scan_id}", response_model=ScanResult)
async def get_scan(scan_id: str) -> ScanResult:
    if scan_id not in SCANS:
        raise HTTPException(status_code=404, detail="Scan not found")
    return SCANS[scan_id]


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Serve the built React app. `npm run build` in frontend/ produces dist/;
# copy or symlink its contents here before running in "production" mode.
# This mount must come last — StaticFiles(html=True) with root path "/"
# would otherwise swallow the /api and /uploads routes above it.
static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
