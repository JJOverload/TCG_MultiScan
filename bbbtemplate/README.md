# MTG card scanner — webserver + BBB capture client

Three pieces, matching the earlier architecture diagram:

```
mtg-scanner/
├── backend/       FastAPI app — runs on the powerful machine, does the scanning
├── frontend/      React app — talks to the backend, viewed on LAN or via Tailscale
└── bbb-client/    Runs ON the BeagleBone Black — captures a photo, uploads it
```

## 1. Backend (on your powerful machine)

```bash
cd backend
pip install -r requirements.txt --break-system-packages
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Drop your `AtomicCards.json` and the EAST model into `backend/` (or point
`pipeline.load_vocab_once()` at wherever you keep them). Right now
`app/pipeline.py` returns a stub result so the whole stack is runnable
before that wiring is done — swap in the real `detect_text_boxes()` /
`best_match_across_rotations()` calls from the standalone scanner's
`main.py` when ready.

## 2. Frontend (dev, on your laptop)

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`, proxying `/api` and `/uploads` to the
backend at `localhost:8000` (see `vite.config.js`). Point that proxy at
your Tailscale or LAN backend address if you're developing against the
real machine.

For "production" (one process serving everything):

```bash
npm run build
cp -r dist/* ../backend/static/
```

Then just run the backend — FastAPI serves the built frontend from `/`.

## 3. BBB capture client (on the BeagleBone Black)

```bash
python3 capture_client.py --backend-url http://100.x.x.x:8000
```

Use the backend's Tailscale IP (or `scanner.local` if you've set up mDNS)
so this works whether the BBB and backend are on the same LAN or not.
Add `--interval 10` to capture every 10 seconds instead of once.

## Access

- **LAN**: browse to `http://<backend-machine>.local:8000` (needs
  `avahi`/mDNS running on the backend machine).
- **Remote**: install [Tailscale](https://tailscale.com) on both the
  backend machine and whatever device you're browsing from, then use the
  backend's Tailscale IP — no port forwarding needed.

## Still to do

- Wire `pipeline.scan_image()` up to the real detection/OCR code.
- Swap the in-memory `SCANS` dict for SQLite once scans need to survive a
  restart.
- Consider a WebSocket or Server-Sent Events endpoint if polling every 3s
  feels laggy once you're testing with real capture volume.
