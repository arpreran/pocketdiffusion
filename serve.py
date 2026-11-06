"""Start the PocketDiffusion full-stack app (FastAPI backend + web frontend + SQLite).

    python serve.py                 # http://127.0.0.1:8000
    python serve.py --share         # + public https link (phones / judges), via Gradio's tunnel
    python serve.py --no-warmup     # skip model pre-loading (UI development)
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
import time
from pathlib import Path

# never hit the network for weights once they are cached (robust live demo); --online to disable
_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_HUB / "models--stable-diffusion-v1-5--stable-diffusion-v1-5").exists() and "--online" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _tunnel_via_resolved_ip(port: int) -> str:
    """Same Gradio relay, but its host name is resolved here (Windows resolver) and the tunnel
    client gets the IP. The bundled frpc client's own DNS lookup times out on some phone hotspots
    ("lookup gradio-live.com: i/o timeout") even when Windows resolves the name fine."""
    import socket

    import httpx
    from gradio import networking
    from gradio.tunneling import Tunnel

    p = httpx.get(networking.GRADIO_API_SERVER, timeout=30).json()[0]
    host, rport = p["host"], int(p["port"])
    ip = socket.getaddrinfo(host, rport, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    cert = Path(networking.CERTIFICATE_PATH)
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(p["root_ca"])
    return Tunnel(ip, rport, "127.0.0.1", port, secrets.token_urlsafe(32), str(cert)).start_tunnel()


def open_tunnel(port: int, attempts: int = 6) -> None:
    """Expose the local server through Gradio's share relay (same mechanism as gradio --share).

    Retries with back-off; if the tunnel client cannot resolve the relay, resolve it here instead.
    """
    from gradio.networking import setup_tunnel
    for i in range(1, attempts + 1):
        try:
            if i % 2 == 1:
                url = setup_tunnel("127.0.0.1", port, secrets.token_urlsafe(32), None, None)
            else:
                url = _tunnel_via_resolved_ip(port)
            bar = "=" * (len(url) + 22)
            print(f"\n{bar}\n  PUBLIC URL:  {url}\n{bar}\n  (valid while this process runs)\n", flush=True)
            return
        except Exception as e:  # tunnel is optional - the local server keeps working
            msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__
            print(f"[share] attempt {i}/{attempts} failed: {msg}", flush=True)
            time.sleep(5 * i)
    print("[share] no public link - the app still runs locally. Check the internet connection and restart.",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()

    import uvicorn
    from backend.main import app, jobs

    if not args.no_warmup:
        jobs.submit("warmup", {})
    if args.share:
        threading.Thread(target=lambda: (time.sleep(2), open_tunnel(args.port)), daemon=True).start()
    print(f"PocketDiffusion -> http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
