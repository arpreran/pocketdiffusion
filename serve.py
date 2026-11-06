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


def _tunnel_with_config(port: int, strategy: str) -> str:
    """Same Gradio relay and the same frpc client, started from a config file so DNS can be fixed.

    The bundled frpc client's own DNS lookup times out on some phone hotspots
    ("lookup gradio-live.com: i/o timeout") although Windows resolves the name fine.
      strategy "ip":  connect to the IP resolved here, verify TLS against the real host name
      strategy "dns": let frpc resolve the name through a public DNS server (1.1.1.1)
    TLS certificate verification stays on in both cases.
    """
    import atexit
    import socket
    import subprocess
    import tempfile

    import httpx
    from gradio import networking, tunneling

    p = httpx.get(networking.GRADIO_API_SERVER, timeout=30).json()[0]
    host, rport = p["host"], int(p["port"])
    cert = Path(networking.CERTIFICATE_PATH)
    cert.parent.mkdir(parents=True, exist_ok=True)
    cert.write_text(p["root_ca"])
    if strategy == "ip":
        addr = socket.getaddrinfo(host, rport, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    else:
        addr = host
    token = secrets.token_urlsafe(32)
    ini = (f"[common]\nserver_addr = {addr}\nserver_port = {rport}\n"
           f"tls_enable = true\ntls_trusted_ca_file = {cert.as_posix()}\ntls_server_name = {host}\n"
           f"dns_server = 1.1.1.1\ndisable_log_color = true\n\n"
           f"[{token}]\ntype = http\nlocal_ip = 127.0.0.1\nlocal_port = {port}\n"
           f"use_encryption = true\nuse_compression = true\nsubdomain = random\n")
    fd, ini_path = tempfile.mkstemp(prefix="pocketdiff_frpc_", suffix=".ini")
    with os.fdopen(fd, "w") as f:
        f.write(ini)

    class ConfigTunnel(tunneling.Tunnel):
        def _start_tunnel(self, binary: str) -> str:
            tunneling.CURRENT_TUNNELS.append(self)
            self.proc = subprocess.Popen([binary, "-c", ini_path],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            atexit.register(self.kill)
            return self._read_url_from_tunnel_stream()

    try:
        return ConfigTunnel(addr, rport, "127.0.0.1", port, token, str(cert)).start_tunnel()
    finally:
        try:
            os.remove(ini_path)   # frpc has read it; the file holds the tunnel token
        except OSError:
            pass


def open_tunnel(port: int, attempts: int = 6) -> None:
    """Expose the local server through Gradio's share relay (same mechanism as gradio --share).

    Retries with back-off and cycles through: Gradio's default client, connect-by-IP, public DNS.
    """
    from gradio.networking import setup_tunnel
    strategies = ["default", "ip", "dns"]
    for i in range(1, attempts + 1):
        strategy = strategies[(i - 1) % len(strategies)]
        try:
            if strategy == "default":
                url = setup_tunnel("127.0.0.1", port, secrets.token_urlsafe(32), None, None)
            else:
                url = _tunnel_with_config(port, strategy)
            bar = "=" * (len(url) + 22)
            print(f"\n{bar}\n  PUBLIC URL:  {url}\n{bar}\n  (valid while this process runs)\n", flush=True)
            return
        except Exception as e:  # tunnel is optional - the local server keeps working
            msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__
            print(f"[share] attempt {i}/{attempts} ({strategy}) failed: {msg}", flush=True)
            time.sleep(3 * i)
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
