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


LIVE_URL_FILE = Path(__file__).resolve().parent / "data" / "live_url.txt"


def open_tunnel(port: int, attempts: int = 6) -> str | None:
    """Expose the local server through Gradio's share relay (same mechanism as gradio --share).

    Retries with back-off and cycles through: Gradio's default client, connect-by-IP, public DNS.
    Returns the public URL (also written to data/live_url.txt) or None.
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
            try:
                LIVE_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
                LIVE_URL_FILE.write_text(f"{url}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
            except OSError:
                pass
            return url
        except Exception as e:  # tunnel is optional - the local server keeps working
            msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__
            print(f"[share] attempt {i}/{attempts} ({strategy}) failed: {msg}", flush=True)
            time.sleep(3 * i)
    print("[share] no public link right now - the app still runs locally; retrying in a minute.", flush=True)
    return None


def _public_ok(url: str) -> bool:
    import httpx
    try:
        return httpx.get(url.rstrip("/") + "/api/system", timeout=20).status_code == 200
    except Exception:
        return False


ROOT_DIR = Path(__file__).resolve().parent
PAGES_DIR = ROOT_DIR / "data" / "pages"        # local clone of the repo's gh-pages branch
PAGES_BRANCH = "gh-pages"


def publish_live_url(url: str) -> None:
    """Write the current public URL to live.json on the repo's gh-pages branch.

    GitHub Pages serves deploy/pages/index.html at a permanent address
    (https://<user>.github.io/<repo>/); that page reads live.json, checks the server is up and
    forwards visitors - so the address given to examiners never changes, even when the tunnel
    URL does. Uses the git credentials already stored on this computer; failures are only logged.
    """
    import json
    import shutil
    import subprocess

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def git(*args, cwd=PAGES_DIR):
        return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=180)

    try:
        if not (PAGES_DIR / ".git").exists():
            origin = git("remote", "get-url", "origin", cwd=ROOT_DIR).stdout.strip()
            if not origin:
                print("[pages] no git remote 'origin' - permanent page not updated", flush=True)
                return
            r = git("clone", "--branch", PAGES_BRANCH, "--single-branch", "--depth", "1", origin,
                    str(PAGES_DIR), cwd=ROOT_DIR)
            if r.returncode:
                print(f"[pages] could not clone {PAGES_BRANCH}: {r.stderr.strip()[-200:]}", flush=True)
                return
        else:
            git("pull", "--ff-only", "--depth", "1")
        shutil.copyfile(ROOT_DIR / "deploy" / "pages" / "index.html", PAGES_DIR / "index.html")
        (PAGES_DIR / "live.json").write_text(
            json.dumps({"url": url, "updated": time.strftime("%d %b %Y, %H:%M")}) + "\n", encoding="utf-8")
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode == 0:
            return                                          # nothing changed
        git("commit", "-m", f"Live demo address: {url}")
        r = git("push", "origin", PAGES_BRANCH)
        if r.returncode:
            print(f"[pages] push failed: {r.stderr.strip()[-200:]}", flush=True)
        else:
            print(f"[pages] permanent page now forwards to {url} (live within ~1 min)", flush=True)
    except Exception as e:  # never let publishing break the server
        print(f"[pages] could not update the permanent page: {type(e).__name__}: {e}", flush=True)


def keep_tunnel_alive(port: int, interval: float = 60.0, publish: bool = False) -> None:
    """Open the public link and keep it alive.

    The relay drops the link when the laptop switches network or sleeps (gradio.live then answers
    404 although the local server is fine). Every minute the public URL is checked from outside;
    after two failed checks in a row the old tunnel is closed and a new one opened (new URL, printed
    in the box and written to data/live_url.txt). With `publish`, every new URL is also pushed to
    the permanent GitHub Pages address (see publish_live_url).
    """
    from gradio import tunneling
    url = open_tunnel(port)
    if url and publish:
        publish_live_url(url)
    fails = 0
    while True:
        time.sleep(interval)
        if url and _public_ok(url):
            fails = 0
            continue
        fails += 1
        if url and fails < 2:
            continue
        print(f"[share] public link {url or '-'} is not reachable - reconnecting ...", flush=True)
        for t in list(tunneling.CURRENT_TUNNELS):
            try:
                t.kill()
            except Exception:
                pass
        tunneling.CURRENT_TUNNELS.clear()
        new = open_tunnel(port)
        if new and url and new != url:
            print(f"[share] THE LINK CHANGED: {url} -> {new}"
                  + ("  (permanent page updated automatically)" if publish else "  (send the new one)"), flush=True)
        if new and publish:
            publish_live_url(new)
        url, fails = new, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--publish-link", action="store_true",
                    help="with --share: keep the permanent GitHub Pages page pointing at the current link")
    args = ap.parse_args()

    import uvicorn
    from backend.main import app, jobs

    if not args.no_warmup:
        jobs.submit("warmup", {})
    if args.share:
        threading.Thread(target=lambda: (time.sleep(2), keep_tunnel_alive(args.port, publish=args.publish_link)),
                         daemon=True).start()
    print(f"PocketDiffusion -> http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
