"""Record a short walkthrough of the running app as an animated WebP (deploy/pages/showcase/walkthrough.webp).

Drives the real app in headless Microsoft Edge through the DevTools protocol: types a prompt,
generates (live denoising), opens Explain, then edits the result. Frames come from the browser's
screencast and are resampled to a constant frame rate. Needs the local server (python serve.py).

  python scripts/record_walkthrough.py      # run with a Python that has aiohttp and Pillow
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import aiohttp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "pages" / "showcase" / "walkthrough.webp"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
APP = "http://127.0.0.1:8000/#studio"
W, H, FPS, OUT_W = 1280, 760, 8, 960
PROMPT = "a cozy wooden cabin in a snowy forest at night, warm light in the windows"
EDIT_PROMPT = "a cozy wooden cabin in a snowy forest at night, warm light in the windows, watercolor painting"


def free_port() -> int:
    import socket
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def stop_browser(profile: str) -> None:
    """Edge relaunches itself, so killing the started process is not enough: stop every Edge
    process that uses this run's temporary profile (never the user's own browser)."""
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | Where-Object { $_.CommandLine "
                    f"-like '*{Path(profile).name}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force "
                    "-ErrorAction SilentlyContinue }"], capture_output=True, timeout=60)


async def record() -> list[tuple[float, bytes]]:
    prof = tempfile.mkdtemp(prefix="pd_rec_")
    port = free_port()
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                             f"--remote-debugging-port={port}", "--hide-scrollbars", "about:blank"])
    frames: list[tuple[float, bytes]] = []
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except Exception:
                time.sleep(0.5)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with aiohttp.ClientSession() as s, s.ws_connect(ws_url, max_msg_size=2**26) as ws:
            n = 0
            pending: dict[int, asyncio.Future] = {}

            async def reader():
                async for msg in ws:
                    m = json.loads(msg.data)
                    if "id" in m and m["id"] in pending:
                        pending.pop(m["id"]).set_result(m)
                    elif m.get("method") == "Page.screencastFrame":
                        p = m["params"]
                        frames.append((p["metadata"]["timestamp"], base64.b64decode(p["data"])))
                        asyncio.ensure_future(call("Page.screencastFrameAck", sessionId=p["sessionId"]))

            async def call(method, **params):
                nonlocal n
                n += 1
                fut = asyncio.get_event_loop().create_future()
                pending[n] = fut
                await ws.send_str(json.dumps({"id": n, "method": method, "params": params}))
                return await fut

            async def js(expr):
                r = await call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
                return r.get("result", {}).get("result", {}).get("value")

            async def wait_for(expr, timeout=60):
                t_end = time.time() + timeout
                while time.time() < t_end:
                    if await js(expr):
                        return True
                    await asyncio.sleep(0.2)
                dbg = await call("Runtime.evaluate", expression="JSON.stringify({href: location.href, "
                                 "st: typeof state, rs: document.readyState, scripts: [...document.scripts].map(x => x.src), "
                                 "res: performance.getEntriesByType('resource').map(e => e.name.slice(-40) + ' ' + Math.round(e.duration))})",
                                 returnByValue=True)
                raise TimeoutError(f"{expr} | page: {dbg}")

            task = asyncio.ensure_future(reader())
            await call("Page.enable")
            await call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=APP)
            await asyncio.sleep(5)       # evaluating while the page is still navigating hits about:blank
            await wait_for("(typeof state !== 'undefined') && !!(state.sys && state.sys.ready)", 120)
            await js("localStorage.setItem('pd-theme','dark'); document.querySelector('#prompt').value=''; "
                     "document.querySelector('#seed').value='21'; state.style=''; renderStyles(); "
                     "document.querySelector('#prompt').dispatchEvent(new Event('input')); 1")
            await asyncio.sleep(1.0)
            await call("Page.startScreencast", format="jpeg", quality=82, maxWidth=W, maxHeight=H, everyNthFrame=1)
            await asyncio.sleep(0.8)
            await js("document.querySelector('#prompt').focus(); 1")
            for ch in PROMPT:                                   # typed live: scene chips appear
                await call("Input.insertText", text=ch)
                await asyncio.sleep(0.035)
            await asyncio.sleep(1.2)
            await js("document.querySelector('#goBtn').click(); 1")
            await wait_for("!document.querySelector('#stageFinal').classList.contains('hidden') && !state.busy", 60)
            await asyncio.sleep(2.2)
            await js("document.querySelector('#viewToggles [data-view=explain]').click(); 1")
            await wait_for("!!(state.explain && document.querySelector('#explainBar [data-w]'))", 60)
            await asyncio.sleep(1.6)
            for word in ("cabin", "windows", "forest"):
                await js(f"(() => {{ const b = [...document.querySelectorAll('#explainBar [data-w]')]"
                         f".find(x => x.textContent.includes('{word}')); if (b) b.click(); return !!b; }})()")
                await asyncio.sleep(1.4)
            await js("document.querySelector('#viewToggles [data-view=final]').click(); "
                     "document.querySelector('#editBtn').click(); 1")
            await asyncio.sleep(1.0)
            await js(f"(() => {{ const p = document.querySelector('#prompt'); p.value = {json.dumps(EDIT_PROMPT)}; "
                     "p.dispatchEvent(new Event('input')); const s = document.querySelector('#strength'); "
                     "s.value = 90; s.dispatchEvent(new Event('input')); window.scrollTo(0, 0); return 1; })()")
            await asyncio.sleep(1.2)
            await js("document.querySelector('#goBtn').click(); 1")
            await wait_for("!state.busy && state.result && state.result.kind === 'edit'", 60)
            await asyncio.sleep(1.0)
            await js("document.querySelector('#viewToggles [data-view=split]').click(); 1")
            for v in (50, 30, 70, 50):                          # sweep the before | after divider
                await js(f"(() => {{ const r = document.querySelector('#splitRange'); r.value = {v}; "
                         "r.dispatchEvent(new Event('input')); return 1; })()")
                await asyncio.sleep(0.6)
            await asyncio.sleep(1.5)
            await call("Page.stopScreencast")
            await asyncio.sleep(0.3)
            task.cancel()
    finally:
        proc.kill()
        stop_browser(prof)
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)
    return frames


def encode(frames: list[tuple[float, bytes]]) -> None:
    frames.sort(key=lambda f: f[0])
    t0, t1 = frames[0][0], frames[-1][0]
    out, j = [], 0
    for k in range(int((t1 - t0) * FPS) + 1):             # constant frame rate: hold the last frame
        t = t0 + k / FPS
        while j + 1 < len(frames) and frames[j + 1][0] <= t:
            j += 1
        im = Image.open(io.BytesIO(frames[j][1])).convert("RGB")
        out.append(im.resize((OUT_W, round(im.height * OUT_W / im.width)), Image.LANCZOS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(OUT, "WEBP", save_all=True, append_images=out[1:], duration=int(1000 / FPS), loop=0,
                quality=62, method=4)
    Image.open(io.BytesIO(frames[-1][1])).convert("RGB").save(OUT.with_name("walkthrough_poster.webp"),
                                                             "WEBP", quality=80)
    print(f"{len(frames)} screencast frames -> {len(out)} frames, {(t1 - t0):.1f} s, "
          f"{OUT.stat().st_size / 2**20:.1f} MB -> {OUT}")


if __name__ == "__main__":
    encode(asyncio.run(record()))
