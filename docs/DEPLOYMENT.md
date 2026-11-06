# Deployment & submission checklist

## Live URL (required deliverable)
The app needs a GPU to answer quickly, so the simplest reliable option is to serve it from the
laptop's RTX 4060 through Gradio's share relay:

```bash
run.bat --share
```

* Starts the full-stack app (`python serve.py --share`). After ~30 s the console prints a box with
  `PUBLIC URL: https://xxxx.gradio.live`: that is the live URL to submit. It works from phones
  and laptops while the process runs (Gradio share links last up to one week).
* Keep the laptop plugged in, disable sleep, and keep the terminal open during judging.
  **Every restart gives a new URL**: update the report and the submission if that happens.
* The models are pre-loaded and warmed up at start, so the first judge request is fast.

**If no URL appears** (`lookup gradio-live.com: i/o timeout`): the tunnel client's DNS lookup
fails on some phone hotspots. `serve.py` retries 6 times and cycles through three strategies:
Gradio's default client, connect-by-IP (the name is resolved by Windows, the TLS certificate is
still checked against the real host name) and a public DNS server for the client. On the team's
iPhone hotspot the connect-by-IP strategy works. If all attempts fail:
1. switch to another network (home/college Wi-Fi) and run `run.bat --share` again, or
2. use a Cloudflare quick tunnel (free, no account), in a second terminal while `run.bat` runs:
   ```bash
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel --url http://localhost:8000
   ```
   It prints `https://<random>.trycloudflare.com`.

Alternatives: a Hugging Face Space with a GPU (paid) using the provided `Dockerfile`, or any cloud
VM with an NVIDIA GPU: `docker build -t pocketdiffusion . && docker run --gpus all -p 8000:8000 -v pdata:/app/data pocketdiffusion`.

## GitHub repository
Published at https://github.com/arpreran/pocketdiffusion (push new commits with `git push`). How it was set up:
1. On github.com: **New repository** → name `pocketdiffusion` → Public → *no* README/.gitignore → Create.
2. In this folder:
   ```bash
   git remote add origin https://github.com/<your-username>/pocketdiffusion.git
   git push -u origin main
   ```
   Git for Windows opens a browser window to sign in the first time.
3. Put the URL and `git rev-parse --short HEAD` (the evaluated commit) in the report header.

`.gitignore` excludes model weights, datasets and generated evaluation images. The distilled
6-layer text encoder (161 MB) exceeds GitHub's 100 MB file limit: it is rebuilt with
`python scripts/distill_text_encoder.py` (~11 min); without it the planner simply skips that option.
No API keys or tokens are needed or stored anywhere.

## Before submitting
- [ ] `team.json`: real team ID and USNs → `python -m eval.make_report` → `eval_P3_<TeamID>.md` (first line = USNs).
- [ ] `Solution_Report_P3.md`: header filled, every TEAM-TODO block written in your own words,
      `python scripts/ai_share.py Solution_Report_P3.md` prints PASS.
- [ ] Architecture diagram: `docs/architecture.md` (Mermaid, renders on GitHub) and `docs/architecture.png`.
- [ ] AI declaration (report section 6) accurate for your team.
- [ ] Live URL tested from a phone on mobile data.
- [ ] Final `git commit` + `git push`; the commit hash in the report matches.
