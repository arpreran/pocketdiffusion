---
title: PocketDiffusion
emoji: 📱
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Resource-efficient text-to-image (CPU backup)
---

# PocketDiffusion – always-on CPU backup

Resource-efficient text-to-image generation for mobile devices
(Computer Vision Hackathon 2026, Problem Statement 3).

This Space runs on Hugging Face's free CPU hardware (2 vCPUs, no GPU), so an image takes about
10–30 seconds and the first visit after a quiet period needs a few minutes to start. The planner
knows it is on a CPU and picks SD-Turbo (1 step) with the tiny TAESD decoder. The race page needs
the SD1.5 baseline and is only available in the GPU version.

Source code, report and all measurements: https://github.com/arpreran/pocketdiffusion
