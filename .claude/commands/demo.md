---
description: Render the radar HUD demo video and GIF
argument-hint: [regime: vacuum | drag | drag_wind]
---
Run `.venv/bin/ballistic demo --regime $ARGUMENTS --out assets/demo.mp4 --gif assets/demo.gif`
(use `drag_wind` if no regime was given).
Report the live FDE stats it prints, extract 2-3 frames with ffmpeg into the scratchpad,
look at them, and point out any visual problem (overlapping labels, clutter, unreadable text).
