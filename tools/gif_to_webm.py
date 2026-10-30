"""Convert the YouBot screen-recording GIFs to small WebM (VP9) clips plus a poster frame.

The GIFs are 220 dithered frames each (7-8 MB). A 3x3 median removes the dither
noise and every second frame is kept (played at half the frame rate, so real time
is unchanged): about 1.3 MB per clip. Pages load them with preload="none".

Usage: python tools/gif_to_webm.py "assets/images/Mobile Manipulator"/*.gif
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

for arg in sys.argv[1:]:
    src = Path(arg)
    frames = [np.array(f.convert("RGB"))[:, :, ::-1] for f in ImageSequence.Iterator(Image.open(src))]
    durations = [f.info.get("duration", 70) for f in ImageSequence.Iterator(Image.open(src))]
    fps = 1000 / np.median(durations) / 2
    h, w = frames[0].shape[:2]
    out = src.with_suffix(".webm")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))
    for f in frames[::2]:
        writer.write(cv2.medianBlur(f, 3))
    writer.release()
    poster = frames[len(frames) // 3][:, :, ::-1]
    Image.fromarray(poster).save(src.with_name(src.stem + "-poster.webp"), quality=80, method=6)
    print(f"{src.name}: {src.stat().st_size // 1024} KB -> {out.stat().st_size // 1024} KB at {fps:.2f} fps")
