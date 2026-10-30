"""Write WebP versions of the site's images, and the small thumbnails the home page uses.

- PNG/JPG -> .webp beside the original (lossless when that is smaller than near-lossless
  lossy, which it is for most plots), capped at 1600 px wide.
- GIF -> animated .webp beside the original; usually a fifth of the size or less.
- Thumbnails -> assets/images/thumbs/<slug>.webp, 640 px wide.

Originals are kept so old links keep working. Usage: python tools/optimize_images.py
"""

import io
from pathlib import Path

from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "assets" / "images"
SKIP = {"car.gif", "car-other1.png", "car-other2.png", "car-other3.png", "car-unique.png", "road_bg.jpg"}

THUMBS = {
    "vi-slam": "VI-SLAM/Pasted image.png",
    "lidar-slam": "LiDAR-SLAM/occupancy_map.png",
    "calibration": "Calibration challenge/vanishing_point.png",
    "youbot": "Mobile Manipulator/Screenshot from 2025-03-12 11-46-40.png",
    "astar-rrt": "astar-rrt/Screenshot from 2024-05-24 20-41-04.png",
    "door-key": "Door-Key/-8x8-28.gif",
    "gpi": "GPI/Figure_6.png",
    "orientation": "Orientation-Tracking/IMUandCamSensors.jpg",
    "live-lab": "lab/screenshot-desktop.jpg",
}


def encode(im, lossless):
    buf = io.BytesIO()
    if lossless:
        im.save(buf, "WEBP", lossless=True, quality=100, method=6)
    else:
        im.save(buf, "WEBP", quality=82, method=6)
    return buf.getvalue()


def still(src, dst, max_w=1600):
    im = Image.open(src)
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    lossy = encode(im, False)
    data = lossy
    if src.suffix.lower() == ".png":
        lossless = encode(im, True)
        if len(lossless) < 1.3 * len(lossy):
            data = lossless
    dst.write_bytes(data)
    return len(data)


def animated(src, dst):
    im = Image.open(src)
    frames = [f.convert("RGBA") for f in ImageSequence.Iterator(im)]
    durations = [f.info.get("duration", 100) for f in ImageSequence.Iterator(Image.open(src))]
    frames[0].save(dst, "WEBP", save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, quality=72, method=4)
    return dst.stat().st_size


def main():
    total_before = total_after = 0
    for src in sorted(IMAGES.rglob("*")):
        if src.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif") or src.name in SKIP:
            continue
        if src.suffix.lower() == ".gif":
            # GIFs with a WebM clip (tools/gif_to_webm.py) are served as video, and their
            # names collide with stills on case-insensitive disks (Best_Case.gif, Best_case.png).
            if src.with_suffix(".webm").exists():
                continue
            dst = src.with_suffix(".webp")
            size = animated(src, dst)
            if size >= src.stat().st_size:  # small palette GIFs: WebP doesn't help
                dst.unlink()
                continue
        else:
            dst = src.with_suffix(".webp")
            size = still(src, dst)
        total_before += src.stat().st_size
        total_after += size
        print(f"{src.relative_to(IMAGES)}: {src.stat().st_size // 1024} KB -> {size // 1024} KB")

    thumbs = IMAGES / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for slug, rel in THUMBS.items():
        im = Image.open(IMAGES / rel)
        im = im.convert("RGB") if im.mode != "RGBA" else im
        w = 640
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im.save(thumbs / f"{slug}.webp", "WEBP", quality=78, method=6)
        print(f"thumb {slug}: {(thumbs / f'{slug}.webp').stat().st_size // 1024} KB")
    print(f"total {total_before // 1024} KB -> {total_after // 1024} KB")


if __name__ == "__main__":
    main()
