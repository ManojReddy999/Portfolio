"""Move the coursework pages onto the new site shell.

For each page: swap the old stylesheet, icon font and script for site.css/site.js,
replace the header and footer with the shared ones, drop icon-font <i> tags,
point images at their .webp versions, and replace GIF screen recordings that have
a WebM version with a click-to-play <video> (nothing downloads until played).

Usage: python tools/restyle_pages.py Calibration-challenge.html GPI.html ...
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEAD = """  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap">
  <link rel="stylesheet" href="assets/css/site.css">
  <script src="assets/js/site.js" defer></script>"""

HEADER = """  <header class="site-header">
    <nav class="navbar" aria-label="Primary">
      <div class="nav-logo"><a href="./" aria-label="Manoj Reddy Manchala, home">Manoj Reddy Manchala</a></div>
      <button class="menu-toggle" type="button" aria-expanded="false" aria-controls="primary-navigation" aria-label="Open menu">☰</button>
      <ul class="nav-links" id="primary-navigation" data-open="false">
        <li><a href="./#work">Robotics</a></li>
        <li><a href="./#production">Production</a></li>
        <li><a href="lab/">Lab</a></li>
        <li><a href="./#about">About</a></li>
        <li><a href="./#contact">Contact</a></li>
        <li><a class="nav-cta" href="Resume_Manoj.pdf">Résumé</a></li>
      </ul>
    </nav>
  </header>"""

FOOTER = """  <footer class="site-footer">
    <div class="footer-inner container">
      <p>© <span data-year>2026</span> Manoj Reddy Manchala</p>
      <div class="footer-links"><a href="https://github.com/ManojReddy999">GitHub</a><a href="https://www.linkedin.com/in/manojreddymanchala/">LinkedIn</a><a href="mailto:manojreddymanchala@gmail.com">Email</a></div>
    </div>
  </footer>"""

BACK = '<a class="back-link" href="./#work">← All projects</a>'


def webp_for(src):
    path = ROOT / src
    alt = path.with_suffix(".webp")
    return src[: -len(path.suffix)] + ".webp" if path.suffix.lower() in (".png", ".jpg", ".jpeg") and alt.exists() else src


def restyle(html):
    html = re.sub(r'\s*<link rel="stylesheet" href="assets/css/style.css" ?/?>', "", html)
    html = re.sub(r'\s*<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@phosphor-icons[^>]*>', "", html)
    html = re.sub(r'\s*<script (?:defer )?src="assets/js/script.js"(?: defer)?></script>', "", html)
    html = html.replace("</head>", HEAD + "\n</head>", 1)
    html = re.sub(r"  <header class=\"site-header\">.*?</header>", HEADER, html, count=1, flags=re.S)
    html = re.sub(r"  <footer class=\"site-footer\">.*?</footer>", FOOTER, html, count=1, flags=re.S)
    html = re.sub(r'\s*<i class="ph [^"]*"[^>]*></i>', "", html)
    html = re.sub(r"\s*data-reveal", "", html)

    def image(m):
        tag, src = m.group(0), m.group(1)
        gif = ROOT / src
        if gif.suffix.lower() == ".gif" and gif.with_suffix(".webm").exists():
            alt = re.search(r'alt="([^"]*)"', tag).group(1)
            stem = src[:-4]
            return (f'<video controls muted loop playsinline preload="none" poster="{stem}-poster.webp" '
                    f'aria-label="{alt}"><source src="{stem}.webm" type="video/webm">'
                    f'<a href="{src}">{alt} (GIF)</a></video>')
        out = tag.replace(f'src="{src}"', f'src="{webp_for(src)}"')
        if "loading=" not in out:
            out = out.replace("<img ", '<img loading="lazy" decoding="async" ', 1)
        return out

    html = re.sub(r'<img [^>]*src="([^"]+)"[^>]*>', image, html)
    # A way back to the project list, above the page title.
    html = re.sub(r'(<main[^>]*>\s*)', lambda m: m.group(1) + BACK + "\n", html, count=1)
    return html


if __name__ == "__main__":
    for name in sys.argv[1:]:
        p = ROOT / name
        p.write_text(restyle(p.read_text(encoding="utf-8")), encoding="utf-8")
        print("restyled", name)
