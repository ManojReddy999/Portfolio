// Navigation menu, the home header's scrolled state, and the footer year.

const toggle = document.querySelector('.menu-toggle');
const links = document.querySelector('.nav-links');

if (toggle && links) {
  const setOpen = (open) => {
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    toggle.textContent = open ? '✕' : '☰';
    links.dataset.open = String(open);
  };
  toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
  links.addEventListener('click', (e) => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
}

const header = document.querySelector('.site-header');
if (header && document.body.classList.contains('home')) {
  const update = () => header.classList.toggle('scrolled', window.scrollY > 40);
  update();
  window.addEventListener('scroll', update, { passive: true });
}

document.querySelectorAll('[data-year]').forEach((el) => { el.textContent = String(new Date().getFullYear()); });

// Screen recordings are VP9 WebM; where that can't play (some older Safari), show the original GIF.
const probe = document.createElement('video');
if (!probe.canPlayType('video/webm; codecs="vp9"')) {
  document.querySelectorAll('video').forEach((video) => {
    const gif = video.querySelector('a[href$=".gif"]');
    if (!gif) return;
    const img = new Image();
    img.loading = 'lazy'; // before src, so an 8 MB GIF only loads when scrolled to
    img.alt = video.getAttribute('aria-label') || '';
    img.src = gif.getAttribute('href');
    video.replaceWith(img);
  });
}
