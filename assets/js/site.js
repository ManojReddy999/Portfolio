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

// ---- Motion: scroll reveals, reading progress, the contents list following along ----
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Content rises in as it reaches the viewport. Only what starts below the fold is
// hidden, so nothing visible on load flickers; siblings in a grid stagger slightly.
const REVEAL = '.section-head, .feature, .cards > .card, .watch-grid > div, .about-copy, .skills > div, .contact > div, '
  + '.case-meta, .case-content section > h2, .case-content figure, .case-content .table-wrap, .case-content .deltas, '
  + '.project-details-card section, .also';
if (!reduceMotion && 'IntersectionObserver' in window) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }, { rootMargin: '0px 0px -6% 0px', threshold: 0.06 });
  for (const el of document.querySelectorAll(REVEAL)) {
    if (el.getBoundingClientRect().top < innerHeight) continue;
    const peers = [...el.parentElement.children].filter((c) => c.matches(REVEAL));
    el.style.setProperty('--i', String(Math.min(peers.indexOf(el) % 3, 2)));
    el.classList.add('reveal');
    io.observe(el);
  }
}

// Reading progress on project pages.
const article = document.querySelector('.case-study-page, .project-details-page');
if (article) {
  const bar = document.createElement('div');
  bar.className = 'read-progress';
  bar.setAttribute('aria-hidden', 'true');
  document.body.append(bar);
  let queued = false;
  const update = () => {
    queued = false;
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.setProperty('--p', String(max > 0 ? Math.min(1, scrollY / max) : 0));
  };
  addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

// The contents list highlights the section on screen.
const tocLinks = [...document.querySelectorAll('.case-aside a[href^="#"]')];
if (tocLinks.length && 'IntersectionObserver' in window) {
  const byId = new Map(tocLinks.map((a) => [a.getAttribute('href').slice(1), a]));
  const visible = new Set();
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id);
    const first = [...byId.keys()].find((id) => visible.has(id));
    if (first) tocLinks.forEach((a) => a.classList.toggle('active', a === byId.get(first)));
  }, { rootMargin: '-20% 0px -55% 0px' });
  byId.forEach((_, id) => { const s = document.getElementById(id); if (s) spy.observe(s); });
}

// Autoplaying hero videos stay on their first frame for reduced-motion visitors.
if (reduceMotion) document.querySelectorAll('video[autoplay]').forEach((v) => { v.removeAttribute('autoplay'); v.pause(); v.controls = true; });
