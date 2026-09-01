document.documentElement.classList.add("has-js");

const menuToggle = document.querySelector(".menu-toggle");
const primaryNavigation = document.querySelector(".nav-links");

function setMenuState(isOpen) {
  if (!menuToggle || !primaryNavigation) {
    return;
  }

  menuToggle.setAttribute("aria-expanded", String(isOpen));
  menuToggle.setAttribute("aria-label", isOpen ? "Close navigation menu" : "Open navigation menu");
  primaryNavigation.dataset.open = String(isOpen);

  const menuIcon = menuToggle.querySelector("i");
  if (menuIcon) {
    menuIcon.classList.toggle("ph-list", !isOpen);
    menuIcon.classList.toggle("ph-x", isOpen);
  }
}

if (menuToggle && primaryNavigation) {
  menuToggle.addEventListener("click", () => {
    const isOpen = menuToggle.getAttribute("aria-expanded") === "true";
    setMenuState(!isOpen);
  });

  primaryNavigation.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      setMenuState(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
      setMenuState(false);
      menuToggle.focus();
    }
  });

  document.addEventListener("click", (event) => {
    const clickIsInsideNavigation = event.target.closest(".navbar");
    if (!clickIsInsideNavigation && menuToggle.getAttribute("aria-expanded") === "true") {
      setMenuState(false);
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 820) {
      setMenuState(false);
    }
  });
}

document.querySelectorAll("[data-year]").forEach((year) => {
  year.textContent = String(new Date().getFullYear());
});

document.querySelectorAll('a[target="_blank"]').forEach((link) => {
  const relationshipTokens = new Set((link.getAttribute("rel") || "").split(/\s+/).filter(Boolean));
  relationshipTokens.add("noopener");
  relationshipTokens.add("noreferrer");
  link.setAttribute("rel", [...relationshipTokens].join(" "));
});

const revealElements = [...document.querySelectorAll("[data-reveal]")];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (reduceMotion || !("IntersectionObserver" in window)) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.02,
      rootMargin: "0px 0px -8%",
    },
  );

  revealElements.forEach((element) => revealObserver.observe(element));
}
