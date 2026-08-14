(function () {
  const nav = document.getElementById("site-nav");
  const toggle = document.querySelector(".nav-toggle");
  const sectionIds = [
    "overview",
    "algorithms",
    "evaluation",
    "impact",
    "citations",
    "next-steps",
  ];

  function setNavOpen(open) {
    if (!nav || !toggle) return;
    nav.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", String(open));
  }

  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      setNavOpen(!nav.classList.contains("is-open"));
    });

    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        setNavOpen(false);
      });
    });
  }

  const navLinks = Array.from(
    document.querySelectorAll('.site-nav a[href^="#"]')
  );

  function highlightNav() {
    const offset = 90;
    let current = sectionIds[0];
    sectionIds.forEach(function (id) {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top - offset <= 0) {
        current = id;
      }
    });
    navLinks.forEach(function (link) {
      const match = link.getAttribute("href") === "#" + current;
      if (match) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  }

  window.addEventListener("scroll", highlightNav, { passive: true });
  highlightNav();

  const models = ["XGBoost", "Logistic regression", "Random forest"];
  const series = [
    { key: "recall", label: "Recall (%)", color: "#0b5cab", values: [78, 53, 45] },
    {
      key: "ap",
      label: "Avg. precision ×100",
      color: "#c45c26",
      values: [63, 60, 57],
    },
    {
      key: "auc",
      label: "ROC-AUC ×100",
      color: "#2a7a4b",
      values: [83, 82, 81],
    },
  ];

  const visibility = { recall: true, ap: true, auc: true };
  const canvas = document.getElementById("model-chart");
  const legendEl = document.getElementById("chart-legend");
  if (!canvas || !legendEl) return;

  const ctx = canvas.getContext("2d");
  let hover = null;

  function visibleSeries() {
    return series.filter(function (s) {
      return visibility[s.key];
    });
  }

  function layout() {
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 800;
    const cssHeight = Math.max(280, Math.round(cssWidth * 0.42));
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: cssWidth, h: cssHeight };
  }

  function barRects(size) {
    const pad = { top: 16, right: 16, bottom: 42, left: 42 };
    const plotW = size.w - pad.left - pad.right;
    const plotH = size.h - pad.top - pad.bottom;
    const vis = visibleSeries();
    const groupCount = models.length;
    const groupW = plotW / groupCount;
    const gap = groupW * 0.22;
    const inner = groupW - gap;
    const barW = vis.length ? inner / vis.length : inner;
    const maxY = 100;
    const rects = [];

    models.forEach(function (model, i) {
      vis.forEach(function (s, j) {
        const value = s.values[i];
        const barH = (value / maxY) * plotH;
        const x = pad.left + i * groupW + gap / 2 + j * barW;
        const y = pad.top + plotH - barH;
        rects.push({
          x: x,
          y: y,
          w: Math.max(barW - 4, 4),
          h: barH,
          color: s.color,
          model: model,
          label: s.label,
          value: value,
        });
      });
    });

    return { pad: pad, plotW: plotW, plotH: plotH, rects: rects, maxY: maxY };
  }

  function draw() {
    const size = layout();
    const geo = barRects(size);
    ctx.clearRect(0, 0, size.w, size.h);

    ctx.strokeStyle = "#d7dee6";
    ctx.fillStyle = "#5b6573";
    ctx.font = "12px Segoe UI, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let tick = 0; tick <= 100; tick += 20) {
      const y = geo.pad.top + geo.plotH - (tick / geo.maxY) * geo.plotH;
      ctx.beginPath();
      ctx.moveTo(geo.pad.left, y);
      ctx.lineTo(geo.pad.left + geo.plotW, y);
      ctx.stroke();
      ctx.fillText(String(tick), geo.pad.left - 8, y);
    }

    geo.rects.forEach(function (r) {
      ctx.fillStyle = r.color;
      ctx.globalAlpha = hover && hover !== r ? 0.45 : 1;
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.globalAlpha = 1;
    });

    ctx.fillStyle = "#1c2430";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const groupW = geo.plotW / models.length;
    models.forEach(function (model, i) {
      const x = geo.pad.left + i * groupW + groupW / 2;
      ctx.fillText(model, x, geo.pad.top + geo.plotH + 12);
    });

    if (hover) {
      const text = hover.model + " · " + hover.label + " · " + hover.value;
      ctx.font = "13px Segoe UI, system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      let tx = hover.x + hover.w / 2 - tw / 2;
      let ty = hover.y - 28;
      if (tx < 8) tx = 8;
      if (tx + tw + 16 > size.w) tx = size.w - tw - 16;
      if (ty < 8) ty = hover.y + 8;
      ctx.fillStyle = "#1c2430";
      ctx.fillRect(tx - 8, ty - 4, tw + 16, 22);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, tx, ty + 7);
    }
  }

  series.forEach(function (s) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "legend-btn";
    btn.innerHTML =
      '<span class="legend-swatch" style="background:' +
      s.color +
      '"></span>' +
      s.label;
    btn.addEventListener("click", function () {
      visibility[s.key] = !visibility[s.key];
      btn.classList.toggle("is-off", !visibility[s.key]);
      hover = null;
      draw();
    });
    legendEl.appendChild(btn);
  });

  function hitTest(event) {
    const size = { w: canvas.clientWidth, h: canvas.clientHeight };
    const geo = barRects(size);
    const bounds = canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    return (
      geo.rects.find(function (r) {
        return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
      }) || null
    );
  }

  canvas.addEventListener("mousemove", function (event) {
    hover = hitTest(event);
    draw();
  });

  canvas.addEventListener("mouseleave", function () {
    hover = null;
    draw();
  });

  window.addEventListener("resize", draw);
  draw();
})();
