/* charts.js — labeled SVG line charts, no dependencies.
   Each chart gets gridlines, y-axis ticks, and an optional dashed
   baseline (the error tolerance) with a label. */

const Charts = (() => {
  const W = 560;
  const H = 160;
  const PAD = { top: 10, right: 12, bottom: 18, left: 38 };

  function svgEl(tag, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  function text(el, x, y, str, cls, anchor) {
    const t = svgEl("text", { x, y, class: cls });
    if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = str;
    el.appendChild(t);
  }

  /**
   * Render a labeled line chart of `values` into `container`.
   * opts: {
   *   min, max        — y range (defaults to data range)
   *   baseline        — dashed reference line value
   *   baselineLabel   — label for the baseline
   *   color           — "green" | "terra" | "blue"
   *   yFormat(v)      — formats y tick labels
   * }
   */
  function render(container, values, opts = {}) {
    container.innerHTML = "";
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}` });
    const color = opts.color || "terra";
    const yFormat = opts.yFormat || ((v) => v.toFixed(2));

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    if (!values || values.length < 2) {
      const msg = svgEl("text", {
        x: W / 2, y: H / 2 + 4, "text-anchor": "middle", class: "empty-msg",
      });
      msg.textContent = "Waiting for data \u2014 check some answers first.";
      svg.appendChild(msg);
      container.appendChild(svg);
      return;
    }

    let lo = opts.min !== undefined ? opts.min : Math.min(...values);
    let hi = opts.max !== undefined ? opts.max : Math.max(...values);
    if (opts.baseline !== undefined) {
      lo = Math.min(lo, opts.baseline);
      hi = Math.max(hi, opts.baseline);
    }
    if (hi === lo) hi = lo + 1;
    const span = hi - lo;

    const x = (i) => PAD.left + (i * plotW) / (values.length - 1);
    const y = (v) => PAD.top + plotH - ((v - lo) / span) * plotH;

    // gridlines + y ticks (4 lines)
    for (let g = 0; g <= 3; g++) {
      const v = lo + (span * g) / 3;
      const yy = y(v);
      svg.appendChild(svgEl("line", {
        x1: PAD.left, x2: W - PAD.right, y1: yy, y2: yy, class: "grid-line",
      }));
      text(svg, PAD.left - 6, yy + 3, yFormat(v), "axis-label", "end");
    }

    // baseline (tolerance)
    if (opts.baseline !== undefined) {
      const yy = y(opts.baseline);
      svg.appendChild(svgEl("line", {
        x1: PAD.left, x2: W - PAD.right, y1: yy, y2: yy, class: "baseline",
      }));
      if (opts.baselineLabel) {
        text(svg, W - PAD.right, yy - 4, opts.baselineLabel, "baseline-label", "end");
      }
    }

    // area + line
    const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("polygon", {
      points: `${x(0).toFixed(1)},${y(lo).toFixed(1)} ${points} ${x(values.length - 1).toFixed(1)},${y(lo).toFixed(1)}`,
      class: `area ${color}`,
    }));
    svg.appendChild(svgEl("polyline", { points, class: `line ${color}` }));

    // last-value dot
    const lastX = x(values.length - 1);
    const lastY = y(values[values.length - 1]);
    svg.appendChild(svgEl("circle", {
      cx: lastX, cy: lastY, r: 3,
      fill: color === "green" ? "#7ba88f" : color === "blue" ? "#5b7fa6" : "#e8833a",
    }));

    container.appendChild(svg);
  }

  return { render };
})();
