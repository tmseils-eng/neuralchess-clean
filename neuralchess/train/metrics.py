"""Run bookkeeping: JSONL metric logs plus a dependency-free SVG plotter."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence


class MetricLogger:
    """Append-only JSONL log; one record per training iteration."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.records: List[dict] = []
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def log(self, **fields) -> dict:
        record = dict(fields)
        record.setdefault("wall_time", time.time())
        self.records.append(record)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def series(self, key: str) -> List[float]:
        return [r[key] for r in self.records if key in r]


def line_chart_svg(series: Dict[str, Sequence[float]], title: str = "",
                   width: int = 720, height: int = 320, ylabel: str = "",
                   xlabel: str = "iteration") -> str:
    """Render one or more series as a standalone SVG.

    Written by hand so that plots can be produced anywhere - including CI and
    cluster nodes - without matplotlib.
    """
    pad_l, pad_r, pad_t, pad_b = 60, 130, 40, 45
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    palette = ["#2f6fdb", "#d1495b", "#2a9d8f", "#e9c46a", "#8367c7", "#f4845f"]

    values = [v for s in series.values() for v in s if v is not None]
    if not values:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    span = vmax - vmin
    vmin -= 0.05 * span
    vmax += 0.05 * span
    nmax = max(len(s) for s in series.values())

    def sx(i: int) -> float:
        return pad_l + (i / max(1, nmax - 1)) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(f'<text x="{pad_l}" y="24" font-size="15" font-weight="600" '
                     f'fill="#1b1b1f">{title}</text>')
    for i in range(5):
        v = vmin + (vmax - vmin) * i / 4
        y = sy(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e6e6ea" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#6b6b76" '
                     f'text-anchor="end">{v:.3g}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
                 f'y2="{pad_t + plot_h}" stroke="#b9b9c2" stroke-width="1"/>')

    for k, (name, data) in enumerate(series.items()):
        color = palette[k % len(palette)]
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(data) if v is not None)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        ly = pad_t + 16 * k + 6
        parts.append(f'<line x1="{pad_l + plot_w + 14}" y1="{ly}" x2="{pad_l + plot_w + 34}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{pad_l + plot_w + 40}" y="{ly + 4}" font-size="11" '
                     f'fill="#3a3a42">{name}</text>')

    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 10}" font-size="11" '
                 f'fill="#6b6b76" text-anchor="middle">{xlabel}</text>')
    if ylabel:
        parts.append(f'<text x="16" y="{pad_t + plot_h / 2}" font-size="11" fill="#6b6b76" '
                     f'text-anchor="middle" transform="rotate(-90 16 {pad_t + plot_h / 2})">'
                     f'{ylabel}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_chart(path: str, series: Dict[str, Sequence[float]], **kwargs) -> str:
    svg = line_chart_svg(series, **kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(svg)
    return path
