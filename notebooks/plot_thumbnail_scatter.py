from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly import colors as pc
from PIL import Image, ImageDraw, ImageFile

# Allow loading truncated/corrupt JPEGs without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pillow resampling fallback for compatibility across versions
if hasattr(Image, "Resampling"):
    _RESAMPLE = Image.Resampling.LANCZOS
else:
    _RESAMPLE = getattr(Image, "LANCZOS", Image.BICUBIC)


def _to_base64_png(img: Image.Image) -> Optional[str]:
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except OSError:
        return None


def _make_circular_thumbnail(
    image_path: Path,
    thumb_px: int = 40,
    border_px: int = 3,
    border_rgb: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    # Load and thumbnail preserve aspect ratio
    try:
        with Image.open(image_path) as im:
            img = im.convert("RGB")
    except OSError:
        return None  # unreadable image
    inner_px = max(1, thumb_px - 2 * border_px)
    try:
        img.thumbnail((inner_px, inner_px), _RESAMPLE)
    except Exception:
        return None

    # Create circular mask for inner image
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, img.size[0] - 1, img.size[1] - 1), fill=255)

    # Prepare full canvas with border circle
    canvas = Image.new("RGBA", (thumb_px, thumb_px), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(canvas)
    draw2.ellipse(
        (0, 0, thumb_px - 1, thumb_px - 1),
        fill=(*border_rgb, 255),
    )

    # Paste inner circular image centered
    offset = (
        (thumb_px - img.size[0]) // 2,
        (thumb_px - img.size[1]) // 2,
    )
    # Knock out center so inner shows without border covering it
    hole = Image.new("L", (thumb_px, thumb_px), 0)
    draw_hole = ImageDraw.Draw(hole)
    draw_hole.ellipse(
        (border_px, border_px, thumb_px - border_px - 1, thumb_px - border_px - 1),
        fill=255,
    )
    try:
        canvas.putalpha(hole)
        canvas.paste(img, offset, mask)
    except Exception:
        return None
    return canvas


def _interp_color(colorscale, t: float) -> tuple[int, int, int]:
    """Sample a colorscale (string name or list) at t in [0,1] and return RGB tuple."""
    t = float(np.clip(t, 0.0, 1.0))
    # Resolve colorscale to Plotly format [(pos, color), ...]
    if isinstance(colorscale, str):
        cs = pc.get_colorscale(colorscale)
    else:
        cols = list(colorscale)
        if len(cols) == 0:
            cs = pc.get_colorscale("Viridis")
        elif isinstance(cols[0], (list, tuple)):
            cs = [(float(p), str(c)) for p, c in cols]
        else:
            # Evenly spaced positions for raw color list
            n = len(cols)
            if n == 1:
                cs = [(0.0, cols[0]), (1.0, cols[0])]
            else:
                cs = [(i / (n - 1), c) for i, c in enumerate(cols)]
    # Sample using Plotly helper
    sampled = pc.sample_colorscale(cs, [t])[0]
    # Convert hex like '#RRGGBB' or 'rgb(r,g,b)' to RGB tuple
    if sampled.startswith('#'):
        return tuple(int(sampled[i : i + 2], 16) for i in (1, 3, 5))
    if sampled.startswith('rgb'):
        nums = sampled.strip('rgba()').split(',')[:3]
        return tuple(int(float(x)) for x in nums)
    # Fallback
    return (0, 0, 0)


def plot_thumbnail_scatter(
    df,
    data_dir: Path | str,
    x_col: str = "x",
    y_col: str = "y",
    image_col: str = "filename",
    color_by: Optional[str] = "year_norm",
    colorscale: Iterable[str] | str = "Viridis",
    thumb_px: int = 40,
    border_px: int = 3,
    size_fraction: float = 0.03,
    show_axes: bool = True,
    show_colorbar: bool = True,
    colorbar_title: str = "Year (normalized)",
    max_points: Optional[int] = None,
    equal_axes: bool = True,
    square_canvas: bool = True,
    canvas_size: int = 900,
) -> go.Figure:
    """
    Build a Plotly figure placing circular thumbnails at (x,y) with a colored border.

    - color_by: column in df used to color borders, expects [0,1] normalized; if None, uses a fixed border color.
    - size_fraction: fraction of axis range used for each thumbnail's width/height in data units.
    """
    try:
        data_dir = Path(data_dir)
        # Subset rows if requested
        dff = df.iloc[: int(max_points)].copy() if max_points is not None else df

        xs = np.asarray(dff[x_col].values, dtype=float)
        ys = np.asarray(dff[y_col].values, dtype=float)
        files = list(dff[image_col].values)

        # Normalize color_by if provided
        if color_by is not None:
            vals = np.asarray(dff[color_by].values, dtype=float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            rng = vmax - vmin if vmax > vmin else 1.0
            norm = (vals - vmin) / rng
        else:
            norm = np.zeros(len(files), dtype=float)

        # Compute image sizes in data units
        xr = float(np.nanmax(xs) - np.nanmin(xs) or 1.0)
        yr = float(np.nanmax(ys) - np.nanmin(ys) or 1.0)
        _data_range = max(xr, yr)
        sizex = _data_range * size_fraction
        sizey = _data_range * size_fraction

        # Invisible scatter for hover/select and axes
        scatter = go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=thumb_px * 0.5, opacity=0),  # invisible markers for interaction
            hoverinfo="text",
            text=[f"{Path(f).stem}" for f in files],
            showlegend=False,
        )
        fig = go.Figure(data=[scatter])

        # Add each thumbnail as a layout image
        failures = 0
        for x, y, f, t in zip(xs, ys, files, norm):
            img_path = data_dir / f
            if not img_path.exists() or not img_path.is_file():
                failures += 1
                continue
            rgb = _interp_color(colorscale, float(t))
            thumb = _make_circular_thumbnail(img_path, thumb_px=thumb_px, border_px=border_px, border_rgb=rgb)
            if thumb is None:
                failures += 1
                continue
            uri = _to_base64_png(thumb)
            if uri is None:
                failures += 1
                continue
            fig.add_layout_image(
                dict(
                    source=uri,
                    xref="x",
                    yref="y",
                    x=float(x),
                    y=float(y),
                    sizex=sizex,
                    sizey=sizey,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )

        # Axes and layout
        # Axes and aspect ratio
        fig.update_xaxes(visible=show_axes, constrain="domain")
        fig.update_yaxes(visible=show_axes, scaleanchor="x", scaleratio=1, constrain="domain")

        # Optionally enforce equal displayed ranges so units look identical
        # Important: set both x and y ranges explicitly; do not autorange one of them,
        # otherwise image markers can appear stretched (oblong) in axis units.
        if equal_axes:
            x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
            y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            half = 0.5 * max(x_max - x_min, y_max - y_min)
            if half <= 0:
                half = 0.5
            # Choose common dtick so grid cells look square
            rng = 2.0 * half
            try:
                import math
                raw = max(rng / 8.0, 1e-9)
                mag = 10 ** math.floor(math.log10(raw))
                norm = raw / mag
                if norm <= 1:
                    base = 1
                elif norm <= 2:
                    base = 2
                elif norm <= 5:
                    base = 5
                else:
                    base = 10
                dtick = base * mag
            except Exception:
                dtick = rng / 8.0

            fig.update_xaxes(range=[x_c - half, x_c + half], tickmode="linear", dtick=dtick)
            fig.update_yaxes(range=[y_c - half, y_c + half], tickmode="linear", dtick=dtick)
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=110 if show_colorbar else 20, t=30, b=20),
            dragmode="pan",
            autosize=False,
            width=canvas_size if square_canvas else None,
            height=canvas_size if square_canvas else None,
        )

        # Colorbar trace matching border colors
        if color_by is not None and show_colorbar:
            if isinstance(colorscale, str):
                cs = colorscale
            else:
                cols = list(colorscale)
                cs = "Viridis" if len(cols) <= 1 else [(i / (len(cols) - 1), c) for i, c in enumerate(cols)]
            fig.add_trace(
                go.Scatter(
                    x=[float(np.nanmin(xs)), float(np.nanmax(xs))],
                    y=[float(np.nanmin(ys)), float(np.nanmax(ys))],
                    mode="markers",
                    marker=dict(
                        size=6,
                        opacity=0.01,  # nearly invisible, still renders colorbar
                        color=[0.0, 1.0],
                        cmin=0,
                        cmax=1,
                        colorscale=cs,
                        showscale=True,
                        colorbar=dict(title=colorbar_title),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Ensure PlotlyJS is available in Jupyter
        try:
            pio.renderers.default
        except Exception:
            pio.renderers.default = "notebook_connected"

        if failures:
            print(f"Thumbnail failures: {failures}")
        return fig
    except Exception as e:
        # Fail safe: never crash the notebook; provide context
        import traceback
        print("plot_thumbnail_scatter encountered an error:", repr(e))
        traceback.print_exc()
        return go.Figure()


__all__ = ["plot_thumbnail_scatter"]
