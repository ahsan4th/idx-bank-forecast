"""Apache ECharts options for st.echarts_chart (pure dicts, testable without Streamlit).

Colour follows the model, never its rank: each model keeps one categorical slot
in every chart (palette validated with the dataviz validator for both modes:
adjacent-pair CVD and normal-vision separation pass). Several light-mode slots
are below 3:1 against the page, so every chart has a table view in the app.
Sentiment variants keep their model's hue and use a dashed line.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecaster.models import ENSEMBLE_KEY, NAIVE_KEY, parse_variant

# (light, dark) steps of the validated categorical palette, in slot order.
MODEL_COLORS: dict[str, tuple[str, str]] = {
    "chronos2": ("#2a78d6", "#3987e5"),
    "timestral": ("#eb6834", "#d95926"),
    "timesfm": ("#1baf7a", "#199e70"),
    "chronos_bolt": ("#eda100", "#c98500"),
    "gbm": ("#e87ba4", "#d55181"),
    ENSEMBLE_KEY: ("#008300", "#0ca30c"),
}
NEUTRAL = ("#7d8496", "#8f97ab")  # random-walk reference
INK = ("#161b2e", "#e7eaf3")  # observed prices

# Streamlit number format: follows the viewer's locale (id-ID shows 9.850).
PRICE_FORMAT = "localized"


def _pick(pair: tuple[str, str], mode: str) -> str:
    if mode not in ("light", "dark"):
        raise ValueError(f"mode must be light or dark, got {mode!r}")
    return pair[0] if mode == "light" else pair[1]


def model_color(key: str, mode: str) -> str:
    base, _ = parse_variant(key)
    if base == NAIVE_KEY:
        return _pick(NEUTRAL, mode)
    return _pick(MODEL_COLORS[base], mode)


def _line_type(key: str) -> str:
    base, with_sent = parse_variant(key)
    if base == NAIVE_KEY or with_sent:
        return "dashed"
    return "solid"


def fan_chart_option(
    history: pd.Series,
    focus_key: str,
    focus_label: str,
    focus: pd.DataFrame,
    compare: dict[str, tuple[str, pd.DataFrame]],
    level: float,
    mode: str,
) -> dict[str, Any]:
    """Recent closes, the focus model's median with its interval band, other medians.

    history: Close indexed by date. focus: columns date, median, lower, upper.
    compare: key -> (label, frame with date, median). Prices are rounded to whole
    Rupiah for display. The band is a translucent area stacked on a hidden line at
    the lower bound (ECharts cannot fill between two lines).
    """
    hist = pd.DataFrame(
        {"date": [d.strftime("%Y-%m-%d") for d in history.index], "Harga aktual": history.to_numpy()}
    )
    fut = pd.DataFrame({"date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in focus["date"]]})
    fut["median"] = focus["median"].to_numpy()
    fut["lower"] = focus["lower"].to_numpy()
    fut["upper"] = focus["upper"].to_numpy()
    data = pd.concat([hist, fut], ignore_index=True)
    last = len(hist) - 1
    last_price = float(hist["Harga aktual"].iloc[-1])
    # Start every forecast line at the last close so the lines join the history.
    for column in ("median", "lower", "upper"):
        data.loc[last, column] = last_price
    compare_columns: list[tuple[str, str, str]] = []
    for i, (key, (label, frame)) in enumerate(compare.items()):
        column = f"cmp_{i}"
        values = dict(
            zip([pd.Timestamp(d).strftime("%Y-%m-%d") for d in frame["date"]], frame["median"])
        )
        data[column] = data["date"].map(values)
        data.loc[last, column] = last_price
        compare_columns.append((key, label, column))
    price_columns = [c for c in data.columns if c != "date"]
    data[price_columns] = data[price_columns].round(0)
    # Band width from the rounded bounds, so lower + band equals the displayed upper.
    data["band"] = data["upper"] - data["lower"]

    color = model_color(focus_key, mode)
    focus_name = f"{focus_label} (interval {level:.0%})"
    band_base = {
        "type": "line",
        "name": focus_name,
        "stack": "band",
        "stackStrategy": "all",
        "showSymbol": False,
        "silent": True,
        "tooltip": {"show": False},
        "lineStyle": {"width": 0},
        "itemStyle": {"color": color},
    }
    series: list[dict[str, Any]] = [
        {
            "type": "line",
            "name": "Harga aktual",
            "encode": {"x": "date", "y": "Harga aktual"},
            "showSymbol": False,
            "lineStyle": {"width": 2, "color": _pick(INK, mode)},
            "itemStyle": {"color": _pick(INK, mode)},
            "markLine": {
                "silent": True,
                "symbol": "none",
                "label": {"formatter": "Origin"},
                "data": [{"xAxis": hist["date"].iloc[-1]}],
            },
        },
        {**band_base, "encode": {"x": "date", "y": "lower"}},
        {
            **band_base,
            "encode": {"x": "date", "y": "band"},
            "areaStyle": {"opacity": 0.2, "color": color},
        },
        {
            "type": "line",
            "name": focus_name,
            "encode": {"x": "date", "y": "median", "tooltip": ["upper", "median", "lower"]},
            "showSymbol": True,
            "symbolSize": 8,
            "lineStyle": {"width": 2, "color": color, "type": _line_type(focus_key)},
            "itemStyle": {"color": color},
        },
    ]
    for key, label, column in compare_columns:
        c = model_color(key, mode)
        series.append(
            {
                "type": "line",
                "name": label,
                "encode": {"x": "date", "y": column},
                "showSymbol": False,
                "lineStyle": {"width": 2, "color": c, "type": _line_type(key)},
                "itemStyle": {"color": c},
            }
        )
    legend = ["Harga aktual", focus_name, *[label for _, label, _ in compare_columns]]
    return {
        "dataset": {"source": data},
        "legend": {"data": legend, "top": 0},
        "grid": {"top": 40, "left": 8, "right": 16, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "valueFormatter": PRICE_FORMAT},
        "xAxis": {"type": "category", "boundaryGap": False},
        "yAxis": {"type": "value", "scale": True, "axisLabel": {"formatter": PRICE_FORMAT}},
        "series": series,
    }


def skill_chart_option(
    rel_mae: pd.DataFrame, labels: dict[str, str], mode: str
) -> dict[str, Any]:
    """Relative MAE (model / random walk) by horizon; below 1 beats the random walk.

    rel_mae: index = model keys, columns = horizons 1..H.
    """
    data = pd.DataFrame({"horizon": [f"h+{h}" for h in rel_mae.columns]})
    series = []
    for i, key in enumerate(rel_mae.index):
        column = labels[key]
        data[column] = np.round(rel_mae.loc[key].to_numpy(dtype=float), 4)
        c = model_color(key, mode)
        entry: dict[str, Any] = {
            "type": "line",
            "name": column,
            "encode": {"x": "horizon", "y": column},
            "symbolSize": 8,
            "lineStyle": {"width": 2, "color": c, "type": _line_type(key)},
            "itemStyle": {"color": c},
        }
        if i == 0:
            entry["markLine"] = {
                "silent": True,
                "symbol": "none",
                "label": {"formatter": "Random walk = 1"},
                "data": [{"yAxis": 1}],
            }
        series.append(entry)
    return {
        "dataset": {"source": data},
        "legend": {"top": 0},
        "grid": {"top": 40, "left": 8, "right": 16, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.3f"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value", "scale": True},
        "series": series,
    }


def tone_chart_option(features: pd.DataFrame, mode: str) -> dict[str, Any]:
    """Daily GDELT tone per trading day and its exponentially weighted mean."""
    data = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in features.index],
            "Tone harian": features["sent_tone"].round(3).to_numpy(),
            "Tone (EWM)": features["sent_tone_ewm"].round(3).to_numpy(),
        }
    )
    ink, neutral = _pick(INK, mode), _pick(NEUTRAL, mode)
    return {
        "dataset": {"source": data},
        "legend": {"top": 0},
        "grid": {"top": 40, "left": 8, "right": 16, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "valueFormatter": "%.2f"},
        "xAxis": {"type": "category", "boundaryGap": False},
        "yAxis": {"type": "value", "scale": True},
        "series": [
            {
                "type": "line",
                "name": "Tone harian",
                "encode": {"x": "date", "y": "Tone harian"},
                "showSymbol": False,
                "connectNulls": False,
                "lineStyle": {"width": 1, "color": neutral},
                "itemStyle": {"color": neutral},
            },
            {
                "type": "line",
                "name": "Tone (EWM)",
                "encode": {"x": "date", "y": "Tone (EWM)"},
                "showSymbol": False,
                "lineStyle": {"width": 2, "color": ink},
                "itemStyle": {"color": ink},
                "markLine": {"silent": True, "symbol": "none", "data": [{"yAxis": 0}]},
            },
        ],
    }
