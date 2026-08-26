from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / "figs"
CONFIG_PATH = FIGS_DIR / "paper_figure_config.json"

_FONT_RC_KEYS = {
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.titlesize",
}


def _merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = dict(base[key])
            base[key] = _merge(nested, value)
        else:
            base[key] = value
    return base


def _lookup(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


@lru_cache(maxsize=1)
def load_style_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def reload_style_config() -> dict[str, Any]:
    load_style_config.cache_clear()
    return load_style_config()


def get_figure_config(figure_id: str | None = None, group: str | None = None) -> dict[str, Any]:
    config = load_style_config()
    merged: dict[str, Any] = {}
    defaults = config.get("defaults", {})
    if isinstance(defaults, Mapping):
        _merge(merged, defaults)
    groups = config.get("groups", {})
    if group and isinstance(groups, Mapping):
        group_cfg = groups.get(group, {})
        if isinstance(group_cfg, Mapping):
            _merge(merged, group_cfg)
    figures = config.get("figures", {})
    if figure_id and isinstance(figures, Mapping):
        figure_cfg = figures.get(figure_id, {})
        if isinstance(figure_cfg, Mapping):
            _merge(merged, figure_cfg)
    return merged


def style_value(
    dotted_key: str,
    default: Any = None,
    *,
    figure_id: str | None = None,
    group: str | None = None,
) -> Any:
    return _lookup(get_figure_config(figure_id=figure_id, group=group), dotted_key, default)


def figure_text(
    key: str,
    default: str,
    *,
    figure_id: str | None = None,
    group: str | None = None,
) -> str:
    value = style_value(f"titles.{key}", default, figure_id=figure_id, group=group)
    return str(value) if value is not None else default


def figure_label(
    key: str,
    default: str,
    *,
    figure_id: str | None = None,
    group: str | None = None,
) -> str:
    value = style_value(f"labels.{key}", default, figure_id=figure_id, group=group)
    return str(value) if value is not None else default


def figure_color(
    key: str,
    default: str,
    *,
    figure_id: str | None = None,
    group: str | None = None,
) -> str:
    value = style_value(f"colors.{key}", default, figure_id=figure_id, group=group)
    return str(value) if value is not None else default


def figure_size(
    figure_id: str,
    default: Sequence[float],
    *,
    group: str | None = None,
) -> tuple[float, ...]:
    value = style_value("figsize", default, figure_id=figure_id, group=group)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(v) for v in value)
    return tuple(float(v) for v in default)


def mapping_with_overrides(
    mapping: Mapping[str, Any],
    *,
    dotted_key: str,
    figure_id: str | None = None,
    group: str | None = None,
) -> dict[str, Any]:
    merged = dict(mapping)
    overrides = style_value(dotted_key, {}, figure_id=figure_id, group=group)
    if isinstance(overrides, Mapping):
        merged.update(overrides)
    return merged


def apply_matplotlib_style(
    plt: Any,
    base_rc: Mapping[str, Any],
    *,
    figure_id: str | None = None,
    group: str | None = None,
) -> dict[str, Any]:
    rc = dict(base_rc)
    scale = style_value("font_scale", 1.0, figure_id=figure_id, group=group)
    if isinstance(scale, (int, float)) and scale != 1:
        for key in _FONT_RC_KEYS:
            value = rc.get(key)
            if isinstance(value, (int, float)):
                rc[key] = value * float(scale)
    extra_rc = style_value("rcParams", {}, figure_id=figure_id, group=group)
    if isinstance(extra_rc, Mapping):
        rc.update(extra_rc)
    plt.rcParams.update(rc)
    return rc
