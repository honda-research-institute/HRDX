"""Shared constants and helpers for RDX map category & attribute handling."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Base class schema
# ---------------------------------------------------------------------------
BASE_CLASSES: List[str] = [
    'lane_line',
    'stop_line',
    'road_boundary',
    'lane_centerline',
    'intersection_centerline',
    'text_on_road',
    'non_drivable_area',
    'parking_spot',
    'crosswalk',
    'bike_lane',
    #'blocked_area',
]

CAT2ID: Dict[str, int] = {name: idx for idx, name in enumerate(BASE_CLASSES)}
ID2CAT: Dict[int, str] = {idx: name for name, idx in CAT2ID.items()}

# ---------------------------------------------------------------------------
# Attribute schema (canonical class names & defaults)
# ---------------------------------------------------------------------------
TEXT_ON_ROAD_CLASSES: List[str] = [
    'left_turn_arrow',
    'straight_arrow',
    'other_class',
    'merge_advisory_arrow',
    'only',
    'stop_text',
    'yield_ahead',
    'ped_xing',
    'keep_clear',
    'rxr',
    'turn_right_arrow',
    'bus_taxi_lane',
    'hov_lane',
    'other_text',
]

ATTRIBUTE_SCHEMAS: Dict[str, Dict[str, object]] = {
    'lane_color': dict(
        classes=['white', 'yellow'],
        default='white',
        applies_to=['lane_line'],
    ),
    'lane_style': dict(
        classes=['solid', 'dashed'],
        default='dashed',
        applies_to=['lane_line'],
    ),
    'intersection_movement': dict(
        classes=['straight', 'left', 'right', 'u_turn'],
        default='straight',
        applies_to=['intersection_centerline'],
    ),
    'text_type': dict(
        classes=TEXT_ON_ROAD_CLASSES,
        default='left_turn_arrow',
        applies_to=['text_on_road'],
    ),
    'parking_status': dict(
        classes=['empty', 'not_empty'],
        default='empty',
        applies_to=['parking_spot'],
    ),
}

ATTRIBUTE_VALUE_TO_INDEX: Dict[str, Dict[str, int]] = {
    attr: {name: idx for idx, name in enumerate(info['classes'])}
    for attr, info in ATTRIBUTE_SCHEMAS.items()
}

ATTRIBUTE_DEFAULT_INDEX: Dict[str, int] = {
    attr: ATTRIBUTE_VALUE_TO_INDEX[attr][info['default']]  # type: ignore[index]
    for attr, info in ATTRIBUTE_SCHEMAS.items()
}

ATTRIBUTE_APPLIES_TO_IDS: Dict[str, List[int]] = {
    attr: [CAT2ID[name] for name in info['applies_to']]  # type: ignore[index]
    for attr, info in ATTRIBUTE_SCHEMAS.items()
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
_TEXT_ALIAS_TO_CANONICAL: Dict[str, str] = {
    'left turn arrow': 'left_turn_arrow',
    'straight arrow': 'straight_arrow',
    'other class': 'other_class',
    'merge advisory arrow': 'merge_advisory_arrow',
    'only': 'only',
    'stop text': 'stop_text',
    'yield (ahead)': 'yield_ahead',
    'yield ahead': 'yield_ahead',
    'ped xing': 'ped_xing',
    'keep clear': 'keep_clear',
    'rxr': 'rxr',
    'turn right arrow': 'turn_right_arrow',
    'bus/taxi lane': 'bus_taxi_lane',
    'bus taxi lane': 'bus_taxi_lane',
    'hov lane': 'hov_lane',
    # Allow already-canonical values
    'left_turn_arrow': 'left_turn_arrow',
    'straight_arrow': 'straight_arrow',
    'other_class': 'other_class',
    'merge_advisory_arrow': 'merge_advisory_arrow',
    'stop_text': 'stop_text',
    'yield_ahead': 'yield_ahead',
    'ped_xing': 'ped_xing',
    'keep_clear': 'keep_clear',
    'turn_right_arrow': 'turn_right_arrow',
    'bus_taxi_lane': 'bus_taxi_lane',
    'hov_lane': 'hov_lane',
}


def _safe_get_attr(attributes: Dict[str, object], key: str) -> Optional[object]:
    for attr_key, value in attributes.items():
        if attr_key.lower() == key.lower():
            return value
    return None


def _ensure_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [str(value)]
    return [str(value)]


def _normalize_lane_style(raw: Optional[object]) -> Tuple[str, Optional[str]]:
    """Returns (style, override_class) where override_class is 'stop_line' if detected."""
    if raw is None:
        return 'dashed', None
    val = str(raw).strip().lower()
    if 'stop' in val:
        return 'dashed', 'stop_line'
    if 'solid' in val:
        return 'solid', None
    if 'dash' in val:
        return 'dashed', None
    return 'dashed', None


def _normalize_lane_color(raw: Optional[object]) -> str:
    if raw is None:
        return 'white'
    val = str(raw).strip().lower()
    if val == 'white':
        return 'white'
    if val in ('yellow', 'other'):
        return 'yellow'
    return 'white'


def _normalize_movement(raw_values: List[str]) -> str:
    for value in raw_values:
        val = value.strip().lower()
        if 'straight' in val:
            return 'straight'
        if val.startswith('left'):
            return 'left'
        if val.startswith('right'):
            return 'right'
        if 'u' in val:
            return 'u_turn'
    return 'straight'


def _normalize_text_token(raw_values: List[str]) -> str:
    if not raw_values:
        return 'left_turn_arrow'
    candidate = raw_values[0].strip().lower()
    canonical = _TEXT_ALIAS_TO_CANONICAL.get(candidate)
    if canonical:
        return canonical
    return 'other_text'


def _normalize_parking_status(raw: Optional[object]) -> str:
    if raw is None:
        return 'empty'
    val = str(raw).strip().lower()
    if val in ('not empty', 'occupied', 'not_empty'):
        return 'not_empty'
    return 'empty'


def normalize_category_and_attributes(
    category: str,
    attributes: Optional[Dict[str, object]],
) -> Tuple[Optional[str], Dict[str, str]]:
    """Normalize raw RDX annotations into base class & canonical attributes.

    Args:
        category: Raw category name from the annotation JSON.
        attributes: Raw attributes dict for the shape.

    Returns:
        (base_class_name, attr_dict). base_class_name is ``None`` if the instance
        should be skipped. ``attr_dict`` maps attribute keys (e.g. ``lane_color``)
        to canonical string labels defined in :data:`ATTRIBUTE_SCHEMAS`.
    """
    if not category:
        return None, {}

    attr_dict: Dict[str, str] = {}
    attrs = attributes if isinstance(attributes, dict) else {}
    cat_key = category.strip().lower()

    if cat_key == 'lane lines':
        lane_type = _safe_get_attr(attrs, 'Type')
        lane_color = _safe_get_attr(attrs, 'Color')

        style, override_class = _normalize_lane_style(lane_type)
        if override_class == 'stop_line':
            return 'stop_line', {}

        color = _normalize_lane_color(lane_color)
        attr_dict['lane_style'] = style
        attr_dict['lane_color'] = color
        return 'lane_line', attr_dict

    if cat_key == 'road boundary':
        return 'road_boundary', attr_dict

    if cat_key == 'lane center lines':
        subcat_raw = _safe_get_attr(attrs, 'category')
        subcat = str(subcat_raw).strip().lower() if subcat_raw is not None else ''
        if subcat == 'intersection centerline':
            movement_raw = _safe_get_attr(attrs, 'attribute')
            movement = _normalize_movement(_ensure_list(movement_raw))
            attr_dict['intersection_movement'] = movement
            return 'intersection_centerline', attr_dict
        return 'lane_centerline', attr_dict

    if cat_key == 'text on the road':
        text_raw = _safe_get_attr(attrs, 'Text on the road')
        text_label = _normalize_text_token(_ensure_list(text_raw))
        attr_dict['text_type'] = text_label
        return 'text_on_road', attr_dict

    if cat_key == 'non-drivable areas':
        return 'non_drivable_area', attr_dict

    if cat_key == 'parking spots':
        status_raw = _safe_get_attr(attrs, 'attribute')
        status = _normalize_parking_status(status_raw)
        attr_dict['parking_status'] = status
        return 'parking_spot', attr_dict

    if cat_key == 'crosswalks':
        return 'crosswalk', attr_dict

    if cat_key == 'bike lanes':
        return 'bike_lane', attr_dict

    if cat_key in ('施工区域 blocked areas', 'blocked areas', 'blocked area'):
        return 'blocked_area', attr_dict

    if cat_key == 'lane centerline':
        # Some data sources may already use normalized naming.
        return 'lane_centerline', attr_dict

    if cat_key == 'intersection centerline':
        movement_raw = _safe_get_attr(attrs, 'attribute')
        movement = _normalize_movement(_ensure_list(movement_raw))
        attr_dict['intersection_movement'] = movement
        return 'intersection_centerline', attr_dict

    if cat_key == 'stop lines':
        return 'stop_line', attr_dict

    # Unknown category – skip.
    return None, {}
