#!/usr/bin/env python3
"""Internal debug utilities for testing error handling.

DO NOT USE IN PRODUCTION - intentionally breaks output format.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, TypedDict


class DebugConfig(TypedDict, total=False):
    """Configuration for debug failure injection.

    Attributes:
        fail_mode: Type of failure to inject (malformed-markers, missing-markers)
        fail_step: Target step to inject failure (None = inject on all steps)
    """

    fail_mode: str
    fail_step: int | None


class DebugFailMode(StrEnum):
    """Types of intentional failures for testing error handling."""

    MALFORMED_MARKERS = "malformed-markers"  # Break field marker syntax
    MISSING_MARKERS = "missing-markers"  # Remove closing markers


def inject_malformed_instructions(
    field_parts: list[str],
    fail_mode: str,
    current_step: int | None = None,
    target_step: int | None = None,
) -> list[str]:
    """Inject intentionally broken instructions based on fail mode and step."""
    should_inject = target_step is None or (current_step is not None and current_step == target_step)
    if not should_inject:
        return field_parts

    fail_mode_enum = DebugFailMode(fail_mode)
    if fail_mode_enum == DebugFailMode.MALFORMED_MARKERS:
        return _break_field_markers(field_parts)
    elif fail_mode_enum == DebugFailMode.MISSING_MARKERS:
        return _remove_closing_markers(field_parts)

    return field_parts


def _break_field_markers(field_parts: list[str]) -> list[str]:
    """Transforms `[[ ## field_name ## ]]` → `[[ ## field_name ]]`"""
    return [re.sub(r"## \]\]", "]]", part) for part in field_parts]


def _remove_closing_markers(field_parts: list[str]) -> list[str]:
    """Transforms `[[ ## field_name ## ]]` → `[[ ## field_name`"""
    return [re.sub(r" ## \]\]", "", part) for part in field_parts]


def inject_debug_field_parts(
    field_parts: list[str],
    debug_config: dict[str, Any],
    current_step: int | None,
) -> str:
    """Inject debug failures into field parts and return formatted string with notice."""
    original_parts = field_parts.copy()
    field_parts = inject_malformed_instructions(
        field_parts,
        debug_config["fail_mode"],
        current_step=current_step,
        target_step=debug_config.get("fail_step"),
    )

    fields_str = ", then ".join(field_parts)
    if field_parts != original_parts:
        return (
            f"{fields_str} [DEBUG: Instructions are intentionally "
            f"asked in malformed format (without the field's closing ##) for error "
            f"handling test. Make sure to write the next field definitions in this breaking format.]"
        )
    return fields_str
