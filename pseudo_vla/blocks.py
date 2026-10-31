"""Shared definitions for the three stackable scene blocks."""

from __future__ import annotations

BLOCK_SIZE = 0.05
BLOCK_COLORS = ("red", "green", "blue")
BLOCK_SPECS = (
    {"id": "red", "name": "red_block", "color": (0.9, 0.05, 0.05), "xy": (-0.22, -0.28)},
    {"id": "green", "name": "green_block", "color": (0.05, 0.8, 0.05), "xy": (0.0, -0.28)},
    {"id": "blue", "name": "blue_block", "color": (0.05, 0.15, 0.95), "xy": (0.22, -0.28)},
)
BLOCKS_BY_COLOR = {spec["name"].removesuffix("_block"): spec for spec in BLOCK_SPECS}
