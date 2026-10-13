from __future__ import annotations

from .cloth import generate_cloth_strokes
from .foliage import generate_foliage_strokes
from .hair import generate_hair_strokes
from .manmade import generate_manmade_strokes
from .registry import TEXTURE_GENERATORS, generate_texture_strokes, get_texture_generator
from .skin import generate_skin_strokes
from .sky_water import generate_sky_water_strokes
from .stone import generate_stone_strokes

__all__ = [
    "TEXTURE_GENERATORS",
    "generate_texture_strokes",
    "get_texture_generator",
    "generate_hair_strokes",
    "generate_foliage_strokes",
    "generate_skin_strokes",
    "generate_cloth_strokes",
    "generate_sky_water_strokes",
    "generate_stone_strokes",
    "generate_manmade_strokes",
]
