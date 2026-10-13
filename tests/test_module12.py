import numpy as np

from graphite.strokes.hand import (
    NibScheduler,
    StyleProfile,
    break_into_hand_strokes,
    make_stroke,
    restate,
)


def _ideal_path(length=80):
    x = np.linspace(0.0, 120.0, length, dtype=np.float32)
    y = 20.0 * np.sin(x / 18.0)
    return np.column_stack([x, y]).astype(np.float32)


def test_make_stroke_and_style_profile():
    style = StyleProfile(
        name="test",
        kind="hatch",
        grade="HB",
        width=5.5,
        pressure=0.6,
        jitter=1.2,
        curvature=0.08,
        spacing_multiplier=1.0,
        blend_amount=0.3,
        layer_set=("L1_tone",),
    )
    rng = np.random.default_rng(7)
    stroke = make_stroke(_ideal_path(), kind="hatch", style=style, rng=rng)
    stroke.validate()
    assert stroke.pts.shape[1] == 2
    assert stroke.width.shape[0] == stroke.pts.shape[0]
    assert stroke.pressure.shape[0] == stroke.pts.shape[0]
    assert stroke.footprint in {"point", "side"}
    assert stroke.grade in {"2H", "H", "HB", "B", "2B", "4B", "6B", "8B"}


def test_same_seed_same_stroke_stream():
    style = StyleProfile(name="test", kind="hatch", grade="2B", width=4.0, pressure=0.5, jitter=1.0, curvature=0.1, spacing_multiplier=1.0, blend_amount=0.2, layer_set=("L1_tone",))
    p0 = _ideal_path()
    s1 = make_stroke(p0, kind="hatch", style=style, rng=np.random.default_rng(123))
    s2 = make_stroke(p0, kind="hatch", style=style, rng=np.random.default_rng(123))
    assert np.allclose(s1.pts, s2.pts)
    assert np.allclose(s1.width, s2.width)
    assert np.allclose(s1.pressure, s2.pressure)


def test_break_into_hand_strokes_and_restate():
    chain = np.array([
        [0.0, 0.0],
        [10.0, 5.0],
        [20.0, 8.0],
        [30.0, 6.0],
        [40.0, 2.0],
        [50.0, 0.0],
    ], dtype=np.float32)
    strokes = break_into_hand_strokes(chain, max_frac=0.2)
    assert len(strokes) >= 1
    restated = restate(strokes[0], np.random.default_rng(5))
    restated.validate()
    assert restated.pts.shape[1] == 2


def test_nib_scheduler_tracks_sharpen_events():
    sched = NibScheduler(initial_width=5.0, wear_rate=0.14)
    assert sched.current_width > 0
    schedule = [sched.advance(distance=8.0) for _ in range(4)]
    assert len(schedule) == 4
    assert all(w > 0 for w in schedule)
