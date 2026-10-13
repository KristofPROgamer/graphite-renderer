from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import time
import json

import numpy as np

from graphite.core.io import load_image, save_png
from graphite.render.canvas import GraphiteCanvas
from graphite.render.blend import tortillon_blend
from graphite.render.erase import ghost_erase, kneaded_erase
from graphite.render.finish import finish_canvas
from graphite.perception.bundle import analyze_all
from graphite.plan.build import build_plan
from graphite.strokes.construction import generate_construction_strokes
from graphite.strokes.contour_strokes import generate_contour_strokes
from graphite.strokes.hatching import generate_hatching_strokes
from graphite.strokes.stipple import stipple
from graphite.strokes.textures.registry import generate_texture_strokes


class DrawingResult:
    def __init__(self, canvas: GraphiteCanvas, timings: Dict[str, float], artifacts: Dict[str, str]):
        self.canvas = canvas
        self.timings = timings
        self.artifacts = artifacts


class DrawingPipeline:
    """Orchestrate the full graphite drawing pipeline: perception -> plan ->
    stroke generation/rasterization -> blend/erase -> finish -> save.

    Every stage is now real. Perception (graphite.perception.bundle.analyze_all),
    plan (graphite.plan.build.build_plan), and stroke generation/rasterization
    onto the canvas were wired in Step 6a. Blend/erase (Module 17) is wired in
    Step 6b to the real tortillon_blend/ghost_erase/kneaded_erase passes.
    Finish and save were already real (the deposit->reflectance bridge from
    Step 4), and Step 6b additionally dumps every canvas layer's final state
    as a debug PNG for inspection.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.out_dir = Path(cfg.get("out_dir", "out"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.out_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str) -> Path:
        return self.out_dir / f"{name}.json"

    def run(self, img_path: str, *, seed: Optional[int] = None) -> DrawingResult:
        timings: Dict[str, float] = {}
        artifacts: Dict[str, str] = {}

        start = time.time()
        img = load_image(img_path, target_long_side=self.cfg.get("working_res", None))
        timings["load_image"] = time.time() - start

        # Create canvas
        start = time.time()
        h, w = img.shape[:2]
        canvas = GraphiteCanvas((h, w), seed=(seed or self.cfg.get("seed", 0)), paper_preset=self.cfg.get("paper_preset", "cold_press"))
        timings["make_canvas"] = time.time() - start

        # Stage: perception (Modules 6-8)
        # Runs the real graphite.perception stack (segmentation, human
        # parsing, face landmarks, depth/normals, edges/lines/vanishing
        # points, saliency, light) and fuses it into one bundle dict.
        # `img_srgb` is added on top since graphite.plan.build.build_plan
        # requires it.
        start = time.time()
        bundle = dict(analyze_all(img))
        bundle["img_srgb"] = img
        timings["perception"] = time.time() - start

        # Stage: plan (Modules 9-11)
        # Builds the real DrawingPlan: tone target + bands, importance,
        # orientation field, and fitted contour chains (see
        # graphite.plan.build.build_plan). `cfg["plan"]` may supply the
        # optional sub-configs build_plan understands (K, importance,
        # orientation, contours); omit any of them to use that stage's own
        # defaults.
        start = time.time()
        plan = build_plan(bundle, dict(self.cfg.get("plan", {})))
        timings["plan"] = time.time() - start

        # Stage: strokes generation & rasterization (Modules 12-16, 2)
        # Generates real Stroke objects from the plan and renders each onto
        # its intended canvas layer. Several generators below build their
        # strokes through the "generic" style profile, whose layer_set is
        # ("L1_tone",) regardless of the generator's actual intent -- so
        # canvas.add_stroke's `layer=` argument is passed explicitly for
        # those, rather than trusting stroke.layer (see
        # graphite.strokes.styles.default_style_profiles).
        start = time.time()
        stroke_rng = np.random.default_rng(canvas.seed)

        # L0 construction: a light, loose underdrawing gesture. Only makes
        # sense for silhouette-class chains (the main proportion/outline
        # edges) -- crease/tonal/texture chains aren't the kind of edge an
        # artist blocks in with construction lines.
        for chain in plan.contours:
            if chain.cls != "silhouette":
                continue
            for s in generate_construction_strokes(chain, rng=stroke_rng):
                canvas.add_stroke(s, layer="L0_construction")

        # L2 contour: the inked/graphite line pass, one per detected chain.
        for chain in plan.contours:
            for s in generate_contour_strokes(chain, rng=stroke_rng):
                canvas.add_stroke(s, layer="L2_contour")

        # L1 tone: orientation-aligned hatching driven by the tone target T,
        # local axial orientation theta, and importance weight W.
        for s in generate_hatching_strokes(plan.T, plan.theta, W=plan.W, rng=stroke_rng):
            canvas.add_stroke(s, layer="L1_tone")

        # stipple: a dot texture over the darkest shadow bands, if the plan
        # identified any. plan.shadow_shapes is currently an all-zero
        # placeholder until a later module computes it for real, so this is
        # commonly a no-op today -- kept here so it activates automatically
        # once that upstream piece is wired up.
        shadow_mask = plan.shadow_shapes > 0
        if np.any(shadow_mask):
            for s in stipple(shadow_mask, None, 800, rng=stroke_rng):
                canvas.add_stroke(s, layer="stipple")

        # Per-region material texture (+ whichever layer each material's own
        # generator prefers): dispatch each semantic region to its matching
        # texture generator. Unlike construction/contour above, these
        # generators already set the correct .layer on every Stroke they
        # return (e.g. hair/foliage -> L3_texture, skin/sky/water ->
        # L1_tone, stone -> stipple, cloth/manmade -> L2_contour), so trust
        # stroke.layer here instead of overriding it.
        region_plan = {"theta": plan.theta, "W": plan.W}
        for region in plan.regions.values():
            for s in generate_texture_strokes(region, region_plan, rng=stroke_rng):
                canvas.add_stroke(s)

        # Save a mid-pipeline debug snapshot of raw tone deposit (before
        # blend/erase/finish) as a quick sanity check.
        debug_path = canvas.save_debug(self.debug_dir, layer="L1_tone")
        artifacts["L1_tone_debug"] = debug_path
        timings["rasterize"] = time.time() - start

        # Stage: blending/erase (Module 17)
        # Mirrors how a graphite drawing is actually finished by hand: blend
        # the tone with a stump/tortillon, lift back any reserved highlights
        # the eraser should protect, then fade the construction underdrawing
        # to a faint ghost now that the real strokes are in place. All of
        # this must run before the deposit->reflectance bridge below, since
        # these passes mutate deposit layers in place and reflectance() reads
        # every layer's *final* state.
        start = time.time()

        # Tortillon blend: soften the hatched tone layer, smearing gently
        # along the local orientation field rather than isotropically.
        # `blend_amount` is the Module 17 blend-strength knob documented on
        # graphite.core.config.Config (default 0.5, same default as
        # configs/default.yaml) -- reused here even though this pipeline's
        # cfg is a plain dict rather than a validated Config.
        tortillon_blend(canvas, theta=plan.theta, layer="L1_tone", strength=self.cfg.get("blend_amount", 0.5))

        # Kneaded erase: lift the reserved-white regions (catchlights,
        # highlights) back toward bare paper. plan.reserves is an int label
        # map (0 = not reserved; see graphite.plan.reserves.find_reserved_whites),
        # so treat any nonzero label as "reserved". Like plan.shadow_shapes in
        # Step 6a, plan.reserves is currently an all-zero placeholder until a
        # later module wires it up for real, so this is commonly a no-op today
        # -- kept here so it activates automatically once that piece lands.
        reserved_mask = plan.reserves > 0
        if np.any(reserved_mask):
            kneaded_erase(canvas, layer="L1_tone", mask=reserved_mask, strength=self.cfg.get("highlight_lift", 0.7))

        # Ghost erase: fade the L0 construction/underdrawing gesture lines to
        # a faint residual now that the real contour/tone/texture strokes are
        # down, the way an artist lightens (rather than fully erases) their
        # initial sketch.
        ghost_erase(canvas, source_layer="L0_construction", target_layer="L0_construction", keep_fraction=self.cfg.get("construction_fade", 0.45))

        timings["blend"] = time.time() - start

        # Stage: finish (Module 18)
        # GraphiteCanvas layers store DEPOSIT (1 = full black ink), but
        # paper_finish()/finish_canvas() blend their input layer *toward* a
        # bright paper_base (see graphite/render/finish.py) -- i.e. they
        # expect REFLECTANCE (1 = white paper) as input. Feeding deposit
        # straight in (e.g. copying L1_tone into L4_blend, as this stage used
        # to) inverts the drawing: heavy ink gets pulled toward white and
        # blank paper gets pulled toward black. canvas.reflectance() bridges
        # this correctly -- it saturate-composites every deposit layer
        # (L0_construction, L1_tone, L2_contour, L3_texture, stipple) via
        # GraphiteCanvas._composite() and returns 1-composite, so this must
        # run *after* every stroke layer (including the blend/erase mutation
        # of them above) is in its final state.
        start = time.time()
        canvas.layers["L4_blend"] = canvas.reflectance()
        finished = finish_canvas(canvas, layer="L4_blend", output_dtype=np.float32, preset=self.cfg.get("paper_preset", None), strength=self.cfg.get("finish_strength", 1.0), vignette=self.cfg.get("vignette", None), scanner_noise=self.cfg.get("scanner_noise", 0.01))
        timings["finish"] = time.time() - start

        # Save outputs
        start = time.time()
        out_png = self.out_dir / "result.png"
        # Ensure grayscale for 16-bit PNG output (module 18 expects grayscale finish)
        if finished.ndim == 3 and finished.shape[2] == 3:
            finished_to_save = np.mean(finished, axis=-1)
        else:
            finished_to_save = finished
        save_png(out_png, finished_to_save, bit_depth=16)
        artifacts["result_png"] = str(out_png)

        # Per-layer debug snapshots: dump every canvas layer (construction,
        # tone, contour, texture, stipple, the composited L4_blend
        # reflectance, and the finished paper_finish layer) in its final,
        # post blend/erase/finish state. Written to a "final" subfolder so
        # they don't clobber the raw pre-blend L1_tone_debug snapshot saved
        # earlier -- the two together let a person see exactly what each
        # stage contributed, and are what the constants in
        # graphite/render/graphite.py::GRADE_TABLE get tuned against.
        final_debug_dir = self.debug_dir / "final"
        for layer_name in sorted(canvas.layers):
            layer_debug_path = canvas.save_debug(final_debug_dir, layer=layer_name)
            artifacts[f"final_{layer_name}_debug"] = layer_debug_path

        timings["save"] = time.time() - start

        # Save timing JSON
        timings_path = self.out_dir / "timings.json"
        with open(timings_path, "w", encoding="utf8") as f:
            json.dump(timings, f, indent=2)
        artifacts["timings_json"] = str(timings_path)

        return DrawingResult(canvas, timings, artifacts)


__all__ = ["DrawingPipeline", "DrawingResult"]
