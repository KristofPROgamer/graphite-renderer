# Photo → Graphite Drawing Pipeline: Modular Build Plan

## How to use this file

1. For each module, start a fresh chat with your coding AI (fresh chats are more reliable than one giant thread).
2. Paste, in this order:
   - the **Prompt Wrapper** (below)
   - **Part 0: Global Context** (below)
   - the **Interface Snapshots** of every module already built (none for Module 1)
   - the single **Module N** section you want built
3. Say: `Create Module N.`
4. When it finishes, say: `Output the Interface Snapshot for Module N` (public signatures, dataclasses, file paths, one-line docstrings only). Save it. It goes into every later prompt.
5. Run the module's acceptance tests. Do not start a module until its dependencies pass.

### Prompt Wrapper (paste at the top every time)

```
You are a senior Python engineer building ONE module of a larger project.
Rules:
- Implement ONLY the module I name. If you need another module's function, use the signature from the
  Interface Snapshots I pasted. If none exists, write a minimal stub marked `# STUB(module N)`.
- Deliver complete files with repo paths, pytest tests, and a runnable demo script that writes
  debug images to `out/debug/`.
- Follow the Global Context conventions exactly (array shapes, dtypes, angle conventions, seeding).
- No generative models anywhere. Neural networks are perception-only and never output pixels of the drawing.
- Prefer clear, vectorized NumPy; use Numba for hot loops. Type hints and docstrings on all public functions.
- Finish with: (1) the acceptance checklist and whether each item is met, (2) known limitations,
  (3) the Interface Snapshot for this module.
```

---

# Part 0: Global Context (paste with every module)

## Goal
Convert a photo into an image that looks hand-drawn with a graphite pencil. This is **not** image generation, not a LoRA, not a diffusion/GAN model, and not a per-pixel "pencil filter". Neural networks are used only for **perception** (they output maps, masks, landmarks, vector primitives). Every output pixel comes from a **stroke simulator** that deposits graphite onto a **paper heightfield** using a physical model. The result should read as the residue of simulated hand motion.

## Hard rules (never violate)
1. Networks emit analysis data only (masks, depth, normals, edges, landmarks, saliency, line segments). Never drawing pixels.
2. All drawing pixels come from `GraphiteCanvas` (Module 2) via strokes/dots.
3. Never run Floyd–Steinberg (or any error-diffusion dither) over the pencil render. Grain comes from paper tooth gating.
4. The reference corpus of real graphite scans is used for calibration and scoring only, never to train a generator.
5. Everything is deterministic given (image, config, seed).
6. The renderer stack (Modules 2, 12–18) must not import torch. Only perception (6–8) and refinement (21) may.

## Pipeline flow
```
photo → perception stack (6,7,8) → drawing plan (9,10,11)
      → stroke synthesis in artist order (12–16) → tooth-gated graphite rasterizer (2)
      → blending/erasing (17) → paper finish (18) → [optional] differentiable refinement (21)
```

## Tech stack
Python 3.11+, NumPy, SciPy, OpenCV, scikit-image, Shapely, Numba, PyYAML, pytest, Optuna (Module 20).
PyTorch / ONNX Runtime only in perception and refinement. Optional: Taichi for GPU stamping.

## Repo layout
```
graphite/
  core/        types.py config.py rng.py io.py color.py cache.py geometry.py debug.py
  perception/  base.py segmentation.py humans.py face.py geometry3d.py light.py lines.py edges.py saliency.py bundle.py
  plan/        tone.py importance.py reserves.py shadows.py orientation.py contours.py build.py
  strokes/     hand.py styles.py construction.py contour_strokes.py hatching.py stipple.py textures/*.py
  render/      paper.py graphite.py canvas.py calibration.py blend.py erase.py finish.py
  baseline/    lu_xu_jia.py
  eval/        corpus.py metrics.py report.py
  refine/      diff_raster.py optimize.py
  pipeline.py  cli.py
configs/       default.yaml paper_presets.yaml styles/*.yaml
assets/        paper/ calibration/ corpus/
tests/
```

## Conventions
- **Images**: `float32`, range [0,1]. RGB is `(H,W,3)`, gray/maps are `(H,W)`. Linear light vs sRGB is always stated in the variable name (`img_srgb`, `img_lin`).
- **Coordinates**: `(x, y)`, x right, y down, in **working-resolution pixels** (float). The canvas multiplies by the supersample factor `S` internally.
- **Angles**: radians. Orientation fields `theta` are **axial** (a line has no direction) in `[0, π)`, measured from +x toward +y (clockwise on screen). Orientation blending is always done on **structure tensors**, never by averaging angles.
- **Tone**: `T` is target luminance in [0,1] where 1 = paper white, 0 = darkest graphite reachable. Deposit `D` is in [0,1], reflectance `R = R_paper · (1 − D_max · D)`.
- **Depth**: `depth` in [0,1], **1 = nearest**. Normals: unit vectors, camera space, +z toward camera.
- **Importance** `W` in [0,1]: 1 = focus/detail, 0 = loose/bare paper.
- **Determinism**: one global seed. Child RNGs are derived by `rng.child(seed, layer, index)` so results don't depend on execution order.
- **Caching**: every perception output is cached on disk (`.cache/`) keyed by (model name, model version, image hash, params).
- **Config**: one YAML profile plus a global seed. Defaults in `configs/default.yaml`.

## Core data model (implemented in Module 1)
```python
@dataclass
class Stroke:
    pts: np.ndarray        # (N,2) float32, working-res px
    width: np.ndarray      # (N,) float32, nib diameter along path, working-res px
    pressure: np.ndarray   # (N,) float32 in [0,1]
    grade: str             # "2H","H","HB","B","2B","4B","6B","8B"
    layer: str             # "L0_construction","L1_tone","L2_contour","L3_texture","stipple"
    kind: str              # free tag: "hatch","silhouette","hair_lock","dab",...
    footprint: str         # "point" | "side"
    seed: int

@dataclass
class ContourChain:
    id: int
    pts: np.ndarray            # (N,2) polyline
    beziers: list              # list of (4,2) cubic segments
    kappa: np.ndarray          # (N,) curvature per point
    cls: str                   # "silhouette"|"crease"|"tonal"|"texture"
    score: float               # S = mean(P_edge) * length**gamma * (1 + depth_jump)
    mean_depth: float
    depth_jump: float
    outward_normal: np.ndarray # (N,2)
    junctions: list[int]

@dataclass
class RegionInfo:
    name: str                  # "hair","skin","cloth","foliage","sky","water","stone","manmade","fur","other"
    mask: np.ndarray           # (H,W) bool
    style: str                 # key into style profiles

@dataclass
class DrawingPlan:
    img_srgb: np.ndarray
    T: np.ndarray              # tone target (H,W)
    bands: np.ndarray          # (H,W) int band index, layer selector only
    theta: np.ndarray          # (H,W) axial orientation
    coherence: np.ndarray      # (H,W) in [0,1]
    W: np.ndarray              # importance
    contours: list[ContourChain]
    regions: dict[str, RegionInfo]
    light: dict                # {"l3d":(3,), "l2d":(2,), "ambient":float, "confidence":float}
    reserves: np.ndarray       # (H,W) int labels of reserved-white components (0 = none)
    shadow_shapes: np.ndarray  # (H,W) int labels of simplified shadow shapes
    depth: np.ndarray; normals: np.ndarray; semantic: np.ndarray
    edge_prob: np.ndarray; vps: list; segments: np.ndarray
```

## Perception output contract
Every analyzer implements `analyze(img_srgb) -> dict[str, np.ndarray]` with these standard keys (missing model ⇒ key absent, pipeline degrades gracefully):

| Key | Shape | Meaning |
|---|---|---|
| `semantic`, `instances` | (H,W) int | class / instance labels (+ `class_names` list) |
| `hair_mask`, `skin_mask`, `cloth_mask`, `human_parts` | (H,W) | human parsing |
| `face_landmarks` | (F,478,3) | MediaPipe landmarks, px coords |
| `depth` | (H,W) | 1 = near |
| `normals` | (H,W,3) | unit normals |
| `edge_prob` | (H,W) | perceptual boundary probability |
| `xdog` | (H,W) | fine texture edges |
| `saliency`, `alpha` | (H,W) | saliency and foreground alpha |
| `segments` | (M,4) | straight line segments (x0,y0,x1,y1) |
| `vps` | (K,2) | vanishing points |

## Style parameter conventions
Tunable style parameters (~30–40) live in YAML and are registered with a name, default, and range so Module 20 can tune them.

---

# Modules

**Build order:** 1 → (2, 4, 5 in parallel) → 3 → (6, 7, 8) → (9, 10, 11) → 12 → (13, 14, 15, 16) → 17 → 18 → 19 → 20 → 21

**Milestone map:** M0 = 4+5 · M1 = 6–11 (+ debug overlays) · M2 = 2+3 · M3 = 12+13 · M4 = 14 · M5 = 15–17 · M6 = 18+20 · M7 = 21

---

## Module 1: Foundation & Core Data Model
**Depends on:** nothing

**Deliverables:** repo scaffold, `core/*`, `configs/default.yaml`, `pyproject.toml`, pytest setup.

**Spec**
- `types.py`: `Stroke`, `ContourChain`, `RegionInfo`, `DrawingPlan` exactly as in the Global Context, with `validate()` methods and (de)serialization to `.npz`/JSON.
- `config.py`: YAML load into typed dataclasses, profile merging (default ← style profile ← CLI overrides), schema validation with clear errors, global `seed`, `handedness` ("right"/"left"), `working_res` (long side), `supersample`, and the user controls: grade set, looseness, detail threshold, hatch angle, blend amount, tooth strength, paper preset.
- `rng.py`: `child(seed, *keys) -> np.random.Generator` using SeedSequence spawning.
- `io.py`: load image with EXIF orientation and ICC → sRGB, resize to working res (area/Lanczos), save 8-bit and 16-bit PNG.
- `color.py`: sRGB ↔ linear ↔ CIE Lab, `luminance_Lstar(img)`.
- `cache.py`: content-hash disk cache for dict-of-ndarray outputs (npz), key = (name, version, image hash, params hash), `@cached` decorator.
- `geometry.py`: arc-length resampling, cubic Bézier eval/derivative/curvature, Douglas–Peucker, polyline smoothing, orientation ↔ structure-tensor conversion, **tensor averaging** with weights, tangent/normal helpers, point-in-mask sampling (bilinear).
- `debug.py`: overlay helpers: orientation field as LIC image, heatmap overlay, colored polyline overlay, side-by-side montage.

**Acceptance**
- Round-trip: config → YAML → config is identical; invalid config raises a readable error.
- Same seed ⇒ same random stream regardless of call order of children.
- Tensor averaging of angles 179° and 1° gives ~0°/180°, not 90°.
- Cache hit avoids recomputation (test with a counter).
- `pytest` passes; demo writes a LIC visualization of a synthetic field.

**Interface snapshot must include:** all dataclasses, `Config` fields, `rng.child`, `cache.cached`, geometry helpers.

---

## Module 2: Graphite & Paper Physical Model (the rasterizer)
**Depends on:** 1. **Build this first among the render modules; it dominates realism.**

**Deliverables:** `render/paper.py`, `render/graphite.py`, `render/canvas.py`, `configs/paper_presets.yaml`.

**Spec**
- **Paper heightfield** `H ∈ [0,1]`: 5-octave fractal noise (anisotropic, 1.2–1.5×) + short random fibers. Also `from_photo(path)`: high-pass a raking-light paper photo. Tooth scale ≈ 1–3 px at *final* resolution, so the canvas works at supersample `S` = 2–3.
- **Grade table** (2H, H, HB, B, 2B, 4B, 6B, 8B): per grade `alpha`, `kappa`, `eta`, footprint width scale, dark floor. Defaults are placeholders; Module 3 calibrates them.
- **Stamp** (Numba-jitted), core equation:
  ```python
  def stamp(D, H, p, direction, P, r, g):
      fp      = footprint(r, direction, g)          # elliptical (side of lead) or round (point)
      reach   = 1.0 - kappa[g] * P                  # pressure -> depth into tooth
      contact = smoothstep(reach - eps, reach + eps, H[patch])
      dD      = alpha[g] * P**eta * fp * contact
      D[patch] = 1 - (1 - D[patch]) * (1 - dD)      # saturating accumulation
  ```
  Light pressure deposits only on tooth peaks (natural stochastic grain). Heavy pressure fills valleys (solid tone, slight sheen).
- `render_stroke(stroke)`: resample the path to stamp spacing ≈ 0.25–0.5 r, interpolate width/pressure, apply **nib wear** (`r` grows ∝ √cumulative path length, resets on simulated sharpen events).
- `GraphiteCanvas`: buffers `D` at `S×` resolution, **named layers** (`L0_construction`, `L1_tone`, …) composited by saturating accumulation, `render_stroke`, `add_dot`, `reflectance()` → `R = R_paper·(1 − D_max·D)` plus a small sheen term ∝ `smoothstep(D)·(1−tooth)`, `downsample()` (area/Lanczos), `snapshot(layer)`.
- Tiling hooks: `render_region(strokes, bbox, halo)` so Module 19 can process 1024² tiles.

**Acceptance**
- Uniform hatching swatches: mean `D` increases monotonically with pressure for each grade.
- At light pressure the covered fraction is < 100% and correlates with tooth peaks; at heavy pressure it approaches solid.
- Softer grades are darker at equal pressure (2H < HB < 6B).
- Deterministic output for a fixed seed; benchmark (stamps/sec) printed.
- Demo: value ladder + pressure ladder + gradient ramp swatch images.

---

## Module 3: Graphite Calibration
**Depends on:** 2

**Deliverables:** `render/calibration.py`, `assets/calibration/*.json`, `scripts/calibrate.py`.

**Spec**
- Swatch generators: value ladder (2H→8B), pressure ladder (P = 0.1…1.0 per grade), gradient ramp, and standard uniform-hatch swatch.
- Measure mean `D` / reflectance per (grade, P); build the **tone→pressure LUT** (monotone inverse) exposed as `tone_to_pressure(grade, target_D)` and `pick_grade_pressure(target_D)`.
- Optional fit of `alpha_g`, `kappa_g`, `D_max` to **scanned** ladders (`scipy.optimize.least_squares`) if the user supplies scans; otherwise ship reasoned defaults.
- Report: montage of synthetic swatches beside scan crops, with luminance histogram, radial power spectrum, and grain autocorrelation comparison.

**Acceptance**
- `tone_to_pressure` round-trips: rendering at the returned pressure hits the target `D` within 3%.
- Calibration JSON loads automatically in Module 2.
- Report script produces the comparison montage without a scan (synthetic only) and with one.

---

## Module 4: Baseline Renderer (M0 yardstick)
**Depends on:** 1

**Deliverables:** `baseline/lu_xu_jia.py`, CLI `python -m graphite.baseline in.jpg out.png`.

**Spec**
- Line drawing: FDoG/XDoG edges, plus line-integral-convolution stroke texture along a fixed direction set (Lu–Xu–Jia 2012 style).
- Tone: histogram matching to a parametric graphite tone distribution, then multiply with a tooth texture.
- Purely a **filter**. It exists only as the bar every later stage must beat. Clearly document this in the file header.

**Acceptance**
- Runs on the 10-category test set from Module 5 (or any 5 images), outputs grayscale PNG.
- Runtime < 5 s for 2000 px images.

---

## Module 5: Evaluation Harness
**Depends on:** 1 (uses 4 as the first system under test)

**Deliverables:** `eval/corpus.py`, `eval/metrics.py`, `eval/report.py`, `scripts/eval.py`.

**Spec**
- **Corpus loader**: directory of 100–200 graphite drawing scans (user supplies; open-access sources like Met, Rijksmuseum, NGA, Wikimedia Commons), normalized to grayscale, cropped tiles. Optional downloader driven by a user-supplied `manifest.csv`.
- **Metrics**:
  - multi-scale SSIM on blurred luminance (photo vs render)
  - tone-histogram EMD vs corpus
  - edge F-score / chamfer distance vs `edge_prob` (fall back to XDoG if perception isn't built yet)
  - radially averaged power-spectrum slope and anisotropy vs scans
  - stroke statistics (length, width, angle, overlap) when a `Stroke` list is available
  - grain autocorrelation vs scans
  - optional real-vs-render classifier used **only** as a critic metric
- **Test set runner**: folder per category (frontal portrait, profile portrait, landscape, architecture, animal fur, macro texture, high-key, low-key, noisy phone photo, CG/anime); outputs JSON + HTML report with side-by-sides.
- **Human A/B sheet generator**: randomized left/right pairs, answer key saved separately.

**Acceptance**
- Baseline (Module 4) gets a full report on the test set.
- Metrics are deterministic and unit-tested on synthetic inputs (e.g., identical images score perfect SSIM).

---

## Module 6: Perception I: Segmentation, Humans, Face
**Depends on:** 1

**Deliverables:** `perception/base.py`, `segmentation.py`, `humans.py`, `face.py`, `class_map.py`.

**Spec**
- `base.py`: `Analyzer` base class (`name`, `version`, `analyze()`), lazy imports so torch is only loaded when a model runs, per-model disk caching via Module 1, ONNX/torch backend switch.
- **SAM 2** automatic masks + **Mask2Former** panoptic ⇒ `semantic`, `instances`, `class_names`, merging logic to resolve overlaps.
- **Sapiens or SCHP** human parsing ⇒ `human_parts`, `hair_mask`, `skin_mask`, `cloth_mask`.
- **MediaPipe Face Landmarker** ⇒ 478 landmarks + iris; derived masks for eyes, brows, nose, lips; catchlight candidates (small bright spots in iris).
- `class_map.py`: table mapping model classes → region types (hair, skin, cloth, foliage, sky, water, stone, manmade, fur, other) and default style key.
- Every model wrapper falls back cleanly (returns no key) if weights are unavailable.

**Acceptance**
- With mocked models, the class map and merging logic produce correct `RegionInfo` dicts.
- Cached second run is instant.
- `import graphite.render` never imports torch.

---

## Module 7: Perception II: Geometry (depth, normals, light, lines)
**Depends on:** 1, 6 (base class)

**Deliverables:** `perception/geometry3d.py`, `light.py`, `lines.py`.

**Spec**
- **Depth Anything V2** ⇒ `depth` normalized, 1 = near.
- **DSINE** normals (Sapiens normals for human regions when available) ⇒ unit normals.
- **Light estimation**: regress luminance on normals with order-1 spherical harmonics (least squares) over high-confidence Lambertian regions (trim near-black/near-white, low-confidence normals, use IRLS). Output 3D light direction `l3d`, image-plane `l2d`, `ambient`, `confidence`.
- **DeepLSD** segments + **vanishing-point RANSAC** ⇒ `segments`, `vps`, per-segment VP assignment, and a `manmade_mask` from VP-consistent segment density.
- Derived maps: depth-discontinuity map, normal-discontinuity (crease) map, SSAO-style **occlusion/contact-shadow map** from depth.

**Acceptance**
- Light estimator recovers a known direction (±10°) on a synthetic Lambertian sphere render.
- VP RANSAC finds the correct VPs on a synthetic perspective grid.
- Confidence drops on flat/uninformative images.

---

## Module 8: Perception III: Edges, Saliency & Bundle
**Depends on:** 1, 6, 7

**Deliverables:** `perception/edges.py`, `saliency.py`, `bundle.py`.

**Spec**
- **TEED or PiDiNet** ⇒ `edge_prob` (object/perceptual boundaries).
- **XDoG** ⇒ `xdog` fine texture edges (scale/threshold params in config).
- **BiRefNet or U²-Net** ⇒ `saliency`, `alpha`.
- `bundle.py`: `analyze_all(img_srgb, cfg) -> PerceptionBundle` that runs all analyzers with caching, exposes fused `P_boundary` (nets) and `P_texture` (XDoG), and reports which models ran.

**Acceptance**
- `analyze_all` on the test set returns all standard keys (or documented absences).
- Debug script writes overlays: edge_prob, saliency, depth, normals (RGB), light arrow, segments/VPs.

---

## Module 9: Plan I: Tone Target, Importance, Reserves, Shadow Shapes
**Depends on:** 1, 8

**Deliverables:** `plan/tone.py`, `importance.py`, `reserves.py`, `shadows.py`.

**Spec**
- **Tone target `T`**: `L*` → tone curve compressed to graphite range → edge-preserving abstraction (anisotropic Kuwahara, Kyprianidis 2009, or WLS).
- **Bands**: quantize into K = 6–9 value bands (k-means in `L*`) used as **layer selectors only** (stroke density uses continuous `T`). Per-class remaps: compress skin range, merge the two darkest bands into simplified shadow shapes, drop small blobs by area.
- **Importance `W`** = norm(α·saliency + β·face/eye mask + γ·edge density + δ·nearness); falls off toward frame edges so the drawing dissolves into bare paper.
- **Reserved whites**: top-percentile local luminance ∩ high `W`, as connected components. Eyes and catchlights are forced.
- **Shadow shapes**: connected components of the darkest bands after simplification (morphology, smoothing, area threshold). Artists draw shadow shapes, not pixels.

**Acceptance**
- `T` is monotonic with input luminance in smooth areas; no halos at strong edges.
- Every band boundary is covered by a labeled component or stays continuous in `T`.
- Debug: `W` heatmap, band map, reserves, and shadow shapes overlays.

---

## Module 10: Plan II: Orientation Field & Coherence
**Depends on:** 1, 7, 8, 9

**Deliverables:** `plan/orientation.py`.

**Spec**
- Smoothed structure tensor at σ ≈ 2–5 % of min(H,W); **edge tangent flow** (Kang 2007).
- Coherence `c = ((λ₁−λ₂)/(λ₁+λ₂))²`.
- Region-specific fields, each as a tensor field:
  - **hair**: ETF constrained by the gradient of the mask distance transform
  - **organic/cloth**: minimum-curvature direction of the normal field (cross-contour hatching)
  - **ground planes**: toward the vanishing point
  - **sky/background**: global `θ₀ ≈ 45°` (mirrored for left-handed) + low-frequency noise
- Blend per region by **tensor averaging** with soft region weights, never angle averaging.
- Output `theta`, `coherence`, and the blended tensor for reuse.

**Acceptance**
- No discontinuities at the ±π wrap (test with synthetic fields).
- Hair strands on a synthetic head mask follow the mask contour.
- LIC visualization saved for each test image; high `c` regions look directional, low `c` regions look noisy.

---

## Module 11: Plan III: Contour Graph
**Depends on:** 1, 7, 8, 9

**Deliverables:** `plan/contours.py`, `core/schneider.py`, `plan/build.py` (assembles the full `DrawingPlan`).

**Spec**
- `P_edge` → NMS → hysteresis → chain linking with **junction detection** → prune by length (threshold ∝ 1/`W`).
- Classify each chain: **silhouette** (depth jump or mask boundary), **crease** (normal jump), **tonal** (shadow-shape boundary), **texture**.
- Fit cubic Béziers (**Schneider algorithm**, custom implementation) with tolerance `ε(W)`; store curvature `κ(s)`.
- Compute `outward_normal(s)` (points away from the nearer/figure side using depth/mask).
- Rank by `S = mean(P_edge)·length^γ·(1 + depth_jump)`.
- `build.py`: `build_plan(bundle, cfg) -> DrawingPlan` calling Modules 9, 10, 11 and `regions` from Module 6.

**Acceptance**
- Fit error ≤ ε on synthetic curves; corners are preserved.
- Chain count decreases as `W` decreases; the prune threshold visibly varies across the image.
- Debug overlay: color-coded contour classes, junction dots.
- **Plan cache**: full `DrawingPlan` saved/loaded via `.npz`.

---

## Module 12: Hand Model & Stroke Primitives
**Depends on:** 1, 2

**Deliverables:** `strokes/hand.py`, `strokes/styles.py`, `configs/styles/*.yaml`.

**Spec**
- **Arm-pivot curvature** `κ ≈ 1/R_arm`, R growing with stroke length, consistent sign per handedness (subtly fans hatching).
- **Tremor**: 1/f^1.5 filtered noise perpendicular to the path (≈ 0.3 px + 0.4 % of length).
- **Pressure profile** `p(s)`: short ramp-in, plateau, long taper-out (flick).
- Start/end staggering, overshoot (3–8 %), overlap 0–25 %, restated second pass (lower jitter, higher pressure), micro-gaps at joints.
- `StyleProfile` dataclass: length distribution, width, pressure, jitter, curvature, spacing multiplier, blend amount, layer set. Ship default YAMLs for each region class.
- API: `make_stroke(ideal_path, kind, style, rng) -> Stroke`, `break_into_hand_strokes(chain, max_frac=0.08–0.15)`, `restate(stroke, rng)`, `NibScheduler` (sharpen events).

**Acceptance**
- Tremor power-spectrum slope ≈ −1.5 (log-log fit).
- Same ideal path + different seeds ⇒ visibly different but statistically similar strokes.
- Rendering 100 strokes through the canvas shows tapered ends and slight fan in hatching.

---

## Module 13: Construction & Contour Layers (L0, L2)
**Depends on:** 2, 11, 12

**Deliverables:** `strokes/construction.py`, `strokes/contour_strokes.py`.

**Spec**
- **L0 construction**: simplified silhouette (Douglas–Peucker, high ε) drawn as 1–3 light overlapping near-straight passes with 3–8 % overshoot. Layer is kept separate so L5 can ghost-erase it to 30–50 %.
- **L2 contours**: re-segment chains into hand-length strokes (≤ ~8–15 % of image width, break at high-κ corners), then:
  - weight `w(s) = w₀·f(S)·(1 + λ_s·shadowSide(s))·depthScale·taper(s)`, with `shadowSide = ⟨outward 2D normal, −ℓ₂D⟩ > 0` (heavy on the shadow side, thin/broken on the lit side)
  - **lost-and-found edges**: opacity gated by low-frequency noise and local `|∇T|` across the contour
  - **accents**: heaviest at cast-shadow lines, contact points (depth occlusion), T-junction occluders, and eye/nostril/mouth-corner landmarks
  - corner overshoot, restated second pass, micro-gaps at joints
  - architectural chains snap to VP-consistent segments (≤ 0.3° wobble); organic chains keep full jitter
- Global stroke/ink budget enforced by `W` and `S` ranking.

**Acceptance**
- On a lit sphere/cube test, line weight is higher on the shadow side.
- Architectural test image: long lines converge to the VP.
- Contours look broken/lost in low-`W` areas; no fully closed traced outline everywhere.

---

## Module 14: Hatching Engine (L1 tone masses) + Residual Loop
**Depends on:** 2, 3, 10, 12

**Deliverables:** `strokes/hatching.py`.

**Spec**
- **Placement**: evenly spaced streamlines on `θ` (Jobard–Lefer, farthest-point seeding), RK4 integration, spacing `d(x) = d_min + (d_max − d_min)·T(x)^γ`. Terminate when coherence drops below threshold, when the band no longer matches the layer, at strong edges (probabilistically), or at `ℓ_max ~ lognormal(∝ 1/W)`.
- **Layers**: layer k is active where `T < τ_k` (e.g. .85/.65/.45/.25/.12) at angle `θ + Δ_k` with `Δ = (0, +55°, −35°, +90°, scribble)` and per-stroke `σ_θ` of 3–6°. Dither the thresholds with **blue noise** so no layer boundary shows. Low coherence ⇒ scribble/stipple fallback.
- **Residual loop** (Hertzmann-style, coarse to fine, error-driven): after each layer compute `E = G_σ*(T) − G_σ*(T_rendered)` and place strokes only where `E > τ`. Coarse layers: wide, soft, side-of-lead (6B) footprints; later layers: pointed nib.
- Use the Module 3 LUT to pick (grade, pressure) for each stroke from local target tone.
- Hand model from Module 12 applied to every stroke.

**Acceptance**
- Uniform gray, gradient, and step-edge synthetic targets: final rendered tone matches `T` within tolerance (report RMSE after blur).
- No visible layer boundaries on the gradient.
- No comb-like alignment (stagger works): quantify start-position autocorrelation.

---

## Module 15: Stippling
**Depends on:** 2, 9

**Deliverables:** `strokes/stipple.py`.

**Spec**
- Weighted Voronoi stippling (Secord 2002) with Lloyd relaxation on density `(1 − T)·mask`, plus a structure-aware blue-noise option.
- Dots are emitted as short/zero-length `Stroke`s and stamped through the same deposit model so they inherit paper tooth (no digital dots).
- API: `stipple(region_mask, density, n_max, rng) -> list[Stroke]`; used for stone, pores, and gradient tails.

**Acceptance**
- Radial power spectrum shows blue-noise character (low-frequency dip).
- Dot density tracks the target density map (correlation reported).

---

## Module 16: Class Textures (L3)
**Depends on:** 6, 10, 11, 12, 14, 15

**Deliverables:** `strokes/textures/{hair,foliage,skin,cloth,sky_water,stone,manmade}.py` and a registry.

**Spec** (each generator: `generate(region, plan, style, rng) -> list[Stroke]`)
- **hair**: long S-curved tapered strokes along the ETF, clustered into locks (k-means along the flow-normal axis), dark gaps between locks, highlights reserved.
- **foliage/grass**: short curled or zigzag scribbles from curl-noise curves, Poisson-disk seeded, density ∝ `(1 − T)`.
- **skin**: minimal contours, mostly blended tone; stipple only at high `W`.
- **cloth**: fold lines from normal discontinuities plus hatching along the folds.
- **sky/water**: long light near-parallel strokes, heavy blending.
- **stone/concrete**: stipple plus short irregular dabs.
- **man-made**: ruler-straight hatching aligned with VP segments.

**Acceptance**
- Each generator renders a recognizable texture on a synthetic mask with its class style.
- Registry dispatch by `RegionInfo.style`; unknown class falls back to generic hatching.

---

## Module 17: Blending (L4) & Erasing (L5)
**Depends on:** 2, 10, 14

**Deliverables:** `render/blend.py`, `render/erase.py`.

**Spec**
- **Tortillon**: LIC-smear `D` along `θ` (kernel 5–25 px) plus isotropic blur σ ≈ 1.5–3 px, masked by a **blend map** (skin, sky, low-`W`). Smudging pushes graphite into the valleys, so re-apply tooth at reduced `κ` afterward (smooth gradient, faint grain). Add hand-drag smears (broad, ~2 % opacity, on the trailing side of dark masses) and sparse graphite dust.
- **Kneaded eraser**: dabs on reserved-white components with `D ← D·(1 − e·softbrush·tooth)`, leaving faint residual streaks; global **ghost-erase** of the L0 layer to 30–50 %.

**Acceptance**
- Blended gradient is smoother than un-blended but retains visible grain (measure autocorrelation).
- Erased regions are lighter but never perfectly clean (residual > 0 where graphite was heavy).
- L0 lines are still faintly visible after ghost-erasing.

---

## Module 18: Paper Finish & Output
**Depends on:** 2, 17

**Deliverables:** `render/finish.py`.

**Spec**
- Warm off-white base, low-frequency exposure drift, visible fibers in unshaded areas, slight vignette, ~1 % scanner noise, then supersample-down (area/Lanczos).
- Output **16-bit grayscale** with a near-neutral graphite tone; optional slight tint. Paper presets from Module 2.
- Apply tooth **before** any blur (avoid digital grain).

**Acceptance**
- Unshaded areas show paper fibers, not flat white.
- 16-bit PNG saved; histogram has no banding.
- Preset switching changes paper character without changing the drawing.

---

## Module 19: Pipeline Orchestrator, CLI, Tiling & Debug
**Depends on:** 1–18 (as many as exist)

**Deliverables:** `pipeline.py`, `cli.py`, tile support, debug export.

**Spec**
- `DrawingPipeline.run(img_path, cfg) -> DrawingResult` executing stages in order with per-stage artifact caching and timing.
- **Tiling**: > ~3000 px long side, tile the canvas at 1024² with halos; strokes are generated globally (vector), rasterized per tile. (6000² float32 ≈ 144 MB per buffer; expect ~3·10⁴–2·10⁵ strokes.)
- **CLI**: `graphite draw in.jpg -o out.png --profile portrait --grades HB,2B,6B --looseness 1.2 --detail 0.6 --hatch-angle 45 --hand right --blend 0.5 --tooth 0.8 --paper cold_press --seed 7`.
- Debug export: `θ` LIC, `W` heatmap, contour classes, band map, per-layer canvas snapshots, timing report.
- Stroke export as JSON/SVG (for plotters or refinement).

**Acceptance**
- End-to-end run on all 10 test categories without errors.
- Same seed ⇒ byte-identical output; different seed ⇒ different but similar drawing.
- Tiled and untiled renders match (no seams).

---

## Module 20: Style Tuning (Optuna)
**Depends on:** 5, 19

**Deliverables:** `tune/optuna_search.py`, `configs/tuned/*.yaml`.

**Spec**
- Parameter registry (~30–40 style parameters with ranges) built from the YAML conventions.
- Objective: weighted composite of the Module 5 metrics on the test set; optional multi-objective (fidelity vs realism). Fast proxy mode (small images) for search, full mode for final check.
- Resumable studies (SQLite), best profile exported to YAML, importance analysis report.

**Acceptance**
- Tuned profile beats default on the composite score on a held-out image set.
- Study resumes after interruption.

---

## Module 21: Differentiable Stroke Refinement (optional)
**Depends on:** 2, 19 (torch allowed here)

**Deliverables:** `refine/diff_raster.py`, `refine/optimize.py`.

**Spec**
- **Differentiable twin** of the rasterizer: soft stamping and soft tooth contact (sigmoid in place of the hard smoothstep), strokes as Bézier control points + width + pressure + grade (DiffVG-style).
- Adam on stroke parameters, initialized from the procedural output. Loss:
  - multi-scale blurred tone L1
  - chamfer/distance-transform loss between rendered ink edges and the contour graph
  - low-weight DINOv2 feature loss against the photo luminance
  - priors on stroke length, curvature, overlap, and a stroke-count budget
- Prune redundant strokes; export the refined `Stroke` list and render it with the **true (non-differentiable) rasterizer** for final output.
- No pixel is ever synthesized by a network.

**Acceptance**
- Loss decreases and the multi-scale tone error drops vs the un-refined render.
- Soft vs hard rasterizer parity check (difference reported) on random strokes.
- Stroke count decreases without a fidelity drop beyond a set tolerance.

---

# Progress tracker

| # | Module | Done | Snapshot saved |
|---|---|---|---|
| 1 | Foundation & data model | ☐ | ☐ |
| 2 | Graphite & paper model | ☐ | ☐ |
| 3 | Calibration | ☐ | ☐ |
| 4 | Baseline (M0) | ☐ | ☐ |
| 5 | Evaluation harness | ☐ | ☐ |
| 6 | Perception I | ☐ | ☐ |
| 7 | Perception II | ☐ | ☐ |
| 8 | Perception III + bundle | ☐ | ☐ |
| 9 | Plan I | ☐ | ☐ |
| 10 | Plan II | ☐ | ☐ |
| 11 | Plan III + plan builder | ☐ | ☐ |
| 12 | Hand model | ☐ | ☐ |
| 13 | Construction & contours | ☒ | ☒ |
| 14 | Hatching + residual loop | ☒ | ☒ |
| 15 | Stippling | ☐ | ☐ |
| 16 | Class textures | ☐ | ☐ |
| 17 | Blending & erasing | ☐ | ☐ |
| 18 | Paper finish | ☐ | ☐ |
| 19 | Orchestrator & CLI | ☐ | ☐ |
| 20 | Optuna tuning | ☐ | ☐ |
| 21 | Differentiable refinement | ☐ | ☐ |
