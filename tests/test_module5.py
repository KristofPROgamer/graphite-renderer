from __future__ import annotations

from pathlib import Path

import numpy as np

from graphite.eval.corpus import CorpusEntry, load_corpus
from graphite.eval.metrics import edge_f_score, histogram_emd, power_spectrum_slope, ssim_score
from graphite.eval.report import write_html_report


def _make_test_image(shape=(64, 64), *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random(shape, dtype=np.float32)
    img = img.astype(np.float32)
    return img


def test_identical_image_scores_perfectly():
    img = _make_test_image()
    assert abs(ssim_score(img, img) - 1.0) < 1e-6
    assert histogram_emd(img, img) < 1e-6
    assert edge_f_score(img, img) > 0.99
    assert np.isfinite(power_spectrum_slope(img))


def test_load_corpus_reads_directory_and_manifest(tmp_path: Path):
    category_dir = tmp_path / "portrait"
    category_dir.mkdir()
    ref = _make_test_image((32, 32), seed=7)
    np.save(category_dir / "sample.npy", ref)
    (tmp_path / "landscape").mkdir()
    other = _make_test_image((32, 32), seed=11)
    np.save(tmp_path / "landscape" / "other.npy", other)

    entries = load_corpus(tmp_path)
    assert len(entries) == 2
    assert all(isinstance(e, CorpusEntry) for e in entries)
    assert {e.category for e in entries} == {"portrait", "landscape"}

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("name,category,path\nfoo,portrait,portrait/sample.npy\n", encoding="utf-8")
    manifest_entries = load_corpus(tmp_path, manifest_path=manifest_path)
    assert manifest_entries[0].name == "foo"
    assert manifest_entries[0].category == "portrait"


def test_html_report_is_written(tmp_path: Path):
    report_path = tmp_path / "report.html"
    rows = [{
        "name": "demo",
        "category": "portrait",
        "ssim": 0.85,
        "edge_f_score": 0.82,
        "histogram_emd": 0.15,
        "power_spectrum_slope": -1.8,
    }]
    html = write_html_report(rows, report_path)
    assert report_path.exists()
    assert "demo" in html
    assert "portrait" in html
