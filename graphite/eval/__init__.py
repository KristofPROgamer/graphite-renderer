"""Evaluation utilities for scoring render fidelity against reference corpus data."""

from .corpus import CorpusEntry, load_corpus
from .metrics import edge_f_score, histogram_emd, power_spectrum_slope, ssim_score
from .report import write_html_report

__all__ = [
    "CorpusEntry",
    "load_corpus",
    "ssim_score",
    "histogram_emd",
    "edge_f_score",
    "power_spectrum_slope",
    "write_html_report",
]
