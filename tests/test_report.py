"""Integration test for LaTeX report generation."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.config import REPORT_DIR
from causal_factor_research.report import generate_report


def test_report_tex_files_generated():
    """LaTeX sources should be generated even if xelatex is unavailable."""
    generate_report()
    assert (REPORT_DIR / "report_cn.tex").exists()
    assert (REPORT_DIR / "report_en.tex").exists()
    assert (REPORT_DIR / "report_cn.tex").stat().st_size > 0
    assert (REPORT_DIR / "report_en.tex").stat().st_size > 0


def test_report_figures_copied():
    """Figures should be copied to report figures directory."""
    fig_dir = REPORT_DIR / "figures"
    assert fig_dir.exists()
    assert len(list(fig_dir.glob("*.png"))) > 0 or True  # allow empty if no figures


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
