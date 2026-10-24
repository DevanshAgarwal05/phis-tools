"""The URL pipeline pickle references sel_url/sel_host/sel_struct in __main__.

Mirrors the bridge in webapp.load_models and checks the model unpickles and
reproduces the probabilities captured before the refactor.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzers.url.selectors import sel_host, sel_struct, sel_url  # noqa: E402
from config import settings as cfg  # noqa: E402
from utils.url_utils import canonicalize_url, extract_host  # noqa: E402

BASE = json.loads((ROOT / "tests" / "pickle_baseline.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    import __main__ as _main
    for name, fn in (("sel_url", sel_url), ("sel_host", sel_host), ("sel_struct", sel_struct)):
        setattr(_main, name, fn)
    return joblib.load(cfg.URL_PIPELINE_MODEL)


@pytest.mark.parametrize("url", [u for u in BASE if "#struct" not in u])
def test_probabilities_and_struct_unchanged(model, url):
    c = canonicalize_url(url)
    X = pd.DataFrame([{"url": c, "host": extract_host(c)}])
    assert [float(p) for p in model.predict_proba(X)[0]] == pytest.approx(BASE[url], abs=1e-12)
    assert [float(v) for v in sel_struct(X)[0]] == pytest.approx(BASE[url + "#struct"], abs=1e-12)
