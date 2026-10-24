import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_legacy_keras_flag_set_before_tensorflow_import():
    code = (
        "import os, sys\n"
        "os.environ.pop('TF_USE_LEGACY_KERAS', None)\n"
        "assert 'tensorflow' not in sys.modules\n"
        "import core.model_loader as ml\n"
        "assert os.environ['TF_USE_LEGACY_KERAS'] == '1'\n"
        "assert ml.models['url_text_model'] is not None and ml.models['lstm'] is not None\n"
        "assert not any(ml.models['errors'].values()), ml.models['errors']\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]
