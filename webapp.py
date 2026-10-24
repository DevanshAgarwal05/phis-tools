# webapp.py - Streamlit entry point:  streamlit run webapp.py
import os

# Must run before TensorFlow/Keras is imported anywhere in the process.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import __main__

from analyzers.url.selectors import sel_url, sel_host, sel_struct

# The URL pipeline pickle references these selectors as attributes of __main__.
for _fn in (sel_url, sel_host, sel_struct):
    setattr(__main__, _fn.__name__, _fn)

from core.model_loader import models  # noqa: E402  (loads/caches all ML models)
from ui.main_view import render_ui  # noqa: E402

render_ui(models)
