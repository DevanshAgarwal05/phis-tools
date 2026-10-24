# PhisTools

Automated phishing triage (Streamlit): email text/headers, URL heuristics + ML, threat intel and remote-attachment scanning.

```
pip install -r requirements.txt
streamlit run webapp.py
```

## Layout
| Path | Purpose |
|---|---|
| `webapp.py` | Thin entry point: Keras flag, `__main__` pickle shim, calls `ui.main_view.render_ui` |
| `ui/` | Streamlit page (`render_ui`) |
| `core/` | Risk scoring, policy thresholds, explainability, model loading/inference |
| `analyzers/{email,url,network}/` | Header analysis; URL features + selectors; threat intel, remote fetch, attachment analysis |
| `utils/` | Pure helpers (text/url/model utils, scan logger) |
| `config/settings.py` | `BASE_DIR` and all artifact/log paths |
| `training/` | Training and dataset scripts - run from the repo root, e.g. `python training/train_all_visual.py` |
| `models/`, `data/`, `logs/`, `docs/` | Artifacts, datasets (git-ignored CSVs), scan logs, reports and graphs |
| `tests/` | `pytest tests` - golden UI-output test, pickle-shim test, unit tests |

`models/url_tfidf_logistic_model.pkl` was pickled with `sel_url/sel_host/sel_struct` living in `__main__`;
`webapp.py` and `core.model_loader.load_models` re-register them there. Do not remove that bridge.
