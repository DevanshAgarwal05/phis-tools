# ui/main_view.py
import os
import numpy as np
import pandas as pd
import streamlit as st

from config import settings as cfg

from analyzers.network.remote_fetch import fetch_remote_attachment
from analyzers.email.header_analysis import analyze_headers
from core.policy import policy_thresholds, policy_threat_level, calibrate_final_score
from core.explain import grouped_explainability, classify_attack_types, recommended_actions
from core.model_loader import predict_email_models, predict_url_ml

# ✅ NIS modules (you created these)
from analyzers.url.nis_features import extract_url_nis_features, extract_email_nis_features
from core.risk_scoring import compute_risk_score, build_reasons
from utils.scan_logger import append_log, now_iso

# ✅ threat_intel.py (safe version)
from analyzers.network.threat_intel import offline_intel, check_virustotal, check_abuseipdb


try:
    from analyzers.network.attachment_file_analyzer import analyze_attachment_file
    ATTACHMENT_ANALYZER_AVAILABLE = True
except Exception as e:
    ATTACHMENT_ANALYZER_AVAILABLE = False
    ATTACHMENT_IMPORT_ERROR = str(e)


def render_ui(models):
    """Render the Streamlit page. Statements run sequentially, exactly as the former module-level script."""
    # ----------------------------
    # Streamlit UI
    # ----------------------------
    st.set_page_config(page_title="Phishing Detection & Prevention Tool", layout="centered")
    st.title("Phishing Detection and Prevention Tool")
    st.caption("Detect phishing based on email content, headers, URLs, and remote attachments.")

    # ✅ Policy mode (A upgrade)
    policy = st.selectbox(
        "Organization Policy Mode",
        [
            "Normal Policy (University / General)",
            "Strict Policy (Banking / Enterprise)",
            "Relaxed Policy (Personal Use)"
        ],
        index=0
    )

    ths = policy_thresholds(policy)
    susp_th = float(ths["suspicious"])
    phish_th = float(ths["phishing"])

    st.info(f"Policy thresholds → Suspicious ≥ {susp_th:.0f}, Phishing ≥ {phish_th:.0f} (Risk Score out of 100)")

    # ✅ NIS toggles
    security_mode = st.checkbox("Security Analysis Mode (NIS)", value=True)
    show_history = st.checkbox("Show Scan History (last 20)", value=True)

    email_ok = (models.get("lstm") is not None and models.get("tokenizer") is not None) or (
        models.get("logistic") is not None and models.get("vectorizer") is not None
    )
    url_ok = (models.get("url_text_model") is not None)

    if email_ok:
        st.success("Email ML pipeline is available (LSTM and/or Logistic).")
    else:
        st.warning("Email ML is disabled. Check loaded models in errors below.")

    if models.get("hybrid_meta") is not None:
        st.success("Hybrid Meta model loaded (trained combiner).")
    else:
        st.info("Hybrid Meta model not loaded - app will fallback to averaging if both models exist.")

    if url_ok:
        st.success("URL ML model loaded (Host+URL TF-IDF pipeline).")
    else:
        st.warning("URL ML model not loaded. URL scoring will be neutral (0.5).")

    if not ATTACHMENT_ANALYZER_AVAILABLE:
        st.info(
            "Attachment scanning module is not available on this setup.\n\n"
            f"Import error: {ATTACHMENT_IMPORT_ERROR}"
        )

    with st.expander("Debug: Loaded model status"):
        for k, v in models["errors"].items():
            if v:
                st.write(f"{k}: {v}")
            else:
                st.write(f"{k}: OK")

    st.markdown("### Email Content Analysis")
    subject = st.text_input("Subject:")
    body = st.text_area("Email Body:")
    header_text = st.text_area("Email Headers (optional):")
    urls_input = st.text_area("URLs Found (optional, one per line):")

    if st.button("Analyze Email"):
        # Email side
        if subject.strip() or body.strip():
            log_prob, lstm_prob, hybrid_prob, email_note = predict_email_models(subject, body)
            st.subheader("Email Model Predictions")

            st.write(
                f"**Logistic Regression Probability:** {log_prob:.2f}"
                if log_prob is not None
                else "**Logistic Regression Probability:** (not available)"
            )
            st.write(
                f"**LSTM Model Probability:** {lstm_prob:.2f}"
                if lstm_prob is not None
                else "**LSTM Model Probability:** (not available)"
            )
            st.write(f"**Email Probability Used:** {hybrid_prob:.2f}")
            if email_note:
                st.info(email_note)
        else:
            hybrid_prob = 0.5
            log_prob, lstm_prob = None, None
            st.info("No email content provided - skipping email model prediction.")

        # Header side
        header_risk, header_issues, header_details = analyze_headers(header_text)
        st.subheader("Header Analysis")
        st.write(f"**Header Risk:** {header_risk} *(confidence: {header_details.get('confidence', 'low')})*")
        for issue in header_issues:
            st.write("-", issue)

        # URL side
        url_probs = []
        urls = []
        if urls_input.strip():
            st.subheader("URL Analysis")
            urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
            for u in urls:
                try:
                    p = predict_url_ml(u)
                    url_probs.append(p)
                    st.write(f"{u} → Probability: **{p:.2f}**")
                except Exception as e:
                    st.error(f"Error analyzing {u}: {e}")
            url_mean = float(np.mean(url_probs)) if url_probs else 0.5
            st.write(f"**Average URL Risk Probability:** {url_mean:.2f}")
        else:
            url_mean = 0.5

        # ----------------------------
        # NIS Risk Score
        # ----------------------------
        # Choose top URL for explanation (highest URL risk)
        top_url = None
        if urls and url_probs:
            idx = int(np.argmax(url_probs))
            top_url = urls[idx]

        # Extract NIS features (does NOT affect ML)
        url_feat = extract_url_nis_features(top_url) if top_url else {
            "host": "", "tld": "", "url_len": 0, "has_https": 0, "has_ip_host": 0, "has_at": 0,
            "has_double_slash_in_path": 0, "num_subdomain_dots": 0, "keyword_count": 0,
            "keyword_hits": [], "host_entropy": 0.0, "is_shortener": 0, "suspicious_tld": 0,
            "high_digit_ratio": 0, "many_hyphens": 0
        }
        email_feat = extract_email_nis_features(subject, body, header_text)

        # Threat intel (offline = demo-proof)
        intel = offline_intel(top_url) if top_url else {"source": "offline"}

        # Risk score (0..100)
        risk_score_raw = compute_risk_score(
            email_prob=float(hybrid_prob) if hybrid_prob is not None else 0.5,
            url_prob=float(url_mean) if url_mean is not None else 0.5,
            header_risk=header_risk,
            url_feat=url_feat,
            email_feat=email_feat,
            intel=intel
        )
        risk_score = calibrate_final_score(
            raw_score=risk_score_raw,
            email_prob=float(hybrid_prob) if hybrid_prob is not None else 0.5,
            url_prob=float(url_mean) if url_mean is not None else 0.5,
            header_info=header_details,
            url_feat=url_feat,
        )

        # Policy-aligned threat level
        level = policy_threat_level(risk_score, susp_th, phish_th)

        st.markdown("### Final Combined Verdict (NIS Risk Score)")
        st.write(f"**Risk Score:** {risk_score:.1f} / 100")
        st.write(f"**Threat Level:** {level}  *(Policy Mode: {policy})*")

        # Policy-based verdict thresholds
        if risk_score >= phish_th:
            verdict = "Phishing"
            st.error(f"Phishing Detected (Risk: {risk_score:.1f})")
        elif risk_score >= susp_th:
            verdict = "Suspicious"
            st.warning(f"Suspicious Email (Risk: {risk_score:.1f})")
        else:
            verdict = "Safe"
            st.success(f"Safe Email (Risk: {risk_score:.1f})")

        # ----------------------------
        # ✅ A Upgrade #1: Attack Type Classification
        # ----------------------------
        attack_types = classify_attack_types(url_feat, email_feat, intel, header_risk)
        st.markdown("### Attack Type Classification (NIS)")
        st.write("This helps categorize the incident like a real SOC / security tool would:")
        for t in attack_types:
            st.write("•", t)

        # ----------------------------
        # ✅ A Upgrade #2: Recommended Actions (Incident Response)
        # ----------------------------
        st.markdown("### Recommended Actions (Prevention / Response)")
        actions = recommended_actions(level, attack_types)
        for a in actions:
            st.write("•", a)

        # Reasons + details only in NIS mode
        if security_mode:
            reasons = build_reasons(
                email_prob=float(hybrid_prob) if hybrid_prob is not None else None,
                url_prob_mean=float(url_mean) if url_mean is not None else None,
                header_risk=header_risk,
                url_feat=url_feat,
                email_feat=email_feat,
                intel=intel
            )

            st.markdown("### Why this was flagged (NIS Indicators)")
            grouped = grouped_explainability(
                email_prob=float(hybrid_prob) if hybrid_prob is not None else None,
                url_prob_mean=float(url_mean) if url_mean is not None else None,
                header_info=header_details,
                url_feat=url_feat,
                email_feat=email_feat,
                intel=intel,
                risk_score=float(risk_score),
                policy_name=policy,
            )
            for section_name, items in grouped.items():
                st.markdown(f"**{section_name}**")
                if items:
                    for i in items[:6]:
                        st.write("-", i)
                else:
                    st.write("- No strong signal.")

            with st.expander("Raw compact reasons"):
                for r in reasons:
                    st.write("-", r)

            with st.expander("Technical Details (NIS Features)"):
                st.write("**Top URL used for explanation:**", top_url if top_url else "(none)")
                st.json({
                    "url_features": url_feat,
                    "email_features": email_feat,
                    "threat_intel": intel,
                    "policy": policy,
                    "policy_thresholds": {"suspicious": susp_th, "phishing": phish_th},
                    "attack_types": attack_types,
                    "header_details": header_details,
                    "risk_score_raw": float(risk_score_raw),
                    "risk_score_calibrated": float(risk_score),
                })

            # Optional online intel display (won't break without keys)
            if top_url:
                with st.expander("Online Threat Intel (optional)"):
                    vt = check_virustotal(top_url)
                    st.write("**VirusTotal:**", vt.get("result"))

                    # AbuseIPDB only if host is an IP
                    if url_feat.get("has_ip_host") == 1:
                        ab = check_abuseipdb(url_feat.get("host", ""))
                        st.write("**AbuseIPDB:**", ab.get("result"))
                    else:
                        st.write("**AbuseIPDB:** (host is not an IP)")

        # ✅ Scan logging (safe)
        try:
            append_log({
                "timestamp": now_iso(),
                "risk_score": float(risk_score),
                "threat_level": level,
                "verdict": verdict,
                "email_prob": float(hybrid_prob) if hybrid_prob is not None else 0.5,
                "url_prob_mean": float(url_mean) if url_mean is not None else 0.5,
                "header_risk": str(header_risk),
                "urls": " | ".join(urls) if urls else "",
                "subject_preview": (subject or "")[:80],
            })
        except Exception:
            pass

        # ----------------------------
        # Feedback (kept compatible)
        # ----------------------------
        st.markdown("---")
        st.subheader("Feedback")
        st.write("Was this classification correct?")
        feedback = st.radio("Choose an option:", ["Yes", "No"], horizontal=True, key="feedback_radio")

        if st.button("Submit Feedback"):
            feedback_data = {
                "subject": subject,
                "body": body,
                "urls": urls_input,
                "verdict": verdict,
                "final_score": float(risk_score) / 100.0,
                "email_prob": float(hybrid_prob),
                "url_prob_mean": float(url_mean),
                "user_feedback": feedback,
            }
            feedback_df = pd.DataFrame([feedback_data])

            if os.path.exists(cfg.FEEDBACK_LOG_PATH):
                feedback_df.to_csv(cfg.FEEDBACK_LOG_PATH, mode="a", header=False, index=False)
            else:
                feedback_df.to_csv(cfg.FEEDBACK_LOG_PATH, index=False)

            st.success("Thank you for your feedback. It has been recorded.")

        # ✅ Scan History Viewer
        if show_history:
            try:
                if os.path.exists(cfg.SCAN_LOG_PATH):
                    hist = pd.read_csv(cfg.SCAN_LOG_PATH).tail(20)
                    st.markdown("### Scan History (last 20)")
                    st.dataframe(hist, use_container_width=True)
                else:
                    st.info("No scan history yet.")
            except Exception:
                st.info("Scan history unavailable on this run.")


    # ----------------------------
    # Remote attachment scan
    # ----------------------------
    st.markdown("---")
    st.markdown("### Remote Attachment URL Scan (Optional)")

    remote_url = st.text_input("Enter an attachment URL (e.g., https://example.com/invoice.pdf)")

    if st.button("Scan Remote File"):
        if not remote_url.strip():
            st.warning("Please enter a valid URL.")
        elif not ATTACHMENT_ANALYZER_AVAILABLE:
            st.error("Attachment analyzer is unavailable, cannot scan remote file on this setup.")
        else:
            local_path = None
            try:
                result = fetch_remote_attachment(remote_url)
                st.write("**Remote Fetch Status:**", result.get("status", "unknown"))
                st.write("**Message:**", result.get("message", "No details."))
                if result.get("final_url"):
                    st.write("**Final URL:**", result.get("final_url"))
                if result.get("content_type"):
                    st.write("**Content-Type:**", result.get("content_type"))
                else:
                    st.info("Content-Type was missing; proceeding with cautious local file checks.")

                if not result.get("ok"):
                    status = result.get("status")
                    if status == "blocked":
                        st.warning("Access blocked by remote host (HTTP 403). Try another host/link or manual download.")
                    elif status == "not_found":
                        st.warning("File not found on remote host (HTTP 404).")
                    elif status in ("webpage", "non_direct_url"):
                        st.info("This looks like a webpage link. Use a direct file download URL.")
                    elif status == "timeout":
                        st.warning("Download timed out before completion.")
                    elif status == "invalid_url":
                        st.warning("Invalid URL format for remote scan.")
                    elif status == "tls_error":
                        st.warning("TLS/certificate validation failed for this URL.")
                    elif status == "connection_error":
                        st.warning("Could not connect to remote host.")
                    elif status == "too_many_redirects":
                        st.warning("Too many redirects for safe scanning.")
                    elif status == "oversized_file":
                        st.warning("Remote file exceeded safe size limit.")
                    elif status == "unsupported_content_type":
                        st.warning("Link returned unsupported/unclear file type for attachment scanning.")
                    else:
                        st.warning("Remote fetch failed. Please try a direct downloadable file URL.")
                    dbg = result.get("debug") or {}
                    if dbg:
                        with st.expander("Debug details (remote fetch)"):
                            st.json(dbg)
                else:
                    local_path = result.get("local_path")
                    st.success(f"File downloaded successfully: {result.get('file_name')} ({result.get('bytes', 0)} bytes)")
                    file_risk, file_reasons = analyze_attachment_file(local_path)
                    st.write(f"**File Risk Level:** {file_risk}")
                    with st.expander("View File Analysis Details"):
                        for reason in file_reasons:
                            st.write("-", reason)

                    if "HIGH" in str(file_risk).upper():
                        st.error("File appears malicious.")
                    elif "MEDIUM" in str(file_risk).upper():
                        st.warning("File may be suspicious.")
                    else:
                        st.success("File appears safe.")

            except Exception as e:
                st.error(f"Could not complete remote file scan safely: {e}")

            finally:
                try:
                    if local_path and os.path.exists(local_path):
                        os.remove(local_path)
                        st.info("Temporary file deleted after scan.")
                except Exception:
                    st.info("Temporary file cleanup could not be confirmed.")
