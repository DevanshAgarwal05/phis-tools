"""Golden-output regression test for the Streamlit app.

Runs webapp.py headlessly (streamlit AppTest) on fixed scenarios and compares every
rendered element with tests/golden_baseline.json (captured from the pre-refactor code).

    pytest tests/test_golden.py            # compare
    python tests/test_golden.py --update   # re-capture baseline (only for intentional changes)
    python tests/test_golden.py out.json   # dump to a file
"""
import json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "golden_baseline.json"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from streamlit.testing.v1 import AppTest

CASES = [
    dict(name="safe", subject="Project meeting rescheduled to Friday",
         body="Hi team,\n\nThe project review meeting has been moved to Friday at 3 PM in Lab 2.\n\nRegards,\nAnanya",
         headers="From: ananya@vit.ac.in\nReply-To: ananya@vit.ac.in\nReturn-Path: ananya@vit.ac.in", urls="https://vit.ac.in"),
    dict(name="urgent", subject="Urgent: Your university account will be suspended today",
         body="Dear Student,\n\nYour account has been flagged. You must verify your account immediately or access will be permanently suspended today.\nClick below to verify now:",
         headers="From: support@vit-security.com\nReply-To: verifyaccount@secure-login-alert.com\nReturn-Path: noreply@mailer-check.net",
         urls="http://secure-login-verification-paypal.com-account-update.xyz/login\nhttp://192.168.1.10/login\nhttps://bit.ly/3abcde"),
    dict(name="urls_only", subject="", body="", headers="", urls="WWW.Example.com/Path/\nhttp://xn--pypal-4ve.com/verify?otp=1"),
    dict(name="strict", policy="Strict Policy (Banking / Enterprise)", subject="Invoice payment", body="Please confirm your bank password to reset", headers="", urls="http://login-bank-update.tk"),
    dict(name="relaxed", policy="Relaxed Policy (Personal Use)", subject="Invoice payment", body="Please confirm your bank password to reset", headers="", urls="http://login-bank-update.tk"),
    dict(name="empty", subject="", body="", headers="", urls=""),
]
TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?")

def dump(node, out):
    for c in getattr(node, "children", {}).values():
        t = getattr(c, "type", type(c).__name__)
        val = getattr(c, "value", None)
        if val is None: val = getattr(c, "body", None)
        if val is None: val = getattr(c, "label", None)
        out.append([t, TS.sub("<TS>", str(val))])
        dump(c, out)

def run():
    results = {}
    log = "logs/scan_logs.csv"; bak = log + ".golden_bak"
    shutil.copyfile(log, bak)
    try:
        for case in CASES:
            shutil.copyfile(bak, log)
            at = AppTest.from_file("webapp.py", default_timeout=300).run()
            assert not at.exception, [e.value for e in at.exception]
            if "policy" in case:
                at.selectbox[0].select(case["policy"]).run()
            ti = at.text_input; ta = at.text_area
            ti[0].set_value(case["subject"]); ta[0].set_value(case["body"])
            ta[1].set_value(case["headers"]); ta[2].set_value(case["urls"])
            at.button[0].click().run()
            out = []; dump(at.main, out)
            results[case["name"]] = dict(exceptions=[e.value for e in at.exception], elements=out)
    finally:
        shutil.copyfile(bak, log); os.remove(bak)
    return results

def test_golden_output_unchanged():
    got = json.loads(json.dumps(run(), sort_keys=True))
    want = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert sorted(got) == sorted(want)
    for name in want:
        assert got[name] == want[name], f"golden drift in case '{name}'"


if __name__ == "__main__":
    out = BASELINE if sys.argv[1:] == ["--update"] else Path(sys.argv[1])
    out.write_text(json.dumps(run(), indent=1, sort_keys=True), encoding="utf-8")
    print("wrote", out)
