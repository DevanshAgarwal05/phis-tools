# utils/url_ml_features.py
import urllib.parse
import re

def extract_url_features(url: str) -> dict:
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
    except:
        domain = url.lower()

    features = {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_letters": sum(c.isalpha() for c in url),
        "num_special": len(re.findall(r"[^A-Za-z0-9]", url)),
        "num_subdomains": domain.count('.'),
        "has_https": int(url.lower().startswith("https")),
        "has_ip": int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", domain))),
        "contains_at": int("@" in url),
        "contains_dash": int("-" in domain),
        "contains_equal": int("=" in url),
        "contains_question": int("?" in url),
        "contains_login": int("login" in url.lower()),
        "contains_verify": int("verify" in url.lower()),
        "contains_secure": int("secure" in url.lower()),
        "suspicious_tld": int(bool(re.search(r"\.(ru|cn|xyz|top|tk|ml|ga|cf|gq|zip|click)$", domain))),
    }
    return features
