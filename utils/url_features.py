import re
import urllib.parse
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def analyze_url(url):
    score = 0
    reasons = []

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    # 1️⃣ IP address usage
    if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
        score += 1
        reasons.append("Uses IP address instead of domain name")

    # 2️⃣ Suspicious TLD
    if re.search(r"\.(ru|cn|xyz|top|tk|ml|ga|cf|gq|zip|click)$", domain):
        score += 1
        reasons.append(f"Suspicious TLD: {domain}")

    # 3️⃣ Long / complex domain
    if len(domain) > 60 or domain.count('.') > 3:
        score += 1
        reasons.append("Overly long or nested subdomains")

    # 4️⃣ Encoding / punycode
    if "xn--" in domain or "%" in url:
        score += 1
        reasons.append("Punycode or encoded URL")

    # 5️⃣ '@' symbol trick
    if "@" in url:
        score += 1
        reasons.append("Contains '@' (possible redirection)")

    # 6️⃣ Typosquatting & brand mimicry
    known_brands = [
        "microsoft", "windows", "paypal", "google", "facebook",
        "instagram", "amazon", "apple", "outlook", "netflix"
    ]

    substitutions = {"0": "o", "1": "l", "3": "e", "5": "s", "7": "t"}
    simplified = domain
    for k, v in substitutions.items():
        simplified = simplified.replace(k, v)

    for brand in known_brands:
        sim_ratio = similar(simplified, brand + ".com")
        if sim_ratio > 0.65:  # lowered threshold ✅
            score += 2
            reasons.append(f"Possible typosquatted brand '{brand}' detected ({domain})")
            break

    # 7️⃣ Deceptive words
    if any(k in domain for k in ["login", "secure", "update", "verify", "banking", "signin"]):
        score += 1
        reasons.append("Contains deceptive keyword")

    # Final classification
    if score >= 3:
        risk = "🚨 HIGH RISK"
    elif score == 2:
        risk = "⚠️ MEDIUM RISK"
    else:
        risk = "✅ LOW RISK"

    return risk, reasons
