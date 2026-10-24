# analyzers/network/attachment_file_analyzer.py
import magic
import math
import os
import mimetypes

def shannon_entropy(data):
    """Compute Shannon entropy for given binary data."""
    if not data:
        return 0
    byte_count = [0] * 256
    for b in data:
        byte_count[b] += 1
    entropy = 0
    for count in byte_count:
        if count:
            p = float(count) / len(data)
            entropy -= p * math.log2(p)
    return entropy

def analyze_attachment_file(file_path):
    """Analyze uploaded file by MIME type, extension mismatch, and entropy."""
    reasons = []

    if not os.path.exists(file_path):
        return "File not found", ["Attachment not accessible."]

    try:
        mime = magic.from_file(file_path, mime=True)
    except Exception as e:
        guess, _ = mimetypes.guess_type(file_path)
        if guess:
            mime = guess
            reasons.append(f"MIME library failed; used extension guess: {guess}")
        else:
            return "Unknown", [f"MIME detection error: {e}"]

    _, ext = os.path.splitext(file_path.lower())

    # Read first 2KB for magic header + entropy
    with open(file_path, "rb") as f:
        data = f.read(2048)

    entropy = shannon_entropy(data)

    expected_types = {
        ".pdf": "application/pdf",
        ".zip": "application/zip",
        ".exe": "application/x-dosexec",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".jpg": "image/jpeg",
        ".png": "image/png",
    }

    expected_mime = expected_types.get(ext)
    if expected_mime and expected_mime not in mime:
        reasons.append(f"Extension mismatch: File claims {ext} but MIME is {mime}")

    if entropy > 7.5:
        reasons.append(f"High entropy ({entropy:.2f}) - possible packed or encrypted content.")

    if "application/x-dosexec" in mime or ext == ".exe":
        risk = "HIGH RISK"
    elif entropy > 7.5 or "mismatch" in " ".join(reasons):
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return risk, reasons + [f"Detected MIME: {mime}", f"Entropy: {entropy:.2f}"]
