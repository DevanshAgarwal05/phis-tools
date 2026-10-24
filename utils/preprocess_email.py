# utils/preprocess_email.py
import email
import re

def extract_email_parts(file):
    msg = email.message_from_bytes(file.read())
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")

    urls = re.findall(r'https?://[^\s]+', body)
    return body, urls
