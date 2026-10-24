# utils.py
import re
from urllib.parse import urlparse
import tldextract
from email import policy
from email.parser import BytesParser

URL_REGEX = re.compile(r'https?://[^\s"\'<>]+', re.IGNORECASE)

def parse_eml(path):
    """Return (headers_dict, body_text, urls_list)"""
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    headers = dict(msg.items())
    body = ""
    # Collect both plain and html parts
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            try:
                content = part.get_content()
            except Exception:
                content = ""
            if ctype == 'text/html':
                body += content
            elif ctype == 'text/plain':
                body += content
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = ""
    urls = re.findall(URL_REGEX, str(body))
    # normalize urls (strip trailing punctuation)
    urls = [u.rstrip('.,;:!?)"\']') for u in urls]
    return headers, str(body), urls

def url_features(url: str):
    """Return a dict of numeric/bool features for a single url string"""
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    feats = {}
    feats['length'] = len(url)
    feats['uses_https'] = int(parsed.scheme == 'https')
    feats['has_ip'] = int(bool(re.match(r'https?://\d+\.\d+\.\d+\.\d+', url)))
    feats['num_dots'] = url.count('.')
    feats['num_slashes'] = url.count('/')
    feats['has_at'] = int('@' in url)
    feats['has_dash_domain'] = int('-' in (ext.domain or ''))
    feats['subdomain_parts'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    feats['path_len'] = len(parsed.path or '')
    feats['query_len'] = len(parsed.query or '')
    feats['punycode'] = int('xn--' in url.lower())
    feats['uses_port'] = int(bool(parsed.port))
    return feats

def suspicious_keywords_in_text(text: str):
    kws = ['verify', 'password', 'login', 'bank', 'account', 'click', 'urgent', 'security', 'suspend', 'update', 'confirm']
    text_lower = (text or '').lower()
    return { f'kw_{k}': int(k in text_lower) for k in kws }

def header_mismatch(headers):
    """Simple heuristic: does Reply-To or Return-Path differ from From domain?"""
    frm = headers.get('From','').lower()
    reply = headers.get('Reply-To','').lower()
    ret = headers.get('Return-Path','').lower()
    def domain_of(s):
        try:
            return s.split('@')[-1].strip()
        except:
            return ''
    frm_dom = domain_of(frm)
    reply_dom = domain_of(reply)
    ret_dom = domain_of(ret)
    mismatch = 0
    if reply_dom and reply_dom != frm_dom:
        mismatch = 1
    if ret_dom and ret_dom != frm_dom:
        mismatch = 1
    return mismatch
