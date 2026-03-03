import io
import re
import math
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Negative Accounts Extractor",
    page_icon="📄",
    layout="wide",
)

TODAY = date.today()

# -----------------------------
# Safety / PII masking helpers
# -----------------------------
ACCT_LIKE_RE = re.compile(r"(?<!\d)(?:X{2,}|\*{2,}|\d)[Xx\*\-\s\d]{5,}(?!\d)")
DIGIT_RUN_RE = re.compile(r"(?<!\d)(\d[\d\-\s]{6,}\d)(?!\d)")


def mask_account_number(raw: str) -> str:
    """
    Keep only last 4 digits if present. If no digits, return masked token.
    Examples:
      '1234567890' -> '••••6789'
      'XXXXXX1234' -> '••••1234'
      '***-**-1234' -> '••••1234'
    """
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return f"••••{digits[-4:]}"
    return "••••"


def mask_pii_in_snippet(text: str) -> str:
    """
    Mask account-like sequences in debug snippets. Keeps last 4 digits.
    """
    if not text:
        return ""

    def repl(m):
        token = m.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 4:
            return f"••••{digits[-4:]}"
        return "••••"

    # Mask explicit account-like strings (X/*/digits runs)
    text = ACCT_LIKE_RE.sub(repl, text)
    # Mask long digit runs too
    text = DIGIT_RUN_RE.sub(repl, text)
    return text


# -----------------------------
# Date parsing helpers
# -----------------------------
DATE_PATTERNS = [
    # mm/dd/yyyy or m/d/yyyy
    re.compile(r"\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-]((?:19|20)\d{2})\b"),
    # mm/yyyy or m/yyyy
    re.compile(r"\b(0?[1-9]|1[0-2])[\/\-]((?:19|20)\d{2})\b"),
    # yyyy-mm-dd
    re.compile(r"\b((?:19|20)\d{2})[\/\-](0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])\b"),
]


def parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    t = text.strip()

    # Try mm/dd/yyyy
    m = DATE_PATTERNS[0].search(t)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        try:
            return date(int(yyyy), int(mm), int(dd))
        except ValueError:
            return None

    # Try mm/yyyy -> assume day=1 (conservative)
    m = DATE_PATTERNS[1].search(t)
    if m:
        mm, yyyy = m.group(1), m.group(2)
        try:
            return date(int(yyyy), int(mm), 1)
        except ValueError:
            return None

    # Try yyyy-mm-dd
    m = DATE_PATTERNS[2].search(t)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        try:
            return date(int(yyyy), int(mm), int(dd))
        except ValueError:
            return None

    return None


def format_date(d: Optional[date]) -> str:
    if not d:
        return ""
    return d.strftime("%Y-%m-%d")


def calc_age(opened: Optional[date]) -> str:
    if not opened:
        return ""
    # Age in years/months (approx, but stable)
    months = (TODAY.year - opened.year) * 12 + (TODAY.month - opened.month)
    if months < 0:
        return ""
    years = months // 12
    rem = months % 12
    if years <= 0:
        return f"{rem} mo"
    if rem == 0:
        return f"{years} yr"
    return f"{years} yr {rem} mo"


# -----------------------------
# Money parsing helpers
# -----------------------------
MONEY_RE = re.compile(r"(?i)\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)")


def parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    m = MONEY_RE.search(text.replace("O", "0"))
    if not m:
        return None
    val = m.group(1).replace(",", "")
    try:
        return float(val)
    except ValueError:
        return None


def format_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return ""
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return ""


# -----------------------------
# Bureau detection
# -----------------------------
def detect_bureau(text: str) -> List[str]:
    t = (text or "").lower()
    bureaus = []
    if "experian" in t:
        bureaus.append("Experian")
    if "equifax" in t:
        bureaus.append("Equifax")
    if "transunion" in t or "trans union" in t:
        bureaus.append("TransUnion")
    return bureaus


def detect_bureau_for_block(block: str, doc_bureaus: List[str]) -> str:
    """
    Best effort: detect per-block bureau indicators; fallback to doc-level.
    """
    t = (block or "").lower()
    found = []
    if "experian" in t or re.search(r"\bex\b", t):
        found.append("Experian")
    if "equifax" in t or re.search(r"\beq\b", t):
        found.append("Equifax")
    if "transunion" in t or "trans union" in t or re.search(r"\btu\b", t):
        found.append("TransUnion")

    # If nothing found, use doc-level (if single bureau, assign it)
    if not found:
        if len(doc_bureaus) == 1:
            return doc_bureaus[0]
        if len(doc_bureaus) > 1:
            return "Multiple/Unknown"
        return "Unknown"

    # Deduplicate + return
    found = list(dict.fromkeys(found))
    if len(found) == 1:
        return found[0]
    return "Multiple/Unknown"


# -----------------------------
# Negative detection + parsing
# -----------------------------
NEG_KEYWORDS = [
    "collection", "collections", "charge off", "charged off", "charge-off", "charged-off",
    "late", "past due", "delinquent", "30 days late", "60 days late", "90 days late", "120 days late",
    "repossession", "repo", "repossessed",
    "bankruptcy", "bk", "chapter 7", "chapter 13",
    "foreclosure",
    "judgment", "judgements",
    "write off", "written off", "write-off",
]

# For status normalization
STATUS_MAP = [
    ("Bankruptcy", ["bankruptcy", "chapter 7", "chapter 13", "bk"]),
    ("Foreclosure", ["foreclosure"]),
    ("Judgment", ["judgment", "judgements"]),
    ("Repossession", ["repossession", "repossessed", " repo "]),
    ("Charge-off", ["charge off", "charged off", "charge-off", "charged-off", "write off", "written off", "write-off"]),
    ("Collection", ["collection", "collections"]),
    ("Late (120+)", ["120 days late", "120+", "120 day"]),
    ("Late (90)", ["90 days late", "90 day"]),
    ("Late (60)", ["60 days late", "60 day"]),
    ("Late (30)", ["30 days late", "30 day"]),
    ("Late/Delinquent", ["late", "past due", "delinquent"]),
]

LABEL_PATTERNS = {
    "creditor": [
        re.compile(r"(?i)\b(creditor|subscriber|furnisher|company|lender|collection agency|original creditor)\b\s*[:\-]\s*(.+)"),
        re.compile(r"(?i)\b(account name|name)\b\s*[:\-]\s*(.+)"),
    ],
    "account": [
        re.compile(r"(?i)\b(acct|acct\.|account|account\s*#|account\s*number|a\/c|a\/c\s*#)\b\s*[:\-]?\s*([Xx\*\-\s\d]{4,})"),
    ],
    "balance": [
        re.compile(r"(?i)\b(current\s*balance|balance|bal)\b\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)"),
        re.compile(r"(?i)\b(amount\s*owed)\b\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)"),
    ],
    "last_reported": [
        re.compile(r"(?i)\b(last\s*reported|date\s*reported|reported\s*on|last\s*updated|last\s*update)\b\s*[:\-]?\s*([0-9\/\-]{4,10})"),
    ],
    "opened": [
        re.compile(r"(?i)\b(date\s*opened|opened|open\s*date)\b\s*[:\-]?\s*([0-9\/\-]{4,10})"),
    ],
    "status": [
        re.compile(r"(?i)\b(status|account\s*status|condition|remarks?)\b\s*[:\-]?\s*(.+)"),
    ],
}

# Common “block delimiter” markers in bureau reports
BLOCK_BREAK_HINTS = [
    "account information",
    "account details",
    "trade line",
    "tradeline",
    "collection",
    "public record",
    "inquiries",
    "personal information",
]


@dataclass
class ParseNote:
    block_index: int
    notes: List[str]


@dataclass
class NegativeAccount:
    creditor_name: str = ""
    masked_account_number: str = ""
    current_balance: Optional[float] = None
    last_reported_date: Optional[date] = None
    date_opened: Optional[date] = None
    age_of_account: str = ""
    negative_type_status: str = ""
    bureaus: str = ""
    estimated_impact: str = ""
    raw_block_snippet: str = ""


def normalize_status(block_text: str) -> str:
    t = f" { (block_text or '').lower() } "
    for label, keys in STATUS_MAP:
        for k in keys:
            if k in t:
                return label
    # fallback: look for explicit late codes like 30/60/90/120
    if re.search(r"\b(30|60|90|120)\b", t) and "late" in t:
        return "Late/Delinquent"
    return ""


def looks_negative(block_text: str) -> bool:
    t = (block_text or "").lower()
    return any(k in t for k in NEG_KEYWORDS)


def clean_creditor_name(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r"\s+", " ", name).strip()
    # strip obvious label residue
    n = re.sub(r"(?i)^(creditor|subscriber|furnisher|company|lender|collection agency|original creditor)\s*[:\-]\s*", "", n).strip()
    # remove trailing junk if it's mostly punctuation
    n = re.sub(r"[|•]+$", "", n).strip()
    return n[:80]


def extract_first_match(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for p in patterns:
        m = p.search(text or "")
        if m:
            # return last capturing group usually (value)
            return m.group(m.lastindex or 1).strip()
    return None


def guess_creditor_from_lines(lines: List[str]) -> str:
    """
    Heuristic: pick the first non-empty line that is not a label and isn't just numbers/dates.
    """
    for ln in lines[:8]:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(x in low for x in ["account", "acct", "balance", "reported", "opened", "status", "remarks", "payment", "limit", "high credit"]):
            continue
        # skip lines that look like just money/date/number soup
        if re.fullmatch(r"[\d\W]+", s):
            continue
        # avoid super long paragraphs
        if len(s) > 90:
            continue
        return clean_creditor_name(s)
    return ""


def extract_account_fields(block: str, block_index: int, doc_bureaus: List[str]) -> Tuple[NegativeAccount, ParseNote]:
    notes = []

    raw = (block or "").strip()
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]

    # Creditor
    creditor = extract_first_match(LABEL_PATTERNS["creditor"], raw)
    if creditor:
        creditor = clean_creditor_name(creditor)
        notes.append("Creditor: label match")
    else:
        creditor = guess_creditor_from_lines(lines)
        if creditor:
            notes.append("Creditor: line heuristic")
        else:
            notes.append("Creditor: not found")

    # Account number
    acct_raw = extract_first_match(LABEL_PATTERNS["account"], raw)
    if acct_raw:
        notes.append("Account#: label match")
    else:
        # fallback: find any long account-like token
        m = ACCT_LIKE_RE.search(raw)
        acct_raw = m.group(0).strip() if m else ""
        if acct_raw:
            notes.append("Account#: token heuristic")
        else:
            notes.append("Account#: not found")

    masked_acct = mask_account_number(acct_raw)

    # Balance
    bal_raw = extract_first_match(LABEL_PATTERNS["balance"], raw)
    bal = None
    if bal_raw:
        bal = parse_money(bal_raw)
        notes.append("Balance: label match")
    else:
        # fallback: try first money value near word "balance" or "owed"
        nearby = ""
        m = re.search(r"(?i)\b(balance|amount\s*owed|current\s*balance)\b(.{0,60})", raw)
        if m:
            nearby = m.group(0)
            bal = parse_money(nearby)
        if bal is not None:
            notes.append("Balance: nearby heuristic")
        else:
            notes.append("Balance: not found")

    # Dates
    last_raw = extract_first_match(LABEL_PATTERNS["last_reported"], raw)
    opened_raw = extract_first_match(LABEL_PATTERNS["opened"], raw)

    last_dt = parse_date_from_text(last_raw or "")
    if last_dt:
        notes.append("Last reported: label match")
    else:
        # fallback: search for a date near 'reported' / 'updated'
        m = re.search(r"(?i)\b(reported|updated)\b(.{0,60})", raw)
        last_dt = parse_date_from_text(m.group(0)) if m else None
        notes.append("Last reported: fallback search" if last_dt else "Last reported: not found")

    opened_dt = parse_date_from_text(opened_raw or "")
    if opened_dt:
        notes.append("Opened: label match")
    else:
        m = re.search(r"(?i)\b(opened|date\s*opened|open\s*date)\b(.{0,60})", raw)
        opened_dt = parse_date_from_text(m.group(0)) if m else None
        notes.append("Opened: fallback search" if opened_dt else "Opened: not found")

    # Status / Negative type
    status_raw = extract_first_match(LABEL_PATTERNS["status"], raw)
    if status_raw:
        notes.append("Status: label match")
    status = normalize_status(raw if not status_raw else f"{raw}\n{status_raw}")
    if status:
        notes.append(f"Negative type: {status}")
    else:
        notes.append("Negative type: not found")

    bureaus = detect_bureau_for_block(raw, doc_bureaus)

    acct = NegativeAccount(
        creditor_name=creditor,
        masked_account_number=masked_acct,
        current_balance=bal,
        last_reported_date=last_dt,
        date_opened=opened_dt,
        age_of_account=calc_age(opened_dt),
        negative_type_status=status,
        bureaus=bureaus,
        estimated_impact="",  # set later
        raw_block_snippet=mask_pii_in_snippet(raw[:1400]),
    )
    return acct, ParseNote(block_index=block_index, notes=notes)


def build_blocks_from_text(full_text: str) -> List[str]:
    """
    Block detection:
    1) Split by double newlines as rough paragraphs.
    2) If paragraphs are too large or too few, fall back to sliding-window blocks around negative keyword lines.
    """
    t = (full_text or "").replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]

    # If we got a reasonable number of paragraphs, keep them.
    if 8 <= len(paras) <= 400:
        return paras

    # Otherwise: line-based windows around negative hits
    lines = [ln for ln in t.splitlines()]
    hit_idxs = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in NEG_KEYWORDS):
            hit_idxs.append(i)

    windows = []
    for idx in hit_idxs:
        start = max(0, idx - 8)
        end = min(len(lines), idx + 10)
        windows.append((start, end))

    # merge overlaps
    windows.sort()
    merged = []
    for s, e in windows:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    blocks = []
    for s, e in merged:
        blk = "\n".join(lines[s:e]).strip()
        if blk:
            blocks.append(blk)

    # Final fallback: whole text
    return blocks if blocks else [t.strip()]


# -----------------------------
# Extraction pipeline: Text then OCR
# -----------------------------
def extract_text_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    doc.close()
    return "\n".join(parts).strip()


def extract_text_ocr(pdf_bytes: bytes, dpi: int = 220) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    zoom = dpi / 72.0  # 72 dpi base
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img)
        parts.append(txt or "")

    doc.close()
    return "\n".join(parts).strip()


def compute_confidence(extracted_text: str, accounts: List[NegativeAccount]) -> Tuple[str, Dict[str, float]]:
    """
    Confidence based on:
    - text length
    - field completion rate for key fields
    """
    tlen = len((extracted_text or "").strip())
    n = max(1, len(accounts))

    def filled(v) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    key_fields = ["creditor_name", "masked_account_number", "current_balance", "negative_type_status", "last_reported_date"]
    filled_counts = {k: 0 for k in key_fields}

    for a in accounts:
        for k in key_fields:
            v = getattr(a, k)
            filled_counts[k] += 1 if filled(v) else 0

    completion = sum(filled_counts[k] / n for k in key_fields) / len(key_fields)

    # length score
    if tlen >= 12000:
        len_score = 1.0
    elif tlen >= 5000:
        len_score = 0.7
    elif tlen >= 2000:
        len_score = 0.45
    else:
        len_score = 0.25

    score = 0.55 * completion + 0.45 * len_score

    if score >= 0.78:
        label = "High"
    elif score >= 0.55:
        label = "Medium"
    else:
        label = "Low"

    debug = {
        "text_length": float(tlen),
        "completion_rate": float(completion),
        "score": float(score),
    }
    return label, debug


# -----------------------------
# Impact estimator (heuristics)
# -----------------------------
def impact_range_for_account(
    neg_type: str,
    last_reported: Optional[date],
    balance: Optional[float],
) -> Tuple[str, str]:
    """
    Returns: (tier_label, range_str)
    - No exact scores. Realistic ranges.
    - Uses type + recency + balance severity.
    """
    t = (neg_type or "").lower()
    bal = balance or 0.0

    # recency buckets
    recency = "unknown"
    months = None
    if last_reported:
        months = (TODAY.year - last_reported.year) * 12 + (TODAY.month - last_reported.month)
        if months <= 12:
            recency = "recent"
        elif months <= 24:
            recency = "mid"
        else:
            recency = "old"

    # balance severity
    if bal >= 10000:
        bsev = "high"
    elif bal >= 2000:
        bsev = "med"
    else:
        bsev = "low"

    # base by type
    if "bankruptcy" in t:
        base = (60, 180)
        tier = "Severe"
    elif "foreclosure" in t:
        base = (50, 160)
        tier = "Severe"
    elif "judgment" in t:
        base = (35, 110)
        tier = "High"
    elif "repossession" in t:
        base = (40, 120)
        tier = "High"
    elif "charge-off" in t:
        base = (25, 90)
        tier = "High"
    elif "collection" in t:
        base = (15, 70)
        tier = "Moderate"
    elif "late (120" in t:
        base = (20, 80)
        tier = "High"
    elif "late (90" in t:
        base = (18, 70)
        tier = "Moderate"
    elif "late (60" in t:
        base = (12, 45)
        tier = "Moderate"
    elif "late (30" in t:
        base = (5, 25)
        tier = "Low"
    elif "late" in t or "delinquent" in t:
        base = (8, 35)
        tier = "Low–Moderate"
    else:
        base = (8, 35)
        tier = "Low–Moderate"

    lo, hi = base

    # adjust by recency
    if recency == "recent":
        lo = int(lo * 1.15)
        hi = int(hi * 1.20)
    elif recency == "mid":
        lo = int(lo * 1.00)
        hi = int(hi * 1.00)
    elif recency == "old":
        lo = int(lo * 0.70)
        hi = int(hi * 0.75)
    else:
        lo = int(lo * 0.90)
        hi = int(hi * 0.90)

    # adjust by balance (collections/charge-offs more sensitive to size)
    if "collection" in t or "charge-off" in t or "repossession" in t or "judgment" in t:
        if bsev == "high":
            lo = int(lo * 1.15)
            hi = int(hi * 1.20)
        elif bsev == "med":
            lo = int(lo * 1.05)
            hi = int(hi * 1.08)
        else:
            lo = int(lo * 0.95)
            hi = int(hi * 0.95)

    # clamp sanity
    lo = max(0, lo)
    hi = max(lo + 5, hi)

    return tier, f"{lo}–{hi} pts"


def overall_negative_pressure(accounts: List[NegativeAccount]) -> str:
    """
    Not additive. A qualitative “pressure” indicator.
    """
    if not accounts:
        return "None detected"

    # weight by type
    w = 0.0
    for a in accounts:
        t = (a.negative_type_status or "").lower()
        if "bankruptcy" in t or "foreclosure" in t:
            w += 3.5
        elif "repossession" in t or "judgment" in t:
            w += 2.8
        elif "charge-off" in t:
            w += 2.3
        elif "collection" in t:
            w += 1.6
        elif "late (120" in t:
            w += 1.9
        elif "late (90" in t:
            w += 1.5
        elif "late (60" in t:
            w += 1.1
        elif "late (30" in t:
            w += 0.7
        elif "late" in t or "delinquent" in t:
            w += 0.8
        else:
            w += 0.6

        # small bump for large balances
        bal = a.current_balance or 0.0
        if bal >= 10000:
            w += 0.5
        elif bal >= 2000:
            w += 0.2

        # bump for recent reporting
        if a.last_reported_date:
            months = (TODAY.year - a.last_reported_date.year) * 12 + (TODAY.month - a.last_reported_date.month)
            if months <= 12:
                w += 0.4
            elif months <= 24:
                w += 0.2

    # map weight to category
    if w >= 12:
        return "Very High negative pressure"
    if w >= 7:
        return "High negative pressure"
    if w >= 3.5:
        return "Moderate negative pressure"
    return "Low negative pressure"


# -----------------------------
# Main parse orchestrator
# -----------------------------
def parse_negative_accounts(extracted_text: str) -> Tuple[List[NegativeAccount], List[str], List[str], List[ParseNote], List[str]]:
    doc_bureaus = detect_bureau(extracted_text)
    blocks = build_blocks_from_text(extracted_text)

    negative_accounts: List[NegativeAccount] = []
    notes: List[ParseNote] = []
    negative_blocks: List[str] = []

    for i, blk in enumerate(blocks):
        if not looks_negative(blk):
            continue

        acct, n = extract_account_fields(blk, i, doc_bureaus)

        # Keep only if we have at least a status AND a creditor or balance (avoid random matches)
        has_minimum = bool(acct.negative_type_status) and (bool(acct.creditor_name) or (acct.current_balance is not None))
        if not has_minimum:
            # still store note for debug
            notes.append(ParseNote(block_index=i, notes=n.notes + ["Result: skipped (insufficient fields)"]))
            continue

        # impact
        tier, rng = impact_range_for_account(acct.negative_type_status, acct.last_reported_date, acct.current_balance)
        acct.estimated_impact = f"{tier}: {rng}"

        negative_accounts.append(acct)
        negative_blocks.append(blk)
        notes.append(n)

    # dedupe accounts by (creditor, acct last4, type, balance approx)
    deduped = []
    seen = set()
    for a in negative_accounts:
        key = (
            (a.creditor_name or "").lower().strip()[:40],
            a.masked_account_number,
            (a.negative_type_status or "").lower(),
            int((a.current_balance or 0.0) // 10)  # coarse bucket
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    debug_warnings = []
    if not doc_bureaus:
        debug_warnings.append("Bureau detection: none found (normal for some report formats).")
    if len(blocks) == 1 and len(blocks[0]) > 20000:
        debug_warnings.append("Block detection: report did not split cleanly; using fallback windowing may improve reliability on some formats.")

    return deduped, doc_bureaus, blocks, notes, debug_warnings


# -----------------------------
# PDF export (ReportLab)
# -----------------------------
def build_export_pdf(df: pd.DataFrame, extraction_mode: str, confidence: str, overall_pressure: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Negative Accounts Summary", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    meta = f"Extraction Mode: {extraction_mode} &nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {confidence} &nbsp;&nbsp;|&nbsp;&nbsp; Overall Negative Pressure: {overall_pressure}"
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 0.20 * inch))

    disclaimer = (
        "Disclaimer: This report is an informational extraction from the uploaded credit bureau PDF. "
        "Estimated impact ranges are heuristic, not an exact score change, and vary by scoring model (e.g., FICO vs Vantage), "
        "credit file thickness, overall utilization, and reporting details. Always verify against the source report and/or a qualified professional."
    )
    story.append(Paragraph(disclaimer, styles["Italic"]))
    story.append(Spacer(1, 0.25 * inch))

    if df.empty:
        story.append(Paragraph("No negative accounts detected.", styles["Normal"]))
        doc.build(story)
        return buffer.getvalue()

    cols = [
        "Creditor",
        "Acct (Last 4)",
        "Balance",
        "Last Reported",
        "Opened",
        "Age",
        "Negative Type/Status",
        "Bureau(s)",
        "Estimated Impact",
    ]
    table_data = [cols]

    def safe(v):
        return "" if v is None else str(v)

    for _, r in df.iterrows():
        table_data.append([
            safe(r.get("Creditor")),
            safe(r.get("Acct (Last 4)")),
            safe(r.get("Balance")),
            safe(r.get("Last Reported")),
            safe(r.get("Opened")),
            safe(r.get("Age")),
            safe(r.get("Negative Type/Status")),
            safe(r.get("Bureau(s)")),
            safe(r.get("Estimated Impact")),
        ])

    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)

    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# UI / State
# -----------------------------
def to_table_df(accounts: List[NegativeAccount]) -> pd.DataFrame:
    rows = []
    for a in accounts:
        rows.append({
            "Creditor": a.creditor_name,
            "Acct (Last 4)": a.masked_account_number,
            "Balance": format_money(a.current_balance),
            "Last Reported": format_date(a.last_reported_date),
            "Opened": format_date(a.date_opened),
            "Age": a.age_of_account,
            "Negative Type/Status": a.negative_type_status,
            "Bureau(s)": a.bureaus,
            "Estimated Impact": a.estimated_impact,
        })
    return pd.DataFrame(rows)


def build_manual_editor_df(accounts: List[NegativeAccount]) -> pd.DataFrame:
    """
    Editable fields (no full PII):
      - Creditor, Acct last4, balance, dates, status, bureaus
    """
    rows = []
    for idx, a in enumerate(accounts):
        rows.append({
            "RowID": idx,
            "Creditor": a.creditor_name,
            "Acct (Last 4)": a.masked_account_number,
            "Balance (number)": float(a.current_balance) if a.current_balance is not None else None,
            "Last Reported (YYYY-MM-DD)": format_date(a.last_reported_date),
            "Opened (YYYY-MM-DD)": format_date(a.date_opened),
            "Negative Type/Status": a.negative_type_status,
            "Bureau(s)": a.bureaus,
        })
    return pd.DataFrame(rows)


def apply_manual_overrides(base_accounts: List[NegativeAccount], edited_df: pd.DataFrame) -> List[NegativeAccount]:
    updated = []
    by_id = {int(r["RowID"]): r for _, r in edited_df.iterrows() if pd.notnull(r.get("RowID"))}

    for i, a in enumerate(base_accounts):
        r = by_id.get(i)
        if r is None:
            updated.append(a)
            continue

        creditor = str(r.get("Creditor") or "").strip()
        acct_last4 = str(r.get("Acct (Last 4)") or "").strip()
        bal_num = r.get("Balance (number)")
        last_s = str(r.get("Last Reported (YYYY-MM-DD)") or "").strip()
        open_s = str(r.get("Opened (YYYY-MM-DD)") or "").strip()
        status = str(r.get("Negative Type/Status") or "").strip()
        bureaus = str(r.get("Bureau(s)") or "").strip()

        # parse dates if provided
        last_dt = None
        if last_s:
            try:
                last_dt = datetime.strptime(last_s, "%Y-%m-%d").date()
            except ValueError:
                last_dt = parse_date_from_text(last_s)

        open_dt = None
        if open_s:
            try:
                open_dt = datetime.strptime(open_s, "%Y-%m-%d").date()
            except ValueError:
                open_dt = parse_date_from_text(open_s)

        bal = None
        try:
            if bal_num is not None and not (isinstance(bal_num, float) and math.isnan(bal_num)):
                bal = float(bal_num)
        except Exception:
            bal = None

        # impact re-estimate
        tier, rng = impact_range_for_account(status, last_dt, bal)
        est = f"{tier}: {rng}" if status else ""

        updated.append(NegativeAccount(
            creditor_name=creditor,
            masked_account_number=acct_last4 if acct_last4 else a.masked_account_number,
            current_balance=bal,
            last_reported_date=last_dt,
            date_opened=open_dt,
            age_of_account=calc_age(open_dt),
            negative_type_status=status,
            bureaus=bureaus,
            estimated_impact=est,
            raw_block_snippet=a.raw_block_snippet,  # keep original snippet
        ))
    return updated


# -----------------------------
# Header
# -----------------------------
st.title("📄 Negative Accounts Extractor (PDF → Structured Table)")
st.caption("Uploads are processed in-memory only. Account numbers are masked by default. OCR fallback enabled for scanned PDFs.")


uploaded = st.file_uploader("Upload a credit bureau PDF", type=["pdf"])

if "state" not in st.session_state:
    st.session_state.state = {}

state = st.session_state.state

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.getvalue()


# -----------------------------
# Extraction + Parse
# -----------------------------
with st.spinner("Extracting text..."):
    text_primary = extract_text_pymupdf(pdf_bytes)
    mode = "Text"
    extracted_text = text_primary

    # If text extraction is too thin, OCR
    if len(text_primary.strip()) < 1200:
        with st.spinner("Low text detected. Running OCR fallback..."):
            extracted_text = extract_text_ocr(pdf_bytes)
            mode = "OCR"

with st.spinner("Parsing negative accounts..."):
    accounts, doc_bureaus, blocks, parse_notes, debug_warnings = parse_negative_accounts(extracted_text)
    confidence, conf_debug = compute_confidence(extracted_text, accounts)

# Store base parse in session for manual fix/edit stability
state["extraction_mode"] = mode
state["confidence"] = confidence
state["doc_bureaus"] = doc_bureaus
state["blocks"] = blocks
state["parse_notes"] = parse_notes
state["conf_debug"] = conf_debug
state["debug_warnings"] = debug_warnings
state["base_accounts"] = accounts

# Manual edit state init
if "manual_df" not in state:
    state["manual_df"] = build_manual_editor_df(accounts)

# Apply overrides
final_accounts = apply_manual_overrides(state["base_accounts"], state["manual_df"])
final_df = to_table_df(final_accounts)

# Metrics
def count_type(sub: str) -> int:
    sub = sub.lower()
    return sum(1 for a in final_accounts if sub in (a.negative_type_status or "").lower())

num_neg = len(final_accounts)
num_col = count_type("collection")
num_co = count_type("charge-off") + count_type("charge off")
num_late = sum(1 for a in final_accounts if "late" in (a.negative_type_status or "").lower() or "delinquent" in (a.negative_type_status or "").lower())
total_bal = sum((a.current_balance or 0.0) for a in final_accounts)
overall_pressure = overall_negative_pressure(final_accounts)

# -----------------------------
# Top Metric Cards
# -----------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("# Negative Accounts", num_neg)
c2.metric("# Collections", num_col)
c3.metric("# Charge-offs", num_co)
c4.metric("# Lates", num_late)
c5.metric("Total Negative Balance", format_money(total_bal))
c6.metric("Mode / Confidence", f"{mode} / {confidence}")

st.markdown(
    f"**Overall negative pressure:** {overall_pressure}  \n"
    f"**Detected bureau(s):** {', '.join(doc_bureaus) if doc_bureaus else 'Unknown'}"
)

with st.expander("How the impact ranges work (important)"):
    st.write(
        "Impact is shown as a **range** (never an exact score). It’s based on **negative type**, "
        "**how recently it reported**, and **balance size**. Real impact varies by scoring model "
        "(FICO vs Vantage), credit file thickness, utilization, and what else is on the report."
    )

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Negative Accounts", "Manual Fix", "Debug"])

with tab1:
    st.subheader("Negative Accounts")
    if final_df.empty:
        st.warning("No negative accounts detected with the current parsing rules.")
    else:
        st.dataframe(final_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Account Snippets (for troubleshooting)")
        for i, a in enumerate(final_accounts, start=1):
            title = f"{i}. {a.creditor_name or 'Unknown Creditor'} — {a.negative_type_status or 'Unknown Status'} — {a.masked_account_number}"
            with st.expander(title):
                st.caption("Raw block snippet (PII-masked):")
                st.code(a.raw_block_snippet or "", language="text")

    st.divider()
    colA, colB = st.columns([1, 1])
    with colA:
        pdf_bytes_out = build_export_pdf(final_df, mode, confidence, overall_pressure)
        st.download_button(
            "⬇️ Export PDF: Negative Accounts Summary.pdf",
            data=pdf_bytes_out,
            file_name="Negative Accounts Summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with colB:
        st.info("Export includes a disclaimer and the current table (after Manual Fix overrides).")

with tab2:
    st.subheader("Manual Fix (Live Overrides)")
    st.caption("Edit values below to override parsing results. Account numbers remain masked (last 4).")

    editable = st.data_editor(
        state["manual_df"],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "RowID": st.column_config.NumberColumn("RowID", disabled=True),
            "Balance (number)": st.column_config.NumberColumn("Balance (number)", format="%.2f", step=1.0),
            "Last Reported (YYYY-MM-DD)": st.column_config.TextColumn("Last Reported (YYYY-MM-DD)"),
            "Opened (YYYY-MM-DD)": st.column_config.TextColumn("Opened (YYYY-MM-DD)"),
        },
        key="manual_editor",
    )

    # Save edits instantly
    state["manual_df"] = editable

    st.divider()
    updated_accounts = apply_manual_overrides(state["base_accounts"], state["manual_df"])
    updated_df = to_table_df(updated_accounts)
    st.subheader("Preview: Updated Negative Accounts Table")
    st.dataframe(updated_df, use_container_width=True, hide_index=True)

    st.caption("Tip: Use YYYY-MM-DD for dates (example: 2024-11-15). MM/YYYY and MM/DD/YYYY also usually work.")

with tab3:
    st.subheader("Debug")
    if debug_warnings:
        for w in debug_warnings:
            st.warning(w)

    st.markdown("### Extraction Details")
    st.write({
        "Extraction Mode": mode,
        "Confidence": confidence,
        "Text Length": int(state["conf_debug"]["text_length"]),
        "Completion Rate": round(state["conf_debug"]["completion_rate"], 3),
        "Confidence Score": round(state["conf_debug"]["score"], 3),
        "Detected Bureaus": doc_bureaus if doc_bureaus else ["Unknown"],
        "Blocks Detected": len(state["blocks"]),
        "Negative Blocks Parsed": len(state["base_accounts"]),
    })

    st.markdown("### Field Extraction Notes (per parsed negative block)")
    note_rows = []
    for n in state["parse_notes"]:
        note_rows.append({
            "Block Index": n.block_index,
            "Notes": " | ".join(n.notes),
        })
    st.dataframe(pd.DataFrame(note_rows), use_container_width=True, hide_index=True)

    st.markdown("### Raw Extracted Text Preview (PII-masked best-effort)")
    preview = mask_pii_in_snippet((extracted_text or "")[:12000])
    st.code(preview, language="text")

    st.markdown("### Parsed Block List (first 30 blocks, PII-masked)")
    for i, blk in enumerate(state["blocks"][:30]):
        with st.expander(f"Block #{i}"):
            st.code(mask_pii_in_snippet(blk[:2000]), language="text")
