import io
import re
import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

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


# =============================
# App Config
# =============================
st.set_page_config(page_title="Birchwood Negative Accounts Extractor", page_icon="📄", layout="wide")
TODAY = date.today()

# =============================
# PII masking helpers
# =============================
ACCT_LIKE_RE = re.compile(r"(?<!\d)(?:X{2,}|\*{2,}|\d)[Xx\*\-\s\d]{5,}(?!\d)")
DIGIT_RUN_RE = re.compile(r"(?<!\d)(\d[\d\-\s]{6,}\d)(?!\d)")


def mask_account_number(raw: str) -> str:
    """
    Birchwood can use numeric account IDs in left column.
    We never expose full. We only show last 4 digits when digits exist.
    """
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return f"••••{digits[-4:]}"
    return "••••"


def mask_pii_in_snippet(text: str) -> str:
    if not text:
        return ""

    def repl(m):
        token = m.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 4:
            return f"••••{digits[-4:]}"
        return "••••"

    text = ACCT_LIKE_RE.sub(repl, text)
    text = DIGIT_RUN_RE.sub(repl, text)
    return text


# =============================
# Date + Money helpers
# =============================
def parse_mm_yy(s: str) -> Optional[date]:
    s = (s or "").strip()
    m = re.search(r"\b(0?[1-9]|1[0-2])[/\-]((?:\d{2})|(?:19|20)\d{2})\b", s)
    if not m:
        return None
    mm = int(m.group(1))
    yy = m.group(2)
    yyyy = int("20" + yy) if len(yy) == 2 else int(yy)
    try:
        return date(yyyy, mm, 1)
    except ValueError:
        return None


def format_date(d: Optional[date]) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def calc_age(opened: Optional[date]) -> str:
    if not opened:
        return ""
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
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return ""


# =============================
# Extraction pipeline
# =============================
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
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img)
        parts.append(txt or "")
    doc.close()
    return "\n".join(parts).strip()


# =============================
# Models
# =============================
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


# =============================
# Birchwood negative detection (status)
# =============================
def birchwood_negative_status(text: str) -> Tuple[bool, str]:
    t = (text or "").upper()

    if "CHARGE OFF" in t or "CHARGED OFF" in t:
        return True, "Charge-off"
    if re.search(r"\bCOLLECTION\b", t):
        return True, "Collection"

    m = re.search(r"\bDELINQ\s*(30|60|90|120)\b", t)
    if m:
        n = m.group(1)
        if n == "120":
            return True, "Late (120+)"
        return True, f"Late ({n})"

    # bucket counts sometimes appear (30-59 / 60-89 / 90+)
    # If any of these columns show a non-zero count, treat as a late.
    # (We keep it simple: 90+ > 60-89 > 30-59.)
    def count_bucket(label: str) -> int:
        m2 = re.search(rf"(?i){re.escape(label)}\s*[:\s]*\b(\d+)\b", text)
        return int(m2.group(1)) if m2 else 0

    c90 = count_bucket("90+")
    c60 = count_bucket("60-89")
    c30 = count_bucket("30-59")

    if c90 > 0:
        return True, "Late (90)"
    if c60 > 0:
        return True, "Late (60)"
    if c30 > 0:
        return True, "Late (30)"

    if "BANKRUPT" in t or "CHAPTER 7" in t or "CHAPTER 13" in t:
        return True, "Bankruptcy"
    if "FORECLOS" in t:
        return True, "Foreclosure"
    if "REPOS" in t:
        return True, "Repossession"
    if "JUDG" in t:
        return True, "Judgment"

    return False, ""


def detect_bureaus_from_text(text: str) -> str:
    t = (text or "").upper()
    found = []
    if "XP" in t or "EX" in t:
        found.append("Experian")
    if "TU" in t:
        found.append("TransUnion")
    if "EF" in t or "EQ" in t:
        found.append("Equifax")
    found = list(dict.fromkeys(found))
    if not found:
        return "Unknown"
    if len(found) == 1:
        return found[0]
    return "/".join(found)


# =============================
# Impact estimator (heuristics)
# =============================
def impact_range_for_account(neg_type: str, last_reported: Optional[date], balance: Optional[float]) -> Tuple[str, str]:
    t = (neg_type or "").lower()
    bal = balance or 0.0

    recency = "unknown"
    if last_reported:
        months = (TODAY.year - last_reported.year) * 12 + (TODAY.month - last_reported.month)
        if months <= 12:
            recency = "recent"
        elif months <= 24:
            recency = "mid"
        else:
            recency = "old"

    if bal >= 10000:
        bsev = "high"
    elif bal >= 2000:
        bsev = "med"
    else:
        bsev = "low"

    if "bankruptcy" in t:
        base, tier = (60, 180), "Severe"
    elif "foreclosure" in t:
        base, tier = (50, 160), "Severe"
    elif "judgment" in t:
        base, tier = (35, 110), "High"
    elif "repossession" in t:
        base, tier = (40, 120), "High"
    elif "charge-off" in t:
        base, tier = (25, 90), "High"
    elif "collection" in t:
        base, tier = (15, 70), "Moderate"
    elif "late (120" in t:
        base, tier = (20, 80), "High"
    elif "late (90" in t:
        base, tier = (18, 70), "Moderate"
    elif "late (60" in t:
        base, tier = (12, 45), "Moderate"
    elif "late (30" in t:
        base, tier = (5, 25), "Low"
    else:
        base, tier = (8, 35), "Low–Moderate"

    lo, hi = base

    if recency == "recent":
        lo, hi = int(lo * 1.15), int(hi * 1.20)
    elif recency == "old":
        lo, hi = int(lo * 0.70), int(hi * 0.75)
    else:
        lo, hi = int(lo * 0.90), int(hi * 0.90)

    if any(x in t for x in ["collection", "charge-off", "repossession", "judgment"]):
        if bsev == "high":
            lo, hi = int(lo * 1.15), int(hi * 1.20)
        elif bsev == "med":
            lo, hi = int(lo * 1.05), int(hi * 1.08)
        else:
            lo, hi = int(lo * 0.95), int(hi * 0.95)

    lo = max(0, lo)
    hi = max(lo + 5, hi)
    return tier, f"{lo}–{hi} pts"


def overall_negative_pressure(accounts: List[NegativeAccount]) -> str:
    if not accounts:
        return "None detected"
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
        else:
            w += 0.6

        bal = a.current_balance or 0.0
        if bal >= 10000:
            w += 0.5
        elif bal >= 2000:
            w += 0.2

        if a.last_reported_date:
            months = (TODAY.year - a.last_reported_date.year) * 12 + (TODAY.month - a.last_reported_date.month)
            if months <= 12:
                w += 0.4
            elif months <= 24:
                w += 0.2

    if w >= 12:
        return "Very High negative pressure"
    if w >= 7:
        return "High negative pressure"
    if w >= 3.5:
        return "Moderate negative pressure"
    return "Low negative pressure"


# =============================
# Confidence
# =============================
def compute_confidence(extracted_text: str, accounts: List[NegativeAccount]) -> Tuple[str, Dict[str, float]]:
    tlen = len((extracted_text or "").strip())
    n = max(1, len(accounts))

    def filled(v) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    key_fields = ["creditor_name", "masked_account_number", "current_balance", "negative_type_status", "last_reported_date"]
    completion = 0.0
    for k in key_fields:
        completion += sum(1 for a in accounts if filled(getattr(a, k))) / n
    completion /= len(key_fields)

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

    return label, {"text_length": float(tlen), "completion_rate": float(completion), "score": float(score)}


# =============================
# Birchwood TRADELINES (coordinate-aware parser)
# =============================
TRADELINES_MARKER = "TRADELINES"
TRADE_SUMMARY_MARKER = "TRADE SUMMARY"

# Column headers we expect in Birchwood
HEADER_TOKENS = [
    "Opened", "Reported", "Hi. Credit", "Credit Limit", "Reviewed",
    "30-59", "60-89", "90+", "Past Due", "Payment", "Balance"
]

# For filtering junk rows as "creditor"
CREDITOR_BAD_PREFIXES = (
    "Opened", "Reported", "Hi.", "Hi. Credit", "Credit", "Credit Limit",
    "Reviewed", "30-59", "60-89", "90+", "Past Due", "Payment", "Balance",
    "Source", "ECOA", "CUR", "WAS"
)


def _flatten_page_lines(page: fitz.Page) -> List[Dict]:
    """
    Returns list of line items: {text, x0,x1,y0,y1, cx, cy}
    """
    d = page.get_text("dict")
    out = []
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            # line text from spans
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln.get("bbox", [0, 0, 0, 0])
            out.append({
                "text": re.sub(r"\s+", " ", text).strip(),
                "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1),
                "cx": float((x0 + x1) / 2.0),
                "cy": float((y0 + y1) / 2.0),
            })
    out.sort(key=lambda r: (r["y0"], r["x0"]))
    return out


def _find_tradelines_y_range(lines: List[Dict]) -> Optional[Tuple[float, float]]:
    """
    Find approximate y-range of TRADELINES section on the page using text markers.
    """
    y_start = None
    y_end = None
    for r in lines:
        t = r["text"].upper()
        if TRADELINES_MARKER in t and y_start is None:
            y_start = r["y0"]
        if (TRADE_SUMMARY_MARKER in t or "TRADE SUMMARY" in t) and y_start is not None:
            y_end = r["y0"]
            break
    if y_start is None:
        return None
    if y_end is None:
        y_end = max(r["y1"] for r in lines) if lines else y_start + 2000
    # add buffers
    return (y_start + 5.0, y_end - 5.0)


def _detect_header_columns(lines: List[Dict], y0: float, y1: float) -> Dict[str, float]:
    """
    Find x-centers of column headers by looking for header tokens within the top part of TRADELINES.
    """
    header_band = [r for r in lines if y0 <= r["cy"] <= min(y1, y0 + 180)]
    cols = {}
    for token in HEADER_TOKENS:
        # match token loosely
        best = None
        for r in header_band:
            if token.lower() in r["text"].lower():
                # pick the widest match if multiple
                score = (r["x1"] - r["x0"])
                if best is None or score > best[0]:
                    best = (score, r["cx"])
        if best:
            cols[token] = float(best[1])
    return cols


def _looks_like_creditor_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # avoid header/cell words
    for p in CREDITOR_BAD_PREFIXES:
        if t.startswith(p):
            return False
    # creditor names in Birchwood are often ALL CAPS or Title-ish, and not just numbers/dates
    if re.fullmatch(r"[\d\W]+", t):
        return False
    if parse_mm_yy(t):
        return False
    return True


def _looks_like_account_id(text: str) -> bool:
    """
    Birchwood left column often has the account identifier directly under creditor.
    Usually digits 9+ (example in your screenshot).
    """
    s = (text or "").strip()
    if not s:
        return False
    if parse_mm_yy(s):
        return False
    if re.fullmatch(r"\d{7,}", s):
        return True
    return False


def _pick_near_x(lines: List[Dict], x_center: float, y_top: float, y_bot: float, tol: float = 60.0) -> List[str]:
    """
    Pick texts in y-range whose center x is within tol of x_center.
    """
    picks = []
    for r in lines:
        if y_top <= r["cy"] <= y_bot and abs(r["cx"] - x_center) <= tol:
            picks.append(r["text"])
    return picks


def parse_birchwood_tradelines_structured(pdf_bytes: bytes) -> Tuple[List[Dict], List[str]]:
    """
    Returns list of structured tradeline rows:
      {creditor, acct_raw, opened, reported, balance, bureaus, status_text, snippet}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rows: List[Dict] = []
    debug: List[str] = []
    total_candidates = 0

    for pno, page in enumerate(doc, start=1):
        lines = _flatten_page_lines(page)
        rng = _find_tradelines_y_range(lines)
        if not rng:
            continue
        y0, y1 = rng
        tl = [r for r in lines if y0 <= r["cy"] <= y1]

        if not tl:
            continue

        cols = _detect_header_columns(lines, y0, y1)
        # we mainly need these
        opened_x = cols.get("Opened")
        reported_x = cols.get("Reported")
        balance_x = cols.get("Balance")

        # define "left creditor column" cutoff using the Opened header if found
        left_cut = (opened_x - 80) if opened_x else 220.0

        left_lines = [r for r in tl if r["x0"] <= left_cut]
        left_lines.sort(key=lambda r: (r["y0"], r["x0"]))

        # Detect tradeline starts: creditor line followed shortly by account-id line (below)
        starts: List[Tuple[int, float]] = []  # (index in left_lines, start_y)
        for i in range(len(left_lines) - 1):
            a = left_lines[i]["text"]
            b = left_lines[i + 1]["text"]
            if _looks_like_creditor_line(a) and _looks_like_account_id(b):
                # ensure vertical closeness (same tradeline cell)
                if 0 < (left_lines[i + 1]["y0"] - left_lines[i]["y0"]) <= 45:
                    starts.append((i, left_lines[i]["y0"]))

        if not starts:
            continue

        # build y-ranges per tradeline from start to next start
        for si, (idx, start_y) in enumerate(starts):
            end_y = (starts[si + 1][1] - 5) if si + 1 < len(starts) else (y1 - 5)
            total_candidates += 1

            creditor = left_lines[idx]["text"].strip()
            acct_raw = left_lines[idx + 1]["text"].strip()

            # Pull opened/reported/balance by column proximity (within tradeline y-range)
            opened_dt = None
            reported_dt = None
            bal = None
            bureaus = "Unknown"

            # opened
            if opened_x:
                opened_vals = _pick_near_x(tl, opened_x, start_y, end_y, tol=70)
                # pick first MM/YY
                for v in opened_vals:
                    d = parse_mm_yy(v)
                    if d:
                        opened_dt = d
                        break

            # reported
            if reported_x:
                reported_vals = _pick_near_x(tl, reported_x, start_y, end_y, tol=70)
                for v in reported_vals:
                    d = parse_mm_yy(v)
                    if d:
                        reported_dt = d
                        break

            # balance
            if balance_x:
                bal_vals = _pick_near_x(tl, balance_x, start_y, end_y, tol=90)
                # pick first $ amount
                for v in bal_vals:
                    m = MONEY_RE.search(v)
                    if m:
                        bal = parse_money(m.group(0))
                        break

            # status text (anywhere in tradeline y-range)
            status_text_parts = []
            for r in tl:
                if start_y <= r["cy"] <= end_y:
                    status_text_parts.append(r["text"])
            status_text = " ".join(status_text_parts)
            # bureaus from source-like text in row
            bureaus = detect_bureaus_from_text(status_text)

            # snippet (masked)
            snippet = mask_pii_in_snippet(status_text[:1600])

            rows.append({
                "page": pno,
                "creditor": creditor,
                "acct_raw": acct_raw,
                "opened": opened_dt,
                "reported": reported_dt,
                "balance": bal,
                "bureaus": bureaus,
                "status_text": status_text,
                "snippet": snippet,
            })

    doc.close()
    debug.append(f"Structured TRADELINES: parsed {len(rows)} tradeline rows (candidates={total_candidates}).")
    if len(rows) == 0:
        debug.append("Structured TRADELINES: 0 rows found. If this was OCR-mode or a scanned PDF, table structure may not be recoverable.")
    return rows, debug


# =============================
# Parser orchestration (Birchwood-only)
# =============================
def parse_negative_accounts_birchwood(pdf_bytes: bytes, extracted_text: str, mode: str) -> Tuple[List[NegativeAccount], List[ParseNote], List[str]]:
    """
    Prefer structured parsing when mode==Text.
    Fallback to text-window parsing when OCR or structured fails.
    """
    debug_msgs: List[str] = []
    notes: List[ParseNote] = []
    accounts: List[NegativeAccount] = []

    # 1) Structured parsing for TEXT mode
    if mode == "Text":
        rows, dbg = parse_birchwood_tradelines_structured(pdf_bytes)
        debug_msgs.extend(dbg)

        for i, r in enumerate(rows):
            is_neg, status = birchwood_negative_status(r.get("status_text", ""))
            if not is_neg:
                continue

            creditor = r.get("creditor", "").strip()
            acct_raw = r.get("acct_raw", "").strip()

            opened_dt = r.get("opened")
            reported_dt = r.get("reported")
            bal = r.get("balance")
            bureaus = r.get("bureaus", "Unknown")

            acct = NegativeAccount(
                creditor_name=creditor,
                masked_account_number=mask_account_number(acct_raw),
                current_balance=bal,
                last_reported_date=reported_dt,
                date_opened=opened_dt,
                age_of_account=calc_age(opened_dt),
                negative_type_status=status,
                bureaus=bureaus,
                estimated_impact="",
                raw_block_snippet=r.get("snippet", ""),
            )
            tier, rng = impact_range_for_account(acct.negative_type_status, acct.last_reported_date, acct.current_balance)
            acct.estimated_impact = f"{tier}: {rng}"
            accounts.append(acct)
            notes.append(ParseNote(block_index=i, notes=[
                "Structured TRADELINES row parse",
                f"Page: {r.get('page')}",
                f"Creditor: {bool(creditor)}",
                f"Acct#: {bool(acct_raw)}",
                f"Opened: {bool(opened_dt)}",
                f"Reported: {bool(reported_dt)}",
                f"Balance: {bal is not None}",
                f"Bureaus: {bureaus}",
                f"Status: {status}",
            ]))

        # If we got good results, return them
        if len(accounts) > 0:
            return _dedupe_accounts(accounts), notes, debug_msgs

        debug_msgs.append("Structured TRADELINES produced 0 negatives; falling back to OCR/text-window logic.")

    # 2) Fallback (OCR or structured failed): old-style windows around negative keywords
    debug_msgs.append("Fallback parsing: using keyword-window blocks (less accurate for Birchwood tables).")
    blocks = build_keyword_windows(extracted_text)
    for i, blk in enumerate(blocks):
        is_neg, status = birchwood_negative_status(blk)
        if not is_neg:
            continue

        creditor, acct_raw = guess_creditor_and_acct_from_block(blk)
        opened_dt = guess_date_after_label(blk, "Opened")
        reported_dt = guess_date_after_label(blk, "Reported")
        bal = guess_money_after_label(blk, "Balance")
        bureaus = detect_bureaus_from_text(blk)

        acct = NegativeAccount(
            creditor_name=creditor,
            masked_account_number=mask_account_number(acct_raw),
            current_balance=bal,
            last_reported_date=reported_dt,
            date_opened=opened_dt,
            age_of_account=calc_age(opened_dt),
            negative_type_status=status,
            bureaus=bureaus,
            estimated_impact="",
            raw_block_snippet=mask_pii_in_snippet(blk[:1600]),
        )
        tier, rng = impact_range_for_account(acct.negative_type_status, acct.last_reported_date, acct.current_balance)
        acct.estimated_impact = f"{tier}: {rng}"
        accounts.append(acct)
        notes.append(ParseNote(block_index=i, notes=["Fallback block parse"]))

    return _dedupe_accounts(accounts), notes, debug_msgs


# =============================
# Fallback helpers (only used when OCR/structured fails)
# =============================
NEG_KEYWORDS = ["DELINQ", "CHARGE OFF", "CHARGED OFF", "COLLECTION", "BANKRUPT", "FORECLOS", "REPOS", "JUDG"]


def build_keyword_windows(text: str) -> List[str]:
    t = (text or "").replace("\r\n", "\n")
    lines = t.splitlines()
    hit_idxs = []
    for i, ln in enumerate(lines):
        u = ln.upper()
        if any(k in u for k in NEG_KEYWORDS):
            hit_idxs.append(i)

    windows = []
    for idx in hit_idxs:
        s = max(0, idx - 10)
        e = min(len(lines), idx + 16)
        windows.append((s, e))

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
    return blocks if blocks else [t.strip()]


def guess_creditor_and_acct_from_block(block: str) -> Tuple[str, str]:
    lines = [l.strip() for l in (block or "").splitlines() if l.strip()]
    creditor = ""
    acct = ""
    for i in range(min(8, len(lines))):
        if not creditor and lines[i] and not parse_mm_yy(lines[i]) and not re.fullmatch(r"[\d\W]+", lines[i]):
            creditor = lines[i][:90]
        if i + 1 < len(lines) and re.fullmatch(r"\d{7,}", lines[i + 1]):
            acct = lines[i + 1]
            break
    if not acct:
        m = ACCT_LIKE_RE.search(block or "")
        acct = m.group(0) if m else ""
    return creditor, acct


def guess_date_after_label(block: str, label: str) -> Optional[date]:
    m = re.search(rf"(?i)\b{re.escape(label)}\b\s*[:\-]?\s*([0-9]{{1,2}}[/\-][0-9]{{2,4}})", block)
    return parse_mm_yy(m.group(1)) if m else None


def guess_money_after_label(block: str, label: str) -> Optional[float]:
    m = re.search(rf"(?i)\b{re.escape(label)}\b.*?(\$?\s*[0-9,]+(?:\.[0-9]{{2}})?)", block, flags=re.DOTALL)
    return parse_money(m.group(1)) if m else None


def _dedupe_accounts(accounts: List[NegativeAccount]) -> List[NegativeAccount]:
    deduped = []
    seen = set()
    for a in accounts:
        key = (
            (a.creditor_name or "").lower().strip()[:50],
            a.masked_account_number,
            (a.negative_type_status or "").lower().strip(),
            int((a.current_balance or 0.0) // 10),
            format_date(a.last_reported_date),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    return deduped


# =============================
# Export PDF (ReportLab)
# =============================
def build_export_pdf(df: pd.DataFrame, extraction_mode: str, confidence: str, overall_pressure: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Negative Accounts Summary", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    meta = f"Extraction Mode: {extraction_mode} | Confidence: {confidence} | Overall Negative Pressure: {overall_pressure}"
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 0.20 * inch))

    disclaimer = (
        "Disclaimer: This report is an informational extraction from the uploaded Birchwood credit report PDF. "
        "Estimated impact ranges are heuristic (not an exact score change) and vary by scoring model, file thickness, "
        "utilization, and reporting details. Always verify against the source report."
    )
    story.append(Paragraph(disclaimer, styles["Italic"]))
    story.append(Spacer(1, 0.25 * inch))

    if df.empty:
        story.append(Paragraph("No negative accounts detected.", styles["Normal"]))
        doc.build(story)
        return buffer.getvalue()

    cols = [
        "Creditor", "Acct (Last 4)", "Balance", "Last Reported", "Opened", "Age",
        "Negative Type/Status", "Bureau(s)", "Estimated Impact"
    ]
    table_data = [cols]
    for _, r in df.iterrows():
        table_data.append([str(r.get(c, "")) for c in cols])

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


# =============================
# UI helpers
# =============================
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

    def parse_iso_or_mm(text: str) -> Optional[date]:
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return parse_mm_yy(text)

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

        last_dt = parse_iso_or_mm(last_s)
        open_dt = parse_iso_or_mm(open_s)

        bal = None
        try:
            if bal_num is not None and not (isinstance(bal_num, float) and math.isnan(bal_num)):
                bal = float(bal_num)
        except Exception:
            bal = None

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
            raw_block_snippet=a.raw_block_snippet,
        ))
    return updated


# =============================
# Streamlit UI
# =============================
st.title("📄 Birchwood Negative Accounts Extractor")
st.caption("Birchwood-only. Uses coordinate-aware table parsing in Text mode; OCR fallback available for scanned PDFs.")

uploaded = st.file_uploader("Upload a Birchwood credit report PDF", type=["pdf"])
if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.getvalue()

if "state" not in st.session_state:
    st.session_state.state = {}
state = st.session_state.state

with st.spinner("Extracting text..."):
    text_primary = extract_text_pymupdf(pdf_bytes)
    mode = "Text"
    extracted_text = text_primary
    if len(text_primary.strip()) < 1200:
        with st.spinner("Low text detected. Running OCR fallback..."):
            extracted_text = extract_text_ocr(pdf_bytes)
            mode = "OCR"

with st.spinner("Parsing negative accounts (Birchwood)..."):
    accounts, parse_notes, debug_msgs = parse_negative_accounts_birchwood(pdf_bytes, extracted_text, mode)
    confidence, conf_debug = compute_confidence(extracted_text, accounts)

state["extraction_mode"] = mode
state["confidence"] = confidence
state["base_accounts"] = accounts
state["parse_notes"] = parse_notes
state["conf_debug"] = conf_debug
state["debug_msgs"] = debug_msgs

# Manual editor state init
if "manual_df" not in state:
    state["manual_df"] = build_manual_editor_df(accounts)

final_accounts = apply_manual_overrides(state["base_accounts"], state["manual_df"])
final_df = to_table_df(final_accounts)

# Metrics
def count_type(substr: str) -> int:
    s = substr.lower()
    return sum(1 for a in final_accounts if s in (a.negative_type_status or "").lower())

num_neg = len(final_accounts)
num_col = count_type("collection")
num_co = count_type("charge-off") + count_type("charge off")
num_late = sum(1 for a in final_accounts if "late" in (a.negative_type_status or "").lower())
total_bal = sum((a.current_balance or 0.0) for a in final_accounts)
overall_pressure = overall_negative_pressure(final_accounts)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("# Negative Accounts", num_neg)
c2.metric("# Collections", num_col)
c3.metric("# Charge-offs", num_co)
c4.metric("# Lates", num_late)
c5.metric("Total Negative Balance", format_money(total_bal))
c6.metric("Mode / Confidence", f"{mode} / {confidence}")

tab1, tab2, tab3 = st.tabs(["Negative Accounts", "Manual Fix", "Debug"])

with tab1:
    st.subheader("Negative Accounts")
    if final_df.empty:
        st.warning("No negative accounts detected.")
    else:
        st.dataframe(final_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Account Snippets (for troubleshooting)")
        for i, a in enumerate(final_accounts, start=1):
            title = f"{i}. {a.creditor_name or 'Unknown Creditor'} — {a.negative_type_status or 'Unknown Status'} — {a.masked_account_number}"
            with st.expander(title):
                st.caption("Raw row snippet (PII-masked):")
                st.code(a.raw_block_snippet or "", language="text")

    st.divider()
    pdf_bytes_out = build_export_pdf(final_df, mode, confidence, overall_pressure)
    st.download_button(
        "⬇️ Export PDF: Negative Accounts Summary.pdf",
        data=pdf_bytes_out,
        file_name="Negative Accounts Summary.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with tab2:
    st.subheader("Manual Fix (Live Overrides)")
    st.caption("Edit values below to override parsing. Account numbers remain masked (last 4).")

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
    state["manual_df"] = editable

    st.divider()
    updated_accounts = apply_manual_overrides(state["base_accounts"], state["manual_df"])
    st.subheader("Preview: Updated Negative Accounts Table")
    st.dataframe(to_table_df(updated_accounts), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Debug")

    for msg in state.get("debug_msgs", []):
        st.info(msg)

    st.markdown("### Extraction Details")
    st.write({
        "Extraction Mode": mode,
        "Confidence": confidence,
        "Text Length": int(state["conf_debug"]["text_length"]),
        "Completion Rate": round(state["conf_debug"]["completion_rate"], 3),
        "Confidence Score": round(state["conf_debug"]["score"], 3),
        "Parsed Negative Accounts": len(state["base_accounts"]),
    })

    st.markdown("### Field Extraction Notes")
    note_rows = [{"Row/Block": n.block_index, "Notes": " | ".join(n.notes)} for n in state["parse_notes"]]
    st.dataframe(pd.DataFrame(note_rows), use_container_width=True, hide_index=True)

    st.markdown("### Raw Extracted Text Preview (PII-masked best-effort)")
    st.code(mask_pii_in_snippet((extracted_text or "")[:12000]), language="text")
