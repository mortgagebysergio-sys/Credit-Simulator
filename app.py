import io
import re
import math
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd

# Primary PDF engine
import fitz  # PyMuPDF

# Optional secondary extractor (kept per requirements)
import pdfplumber

# OCR
from PIL import Image
import pytesseract

# PDF export
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# ----------------------------
# App Config / Safety Defaults
# ----------------------------
APP_TITLE = "Mortgage Rapid Rescore / Credit Score Simulator"
PII_MASK = True  # Always mask account numbers in UI
MAX_PAGES_OCR = 25  # safety cap for OCR time on Community Cloud
OCR_RENDER_SCALE = 2.0  # higher => better OCR, slower

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Utilities
# ----------------------------

DATE_PATTERNS = [
    # 01/31/2026 or 1/3/26
    r"(?P<mdy>\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b)",
    # Jan 31 2026 / January 31, 2026
    r"(?P<mname>\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\.?\s+(0?[1-9]|[12]\d|3[01])[,]?\s+(\d{4})\b)",
]

MONEY_PATTERN = r"(?P<money>\$?\s*\(?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*\)?)"
ACCT_PATTERN = r"(?P<acct>(?:\*{2,}|\bx{2,}\b|#)?\s*[0-9xX\*]{4,})"

NEGATIVE_KEYWORDS = {
    "collection": ["collection", "collections", "coll"],
    "charge_off": ["charge off", "charged off", "charge-off", "c/o", "co"],
    "late": ["30 days", "60 days", "90 days", "120 days", "late", "delinquent"],
    "repossession": ["repo", "repossession"],
    "bankruptcy": ["bankruptcy", "chapter 7", "chapter 13"],
    "foreclosure": ["foreclosure"],
    "judgment": ["judgment"],
}

BUREAU_HINTS = {
    "Experian": ["experian"],
    "Equifax": ["equifax"],
    "TransUnion": ["transunion", "trans union"],
}

REVOLVING_HINTS = ["revolving", "credit card", "bankcard", "visa", "mastercard", "discover", "amex"]
INSTALLMENT_HINTS = ["installment", "auto", "student", "mortgage", "loan"]

DISCLAIMER = (
    "⚠️ **Disclaimer:** This simulator provides **estimates** only. Actual score changes vary by bureau, model "
    "(FICO/Vantage), file thickness, timing, and how creditors/bureaus report. Results are **not guaranteed**."
)

PRIVACY_NOTE = (
    "🔒 **Privacy:** PDFs are processed **in-memory** only. The app does **not** intentionally store your PDF. "
    "Account numbers are masked by default."
)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stable_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def mask_account_number(acct: str) -> str:
    if not acct:
        return ""
    digits = re.sub(r"[^0-9]", "", acct)
    if len(digits) >= 4:
        return f"****{digits[-4:]}"
    # fallback: keep last 4 chars of raw
    raw = acct.strip().replace(" ", "")
    return f"****{raw[-4:]}" if len(raw) >= 4 else "****"


def parse_money_to_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    # handle parentheses negative
    neg = "(" in s and ")" in s
    s = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    s = re.sub(r"[^\d\.]", "", s)
    if not s:
        return None
    try:
        v = float(s)
        return -v if neg else v
    except:
        return None


def safe_snip(text: str, limit: int = 1800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…(truncated)…"


def detect_bureaus(text: str) -> List[str]:
    t = text.lower()
    present = []
    for bureau, hints in BUREAU_HINTS.items():
        if any(h in t for h in hints):
            present.append(bureau)
    return present or ["Unknown"]


def classify_status(block: str) -> Tuple[str, List[str]]:
    """Return (primary_status, matched_statuses)."""
    b = block.lower()
    matched = []
    for k, kws in NEGATIVE_KEYWORDS.items():
        if any(kw in b for kw in kws):
            matched.append(k)

    # prioritize
    priority = ["bankruptcy", "foreclosure", "judgment", "repossession", "charge_off", "collection", "late"]
    for p in priority:
        if p in matched:
            return p, matched
    return "ok", matched


def guess_account_type(block: str) -> str:
    b = block.lower()
    if any(h in b for h in REVOLVING_HINTS):
        return "revolving"
    if any(h in b for h in INSTALLMENT_HINTS):
        return "installment"
    return "unknown"


def extract_dates(block: str) -> List[str]:
    out = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, block, flags=re.IGNORECASE):
            out.append(m.group(0))
    # de-dupe preserve order
    seen = set()
    uniq = []
    for d in out:
        if d.lower() not in seen:
            uniq.append(d)
            seen.add(d.lower())
    return uniq


def pick_last_reported_date(block: str) -> str:
    """
    Heuristic: if we see "reported" or "last reported", choose nearest date after that.
    Else pick the last date mentioned in block.
    """
    b = block
    dates = extract_dates(b)
    if not dates:
        return ""
    # anchor on "last reported"/"reported"
    anchor = re.search(r"(last\s+reported|reported\s+on|date\s+reported|as\s+of)\s*[:\-]?\s*(.*)$", b, flags=re.IGNORECASE)
    if anchor:
        tail = anchor.group(0)
        tail_dates = extract_dates(tail)
        if tail_dates:
            return tail_dates[0]
    return dates[-1]


def pick_balance(block: str) -> Optional[float]:
    """
    Heuristic: look for 'balance' then money; else largest plausible money value.
    """
    b = block
    # balance labeled
    m = re.search(r"(balance|current\s+balance|amt\s+due|amount\s+due)\s*[:\-]?\s*" + MONEY_PATTERN, b, flags=re.IGNORECASE)
    if m:
        return parse_money_to_float(m.group("money"))

    # else: collect all money-like numbers and pick max absolute
    vals = []
    for m2 in re.finditer(MONEY_PATTERN, b):
        v = parse_money_to_float(m2.group("money"))
        if v is None:
            continue
        # ignore tiny values that are likely not balances
        if abs(v) < 1:
            continue
        vals.append(v)
    if not vals:
        return None
    # choose max by abs
    return sorted(vals, key=lambda x: abs(x), reverse=True)[0]


def pick_creditor_name(block: str) -> str:
    """
    Heuristic: use first all-caps-ish line, or a line before 'account' token, etc.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Prefer a line that looks like a creditor heading: longer, mostly letters/spaces/&/.- and not starting with 'account'
    def score_line(ln: str) -> float:
        l = ln.strip()
        if len(l) < 4:
            return -10
        if re.search(r"\b(account|acct|opened|balance|reported|status)\b", l, flags=re.IGNORECASE):
            return -2
        # fewer digits is better
        digits = sum(ch.isdigit() for ch in l)
        letters = sum(ch.isalpha() for ch in l)
        if letters == 0:
            return -10
        ratio = letters / max(1, digits + letters)
        # all caps-ish bonus
        caps = sum(ch.isupper() for ch in l if ch.isalpha())
        caps_ratio = caps / max(1, letters)
        return (len(l) * 0.05) + (ratio * 2.0) + (caps_ratio * 0.5) - (digits * 0.2)

    ranked = sorted(lines[:10], key=score_line, reverse=True)
    best = ranked[0]

    # Clean common prefixes
    best = re.sub(r"^(creditor|furnisher)\s*[:\-]\s*", "", best, flags=re.IGNORECASE).strip()
    return best


def pick_account_number(block: str) -> str:
    """
    Heuristic: look for 'account' labeled; else find masked-like sequence with digits/x/*
    """
    b = block

    m = re.search(r"(account|acct|account\s*#|acct\s*#)\s*[:\-]?\s*" + ACCT_PATTERN, b, flags=re.IGNORECASE)
    if m:
        return m.group("acct").strip()

    # fallback: find sequences that include x/* or long digit runs
    candidates = []
    for m2 in re.finditer(ACCT_PATTERN, b, flags=re.IGNORECASE):
        s = m2.group("acct").strip()
        digits = len(re.sub(r"[^0-9]", "", s))
        if digits >= 4:
            candidates.append((digits, s))
    if not candidates:
        return ""
    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates[0][1]


def split_into_tradeline_blocks(text: str) -> List[str]:
    """
    Resilient-ish block splitting:
    - Try by common separators
    - Then fallback to chunking around repeated 'Account' headings
    """
    t = normalize_whitespace(text)

    # Common separators found in bureau PDFs
    seps = [
        r"\n\s*[-_]{5,}\s*\n",               # ----- lines
        r"\n\s*={5,}\s*\n",                 # ===== lines
        r"\n\s*\*\s*\*\s*\*\s*\*\s*\*\s*\n" # ***** pattern
    ]

    blocks = [t]
    for sep in seps:
        if re.search(sep, t):
            blocks = re.split(sep, t)
            break

    # If still one big block, split on repeated "Account" or "Creditor"
    if len(blocks) <= 2:
        blocks = re.split(r"\n(?=(?:account|acct|creditor|furnisher)\b)", t, flags=re.IGNORECASE)

    # Clean / filter tiny blocks
    cleaned = []
    for b in blocks:
        b = b.strip()
        if len(b) < 80:
            continue
        cleaned.append(b)

    # If we got too few, do a fallback chunking
    if len(cleaned) < 4 and len(t) > 4000:
        # chunk by paragraphs
        paras = [p.strip() for p in t.split("\n\n") if p.strip()]
        cleaned = [p for p in paras if len(p) >= 120]

    return cleaned


# ----------------------------
# Data Model
# ----------------------------
@dataclass
class Tradeline:
    creditor_name: str = ""
    account_number: str = ""  # raw-ish, masked in UI
    balance: Optional[float] = None
    last_reported: str = ""
    opened_date: str = ""
    status: str = "ok"  # ok / collection / charge_off / late / etc
    bureaus: str = "Unknown"
    account_type: str = "unknown"  # revolving/installment/unknown
    notes: str = ""  # debugging / extra


def tradelines_to_df(lines: List[Tradeline]) -> pd.DataFrame:
    rows = []
    for tl in lines:
        d = asdict(tl)
        # display-friendly fields
        d["balance"] = "" if tl.balance is None else float(tl.balance)
        d["account_last4"] = mask_account_number(tl.account_number)
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "creditor_name", "account_number", "account_last4", "balance",
            "last_reported", "opened_date", "status", "bureaus", "account_type", "notes"
        ])
    return df


(working_df, list):     working_df = pd.DataFrame(working_df)(obj) -> List[Tradeline]:
    """
    Accepts either:
      - pandas DataFrame (preferred)
      - list[dict] or list[Tradeline] (fallback)
    Returns list[Tradeline].
    """
    # If already a list of Tradeline objects, return as-is
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], Tradeline):
        return obj

    # If list of dicts, convert to DataFrame
    if isinstance(obj, list):
        try:
            obj = pd.DataFrame(obj)
        except Exception:
            return []

    # If it's not a DataFrame by now, bail out safely
    if not isinstance(obj, pd.DataFrame):
        return []

    df = obj

    out: List[Tradeline] = []
    for _, r in df.iterrows():
        creditor = str(r.get("creditor", "") or "").strip()
        acct = str(r.get("account_last4", "") or "").strip()
        bureau = str(r.get("bureau", "") or "").strip()

        # balance
        bal = r.get("balance", None)
        bal_f = None
        try:
            if bal is not None and not pd.isna(bal) and str(bal).strip() != "":
                bal_f = float(str(bal).replace("$", "").replace(",", "").strip())
        except Exception:
            bal_f = None

        last_rep = str(r.get("last_reported", "") or "").strip()
        opened = str(r.get("opened", "") or "").strip()
        status = str(r.get("status", "") or "").strip()
        notes = str(r.get("notes", "") or "").strip()

        out.append(
            Tradeline(
                creditor=creditor,
                account_last4=acct,
                bureau=bureau,
                balance=bal_f,
                last_reported=last_rep,
                opened=opened,
                status=status,
                notes=notes,
            )
        )
    return out
        bal = r.get("balance", None)
        if pd.isna(bal) or bal == "":
            bal_f = None
        else:
            try:
                bal_f = float(bal)
            except:
                bal_f = None

        out.append(Tradeline(
            creditor_name=str(r.get("creditor_name", "")).strip(),
            account_number=str(r.get("account_number", "")).strip(),
            balance=bal_f,
            last_reported=str(r.get("last_reported", "")).strip(),
            opened_date=str(r.get("opened_date", "")).strip(),
            status=str(r.get("status", "ok")).strip(),
            bureaus=str(r.get("bureaus", "Unknown")).strip(),
            account_type=str(r.get("account_type", "unknown")).strip(),
            notes=str(r.get("notes", "")).strip(),
        ))
    return out


# ----------------------------
# PDF Extraction (Text -> OCR fallback)
# ----------------------------
def extract_text_pymupdf(pdf_bytes: bytes) -> Tuple[str, Dict]:
    meta = {"engine": "PyMuPDF", "pages": 0, "mode": "Text", "ms": 0}
    t0 = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta["pages"] = doc.page_count

    parts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        parts.append(page.get_text("text") or "")
    text = "\n".join(parts)
    meta["ms"] = int((time.time() - t0) * 1000)
    return normalize_whitespace(text), meta


def extract_text_pdfplumber(pdf_bytes: bytes) -> Tuple[str, Dict]:
    meta = {"engine": "pdfplumber", "pages": 0, "mode": "Text", "ms": 0}
    t0 = time.time()
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        meta["pages"] = len(pdf.pages)
        for p in pdf.pages:
            parts.append(p.extract_text() or "")
    meta["ms"] = int((time.time() - t0) * 1000)
    return normalize_whitespace("\n".join(parts)), meta


def ocr_pdf_pymupdf(pdf_bytes: bytes, max_pages: int = MAX_PAGES_OCR) -> Tuple[str, Dict]:
    meta = {"engine": "PyMuPDF+Tesseract", "pages": 0, "mode": "OCR", "ms": 0, "pages_ocr": 0}
    t0 = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta["pages"] = doc.page_count
    pages_to_ocr = min(doc.page_count, max_pages)
    meta["pages_ocr"] = pages_to_ocr

    parts = []
    matrix = fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE)
    for i in range(pages_to_ocr):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img)
        parts.append(txt or "")

    meta["ms"] = int((time.time() - t0) * 1000)
    return normalize_whitespace("\n".join(parts)), meta


def extraction_quality(text: str, parsed_fields_completion: float) -> Tuple[str, str]:
    """
    Return (confidence: High/Medium/Low, hint string)
    """
    L = len(text)
    if L > 9000 and parsed_fields_completion >= 0.70:
        return "High", "Strong text volume + good field completion."
    if L > 3500 and parsed_fields_completion >= 0.45:
        return "Medium", "Decent extraction; some fields may need Manual Fix."
    return "Low", "Sparse text or low field completion. Expect Manual Fix and/or OCR."


def should_fallback_to_ocr(text: str) -> bool:
    """
    Decide if text extraction is insufficient.
    """
    if len(text) < 1200:
        return True
    # If very few digits exist, likely scanned or badly extracted
    digits = sum(ch.isdigit() for ch in text)
    if digits < 120:
        return True
    return False


# ----------------------------
# Parsing
# ----------------------------
def parse_tradelines(text: str, bureaus: List[str]) -> Tuple[List[Tradeline], Dict]:
    blocks = split_into_tradeline_blocks(text)
    parsed: List[Tradeline] = []

    for b in blocks:
        status, matched_statuses = classify_status(b)
        acct = pick_account_number(b)
        creditor = pick_creditor_name(b)
        bal = pick_balance(b)
        last_rep = pick_last_reported_date(b)

        # opened date heuristic: look for "opened" label
        opened = ""
        mo = re.search(r"(opened|date\s+opened)\s*[:\-]?\s*(.*)$", b, flags=re.IGNORECASE)
        if mo:
            opened_candidates = extract_dates(mo.group(0))
            if opened_candidates:
                opened = opened_candidates[0]

        acc_type = guess_account_type(b)

        tl = Tradeline(
            creditor_name=creditor,
            account_number=acct,
            balance=bal,
            last_reported=last_rep,
            opened_date=opened,
            status=status,
            bureaus=", ".join(bureaus) if bureaus else "Unknown",
            account_type=acc_type,
            notes=";".join(matched_statuses) if matched_statuses else "",
        )

        # Filter obvious non-tradeline noise: require creditor name + (account or balance or date)
        signal = 0
        if tl.creditor_name and len(tl.creditor_name) >= 4:
            signal += 1
        if tl.account_number:
            signal += 1
        if tl.balance is not None:
            signal += 1
        if tl.last_reported:
            signal += 1

        if signal >= 2:
            parsed.append(tl)

    # de-dupe by (creditor + last4 + balance-ish)
    uniq = []
    seen = set()
    for tl in parsed:
        k = (
            tl.creditor_name.lower().strip(),
            mask_account_number(tl.account_number),
            "" if tl.balance is None else round(float(tl.balance), 0),
            tl.status
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(tl)

    debug = {
        "blocks_count": len(blocks),
        "parsed_count": len(uniq),
        "raw_blocks_preview": blocks[:8],
    }
    return uniq, debug


def compute_field_completion(lines: List[Tradeline]) -> float:
    """
    Completion over core fields for confidence hint.
    """
    if not lines:
        return 0.0
    fields = ["creditor_name", "account_number", "balance", "last_reported", "status"]
    total = len(lines) * len(fields)
    filled = 0
    for tl in lines:
        if tl.creditor_name:
            filled += 1
        if tl.account_number:
            filled += 1
        if tl.balance is not None:
            filled += 1
        if tl.last_reported:
            filled += 1
        if tl.status and tl.status != "ok":
            filled += 1  # negative status is useful signal
        else:
            # count ok status too, but less useful; still count as filled
            filled += 0.5
    return min(1.0, filled / total)


# ----------------------------
# Recommendations Engine
# ----------------------------
def recommend_actions(tl: Tradeline) -> List[Dict]:
    """
    Returns ranked actions with reason + impact range.
    Range is intentionally conservative and non-additive in combined scenarios.
    """
    actions = []
    bal = 0.0 if tl.balance is None else float(tl.balance)

    if tl.status == "collection":
        actions.append({
            "action": "Pay-for-Delete (PFD) negotiation",
            "why": "Collections can heavily suppress scores. A deletion is often more beneficial than a paid update.",
            "impact": (15, 60),
            "effort": "Medium",
        })
        actions.append({
            "action": "Dispute inaccuracies (if any) with bureaus",
            "why": "If dates/amounts/ownership are wrong, removal can be possible. Use only truthful, supportable claims.",
            "impact": (10, 50),
            "effort": "Medium",
        })
        actions.append({
            "action": "Settle / Pay to $0 balance (if deletion not possible)",
            "why": "Even without deletion, paid collections can help in some models and helps underwriting optics.",
            "impact": (0, 25),
            "effort": "Low",
        })

    elif tl.status == "charge_off":
        actions.append({
            "action": "Settle and obtain 'paid/settled' letter (and request update)",
            "why": "Charge-offs can remain, but updating balance to $0 improves underwriting and may modestly help score.",
            "impact": (0, 25),
            "effort": "Medium",
        })
        actions.append({
            "action": "Dispute factual inaccuracies (carefully)",
            "why": "If reporting is inaccurate, removal or correction can improve score significantly.",
            "impact": (10, 50),
            "effort": "Medium",
        })
        actions.append({
            "action": "Goodwill request (if lates embedded and you have strong history)",
            "why": "If there were lates but you have otherwise good history, goodwill adjustments are sometimes granted.",
            "impact": (0, 20),
            "effort": "Low",
        })

    elif tl.status == "late":
        actions.append({
            "action": "Goodwill adjustment request",
            "why": "If the late is isolated and you’re now current, some lenders will remove late notations.",
            "impact": (5, 35),
            "effort": "Low",
        })
        actions.append({
            "action": "Bring current / set autopay + document hardship if applicable",
            "why": "Stopping future lates is the fastest path to stabilizing scores.",
            "impact": (0, 15),
            "effort": "Low",
        })
        actions.append({
            "action": "Dispute inaccuracies (only if truly inaccurate)",
            "why": "Incorrect late reporting can be corrected/removed.",
            "impact": (10, 40),
            "effort": "Medium",
        })

    elif tl.status in ["bankruptcy", "foreclosure", "judgment", "repossession"]:
        actions.append({
            "action": "Dispute inaccuracies / verify dates and public record matching",
            "why": "High-impact derogatories are hard to remove, but inaccuracies can be corrected.",
            "impact": (10, 40),
            "effort": "Medium",
        })
        actions.append({
            "action": "Focus on utilization + positive tradelines",
            "why": "With major derogatories, score gains are typically driven by revolving utilization and adding positives.",
            "impact": (10, 50),
            "effort": "Low",
        })

    else:
        # Not negative by keyword, but may still be a leverage point (e.g., utilization)
        if tl.account_type == "revolving" and bal > 0:
            actions.append({
                "action": "Utilization paydown strategy (target 30% → 10% → 1–9%)",
                "why": "Revolving utilization is one of the fastest-moving score factors.",
                "impact": (10, 70),
                "effort": "Low",
            })

    # Sort by "upper impact" then lower effort
    effort_rank = {"Low": 0, "Medium": 1, "High": 2}
    actions.sort(key=lambda a: (a["impact"][1], -effort_rank.get(a["effort"], 1)), reverse=True)
    return actions[:5]


def summarize_negatives(lines: List[Tradeline]) -> Dict[str, int]:
    c = {"collections": 0, "charge_offs": 0, "lates": 0}
    for tl in lines:
        if tl.status == "collection":
            c["collections"] += 1
        elif tl.status == "charge_off":
            c["charge_offs"] += 1
        elif tl.status == "late":
            c["lates"] += 1
    return c


# ----------------------------
# Score Estimator (Range-based, non-additive)
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def baseline_score_range(lines: List[Tradeline], user_known_score: Optional[int]) -> Tuple[int, int, Dict]:
    """
    If user provides a known score, we center around it with a band.
    Otherwise infer a conservative range from negative counts.
    """
    neg = summarize_negatives(lines)
    major = sum(1 for tl in lines if tl.status in ["bankruptcy", "foreclosure", "judgment", "repossession"])
    total_negs = neg["collections"] + neg["charge_offs"] + neg["lates"] + major

    info = {"method": "", "negatives": neg, "major_derogs": major}

    if user_known_score:
        lo = clamp(user_known_score - 25, 300, 850)
        hi = clamp(user_known_score + 25, 300, 850)
        info["method"] = "User provided"
        return int(lo), int(hi), info

    # Inference heuristic (intentionally wide)
    # Start mid
    center = 690
    center -= neg["collections"] * 35
    center -= neg["charge_offs"] * 30
    center -= neg["lates"] * 18
    center -= major * 45

    # Wider range when more uncertainty
    width = 60 + total_negs * 10
    lo = clamp(center - width, 300, 850)
    hi = clamp(center + width, 300, 850)
    info["method"] = "Inferred"
    return int(lo), int(hi), info


def combined_scenario_adjustment(
    current_lo: int,
    current_hi: int,
    toggles: Dict,
) -> Tuple[int, int, Dict]:
    """
    Apply scenario adjustments with diminishing returns and interactions.
    Not additive: we compress gains as more toggles are active.
    """
    # Base potential gains per toggle
    potentials = []

    # Utilization: assume meaningful if user changes slider
    util_target = toggles.get("util_target", 30)
    if util_target <= 9:
        potentials.append(("utilization_1_9", 20, 80))
    elif util_target <= 10:
        potentials.append(("utilization_10", 15, 60))
    elif util_target <= 30:
        potentials.append(("utilization_30", 5, 35))

    if toggles.get("remove_collections", False):
        potentials.append(("remove_collections", 15, 70))
    if toggles.get("remove_chargeoffs", False):
        potentials.append(("remove_chargeoffs", 5, 45))
    if toggles.get("remove_lates", False):
        potentials.append(("remove_lates", 10, 55))
    if toggles.get("add_authorized_user", False):
        potentials.append(("authorized_user", 5, 35))
    if toggles.get("add_new_revolver", False):
        potentials.append(("new_revolver", 3, 25))
    months = int(toggles.get("months_pass", 0))
    if months > 0:
        # time helps modestly, more if no new negatives; keep conservative
        potentials.append(("time", 0, min(30, 2 * months)))

    # Diminishing returns: if many actions, compress sum by factor
    n = len(potentials)
    if n == 0:
        return current_lo, current_hi, {"potentials": [], "compression": 1.0}

    # Sum lower/upper then compress
    sum_lo = sum(p[1] for p in potentials)
    sum_hi = sum(p[2] for p in potentials)

    # Compression factor: more actions => less additivity
    # 1 action: 1.0, 2: 0.85, 3: 0.75, 4: 0.68, 5+: 0.62
    compression = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.68}.get(n, 0.62)

    gain_lo = int(round(sum_lo * compression))
    gain_hi = int(round(sum_hi * compression))

    new_lo = clamp(current_lo + gain_lo, 300, 850)
    new_hi = clamp(current_hi + gain_hi, 300, 850)

    # Also widen uncertainty a bit when OCR/low quality? handled elsewhere via confidence hint
    return int(new_lo), int(new_hi), {"potentials": potentials, "compression": compression}


# ----------------------------
# Utilization Helper
# ----------------------------
def utilization_paydown_helper(total_revolving_balance: float, total_revolving_limit: float) -> Dict[str, float]:
    """
    Returns paydown needed to reach utilization targets.
    If limit is missing/zero, caller should handle.
    """
    out = {}
    for target in [30, 10, 9]:
        target_bal = (target / 100.0) * total_revolving_limit
        paydown = max(0.0, total_revolving_balance - target_bal)
        out[f"to_{target}%"] = round(paydown, 2)
    # 1–9% band: suggest 5% midpoint
    target = 5
    target_bal = (target / 100.0) * total_revolving_limit
    out["to_1_9%_mid(5%)"] = round(max(0.0, total_revolving_balance - target_bal), 2)
    return out


# ----------------------------
# Checklist PDF Export
# ----------------------------
def build_checklist_pdf_bytes(
    score_lo: int,
    score_hi: int,
    goal_score: int,
    top_actions: List[str],
    negative_lines: List[Tradeline],
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    w, h = LETTER

    def draw_header(title: str):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.75 * inch, h - 0.9 * inch, title)
        c.setFont("Helvetica", 9)
        c.drawRightString(w - 0.75 * inch, h - 0.85 * inch, f"Generated: {now_str()}")

    def draw_wrapped(text: str, x: float, y: float, max_width: float, line_height: float = 12, font="Helvetica", size=10):
        c.setFont(font, size)
        words = text.split()
        line = ""
        yy = y
        for word in words:
            test = (line + " " + word).strip()
            if c.stringWidth(test, font, size) <= max_width:
                line = test
            else:
                c.drawString(x, yy, line)
                yy -= line_height
                line = word
        if line:
            c.drawString(x, yy, line)
            yy -= line_height
        return yy

    # --- Summary page
    draw_header("Rapid Rescore Checklist")
    y = h - 1.35 * inch

    c.setFont("Helvetica", 11)
    c.drawString(0.75 * inch, y, f"Estimated current score range: {score_lo}–{score_hi}")
    y -= 16
    c.drawString(0.75 * inch, y, f"Selected goal score: {goal_score}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Top recommended actions (overall):")
    y -= 14
    c.setFont("Helvetica", 10)
    for i, a in enumerate(top_actions[:5], start=1):
        y = draw_wrapped(f"{i}. {a}", 0.9 * inch, y, max_width=w - 1.6 * inch, line_height=12, size=10)
        y -= 2

    y -= 8
    c.setFont("Helvetica-Oblique", 9)
    y = draw_wrapped(
        "Disclaimer: Score outcomes are estimates and not guaranteed. Actual results vary by bureau, model, "
        "timing, reporting practices, and credit file characteristics.",
        0.75 * inch, y, max_width=w - 1.5 * inch, line_height=11, size=9
    )

    c.showPage()

    # --- Detail pages per negative account
    for tl in negative_lines:
        draw_header("Account Checklist")
        y = h - 1.35 * inch

        acct_mask = mask_account_number(tl.account_number)
        bal_str = "" if tl.balance is None else f"${tl.balance:,.2f}"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.75 * inch, y, f"{tl.creditor_name} ({acct_mask})")
        y -= 16

        c.setFont("Helvetica", 10)
        c.drawString(0.75 * inch, y, f"Status: {tl.status}")
        y -= 14
        c.drawString(0.75 * inch, y, f"Current balance: {bal_str}")
        y -= 14
        c.drawString(0.75 * inch, y, f"Last reported date: {tl.last_reported or '—'}")
        y -= 14
        c.drawString(0.75 * inch, y, f"Bureau(s): {tl.bureaus}")
        y -= 18

        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.75 * inch, y, "Recommended action(s):")
        y -= 14

        recs = recommend_actions(tl)
        c.setFont("Helvetica", 10)
        if recs:
            for r in recs[:3]:
                y = draw_wrapped(f"• {r['action']} — Why: {r['why']}", 0.9 * inch, y, w - 1.6 * inch, 12, "Helvetica", 10)
                y -= 2
        else:
            y = draw_wrapped("• Review account for potential disputes or paydown opportunities.", 0.9 * inch, y, w - 1.6 * inch, 12, "Helvetica", 10)

        y -= 10
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.75 * inch, y, "Documents to collect:")
        y -= 14
        docs = [
            "Proof of payment / receipt",
            "Settlement letter (if settling)",
            "Updated $0 balance letter (if paid)",
            "Most recent account statement",
            "Identity verification (as required by bureau)",
            "Dispute notes (what’s inaccurate, dates, amounts, ownership)",
        ]
        c.setFont("Helvetica", 10)
        for d in docs:
            c.drawString(0.9 * inch, y, f"☐ {d}")
            y -= 12

        y -= 10
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.75 * inch, y, "Notes / Follow-up:")
        y -= 10
        c.setFont("Helvetica", 10)
        # blank lines
        for _ in range(10):
            c.line(0.75 * inch, y, w - 0.75 * inch, y)
            y -= 14

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.read()


# ----------------------------
# Session State
# ----------------------------
def init_state():
    st.session_state.setdefault("pdf_hash", "")
    st.session_state.setdefault("raw_text", "")
    st.session_state.setdefault("extraction_meta", {})
    st.session_state.setdefault("bureaus", ["Unknown"])
    st.session_state.setdefault("parse_debug", {})
    st.session_state.setdefault("tradelines_df", pd.DataFrame())
    st.session_state.setdefault("edited_df", None)
    st.session_state.setdefault("quality_confidence", "Low")
    st.session_state.setdefault("quality_hint", "")
    st.session_state.setdefault("user_score", None)


init_state()


# ----------------------------
# UI - Header
# ----------------------------
st.title("📈 Mortgage Rapid Rescore / Credit Score Simulator")
st.caption("Comparable UX concepts to Credit Karma / CreditWise / NerdWallet — with a rapid-rescore workflow, OCR fallback, manual fixes, and checklist export.")
st.markdown(DISCLAIMER)
st.markdown(PRIVACY_NOTE)
st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Upload & Settings")

    uploaded = st.file_uploader("Upload a credit bureau PDF", type=["pdf"])

    # Optional Known Score Toggle (FIXES STREAMLIT ERROR)
    use_known_score = st.checkbox(
        "I know the current credit score (optional)",
        value=False
    )

    if use_known_score:
        user_score_val = st.number_input(
            "Current credit score",
            min_value=300,
            max_value=850,
            value=680,
            step=1,
        )
        user_score_val = int(user_score_val)
    else:
        user_score_val = None

    goal_score = st.selectbox(
        "Goal score",
        [620, 660, 680, 700],
        index=3
    )

    st.subheader("OCR Safety")
    st.write(f"Max OCR pages: **{MAX_PAGES_OCR}**")
    st.caption("Large PDFs can be slow on Streamlit Cloud. OCR pages are capped for reliability.")

    st.subheader("OCR Safety")
    st.write(f"Max OCR pages: **{MAX_PAGES_OCR}**")
    st.caption("Large PDFs can be slow on Streamlit Cloud. If your report is long, OCR will cap pages for reliability.")

    run_btn = st.button(
    "Process PDF",
    type="primary",
    use_container_width=True,
    key="process_pdf_button"
)

# ----------------------------
# ----------------------------
if run_btn:
    if not uploaded:
        st.error("Please upload a PDF first.")
    else:
        pdf_bytes = uploaded.read()
        pdf_hash = stable_hash_bytes(pdf_bytes)

        st.session_state["pdf_hash"] = pdf_hash

        # 1) Primary: PyMuPDF text
        text1, meta1 = extract_text_pymupdf(pdf_bytes)

        # 2) Optional secondary: pdfplumber if PyMuPDF weak (sometimes helps)
        # We'll only use it if PyMuPDF is short.
        text2, meta2 = ("", {})
        if len(text1) < 2000:
            text2, meta2 = extract_text_pdfplumber(pdf_bytes)

        # Choose best text candidate pre-OCR
        chosen_text = text1
        chosen_meta = meta1
        if len(text2) > len(text1) * 1.15:
            chosen_text = text2
            chosen_meta = meta2

        # Determine if OCR fallback needed
        if should_fallback_to_ocr(chosen_text):
            ocr_text, ocr_meta = ocr_pdf_pymupdf(pdf_bytes, max_pages=MAX_PAGES_OCR)
            # If OCR yields more usable text, use it
            if len(ocr_text) > len(chosen_text) * 1.05:
                chosen_text = ocr_text
                chosen_meta = ocr_meta

        st.session_state["raw_text"] = chosen_text
        st.session_state["extraction_meta"] = chosen_meta
        st.session_state["bureaus"] = detect_bureaus(chosen_text)

        # Parse
        tradelines, dbg = parse_tradelines(chosen_text, st.session_state["bureaus"])
        st.session_state["parse_debug"] = dbg

        df = tradelines_to_df(tradelines)

        # Add columns for manual fixes / simulator flexibility
        if "credit_limit" not in df.columns:
            df["credit_limit"] = ""  # optional for utilization helper
        if "is_negative" not in df.columns:
            df["is_negative"] = df["status"].apply(lambda s: str(s).strip().lower() != "ok")

        st.session_state["tradelines_df"] = df
        st.session_state["edited_df"] = None

        # Quality confidence
        completion = compute_field_completion(tradelines)
        conf, hint = extraction_quality(chosen_text, completion)
        st.session_state["quality_confidence"] = conf
        st.session_state["quality_hint"] = hint

        st.success(f"Processed PDF ({pdf_hash}). Extraction mode: **{chosen_meta.get('mode','Text')}** via **{chosen_meta.get('engine','')}**.")

# ----------------------------
# Load current working DF (edited if present)
# ----------------------------
base_df = st.session_state["tradelines_df"].copy()
working_df = st.session_state["edited_df"] if st.session_state["edited_df"] is not None else base_df

# If no data yet, show friendly empty state
if working_df is None or working_df.empty:
    st.stop()

# Always mask account display for UI
display_df = working_df.copy()
display_df["account_last4"] = display_df["account_number"].apply(lambda s: mask_account_number(str(s)))
if PII_MASK:
    # do not display full account numbers in main tables
    display_df["account_number"] = display_df["account_last4"]

# ----------------------------
# Top Metrics
# ----------------------------
neg_counts = summarize_negatives(if isinstance(working_df, list):     working_df = pd.DataFrame(working_df)(working_df.fillna("").to_dict(orient="records") if isinstance(working_df, pd.DataFrame) else []))
# Better: compute from df directly
collections_ct = int((working_df["status"] == "collection").sum()) if "status" in working_df.columns else 0
chargeoffs_ct = int((working_df["status"] == "charge_off").sum()) if "status" in working_df.columns else 0
lates_ct = int((working_df["status"] == "late").sum()) if "status" in working_df.columns else 0

score_lo, score_hi, score_info = baseline_score_range(if isinstance(working_df, list):     working_df = pd.DataFrame(working_df)(working_df.assign(balance=working_df.get("balance", "")).fillna("").to_dict(orient="records")), user_score_val)

# Utilization (optional)
def coerce_float(x):
    try:
        if x == "" or pd.isna(x):
            return None
        return float(x)
    except:
        return None

rev_mask = working_df["account_type"].astype(str).str.lower().eq("revolving") if "account_type" in working_df.columns else pd.Series([False] * len(working_df))
rev_balances = []
rev_limits = []
for _, r in working_df[rev_mask].iterrows():
    b = coerce_float(r.get("balance", None))
    l = coerce_float(r.get("credit_limit", None))
    if b is not None:
        rev_balances.append(max(0.0, b))
    if l is not None:
        rev_limits.append(max(0.0, l))

total_rev_bal = float(sum(rev_balances)) if rev_balances else 0.0
total_rev_lim = float(sum(rev_limits)) if rev_limits else 0.0
util_pct = (total_rev_bal / total_rev_lim * 100.0) if total_rev_lim > 0 else None

# Quality indicator
mode = st.session_state["extraction_meta"].get("mode", "Text")
engine = st.session_state["extraction_meta"].get("engine", "")
conf = st.session_state.get("quality_confidence", "Low")
hint = st.session_state.get("quality_hint", "")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Estimated Score Range", f"{score_lo}–{score_hi}", f"Method: {score_info.get('method','')}")
m2.metric("Collections", f"{collections_ct}")
m3.metric("Charge-offs", f"{chargeoffs_ct}")
m4.metric("Lates", f"{lates_ct}")
m5.metric("Extraction Mode", f"{mode}", f"{engine}")
m6.metric("Report Quality", f"{conf}", hint)

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab_sim, tab_recs, tab_parsed, tab_debug = st.tabs(
    ["What-If Simulator", "Recommendations", "Parsed Accounts", "Debug"]
)

# ----------------------------
# Parsed Accounts + Manual Fix
# ----------------------------
with tab_parsed:
    st.subheader("Parsed Accounts (with Manual Fix)")
    st.caption("If parsing isn’t perfect, edit fields below. The simulator uses your edited values instantly.")

    editable_cols = [
        "creditor_name",
        "account_number",   # store raw here (UI shows masked only in other places)
        "balance",
        "credit_limit",
        "last_reported",
        "opened_date",
        "status",
        "bureaus",
        "account_type",
        "is_negative",
    ]
    for c in editable_cols:
        if c not in working_df.columns:
            working_df[c] = ""

    # Show editor with safer defaults
    edited = st.data_editor(
        working_df[editable_cols],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "balance": st.column_config.NumberColumn(format="%.2f"),
            "credit_limit": st.column_config.NumberColumn(format="%.2f", help="Optional. Helps utilization calculations."),
            "status": st.column_config.SelectboxColumn(
                options=["ok", "collection", "charge_off", "late", "repossession", "bankruptcy", "foreclosure", "judgment"],
                help="Set the correct negative status for best recommendations."
            ),
            "account_type": st.column_config.SelectboxColumn(
                options=["unknown", "revolving", "installment"],
                help="Set to 'revolving' for credit cards so utilization math works."
            ),
            "is_negative": st.column_config.CheckboxColumn(help="Mark if this account should be treated as negative."),
        },
        key="editor",
    )

    st.session_state["edited_df"] = edited.copy()

    st.success("Manual Fix saved in session. (Nothing is stored to disk; refresh resets unless you re-upload.)")

# ----------------------------
# Recommendations
# ----------------------------
with tab_recs:
    st.subheader("Recommendations & Rapid Rescore Plan")

    # Build negative list
    df_work = st.session_state["edited_df"] if st.session_state["edited_df"] is not None else base_df
    df_work = df_work.copy()

    # Normalize status/is_negative
    if "is_negative" not in df_work.columns:
        df_work["is_negative"] = df_work["status"].astype(str).str.lower().ne("ok")

    neg_df = df_work[df_work["is_negative"] == True].copy()
    neg_df["account_last4"] = neg_df["account_number"].apply(lambda s: mask_account_number(str(s)))

    if neg_df.empty:
        st.info("No negative accounts flagged. If something is negative but parsed as OK, go to **Parsed Accounts** and set status / is_negative.")
    else:
        # Bad Accounts list
        st.markdown("### Bad Accounts")
        bad_cols = ["creditor_name", "account_last4", "balance", "last_reported", "status", "bureaus"]
        show_bad = neg_df[bad_cols].copy()
        st.dataframe(show_bad, use_container_width=True, hide_index=True)

        st.markdown("### Account-by-account actions")
        for _, row in neg_df.iterrows():
            tl = Tradeline(
                creditor_name=str(row.get("creditor_name", "")),
                account_number=str(row.get("account_number", "")),
                balance=coerce_float(row.get("balance", None)),
                last_reported=str(row.get("last_reported", "")),
                opened_date=str(row.get("opened_date", "")),
                status=str(row.get("status", "ok")).lower().strip(),
                bureaus=str(row.get("bureaus", "Unknown")),
                account_type=str(row.get("account_type", "unknown")),
                notes="",
            )
            acct_mask = mask_account_number(tl.account_number)
            bal_str = "—" if tl.balance is None else f"${tl.balance:,.2f}"

            with st.expander(f"{tl.creditor_name} ({acct_mask}) — {tl.status.upper()} — Balance {bal_str}", expanded=False):
                actions = recommend_actions(tl)
                if not actions:
                    st.write("No specific actions detected. Consider reviewing for inaccuracies and improving utilization/positive credit.")
                else:
                    for a in actions:
                        lo, hi = a["impact"]
                        st.markdown(f"**{a['action']}**  \n"
                                    f"- Why: {a['why']}  \n"
                                    f"- Estimated impact: **+{lo} to +{hi}** points (range)  \n"
                                    f"- Effort: **{a['effort']}**")

        # Prioritized action plan (ROI style)
        st.markdown("### Prioritized Action Plan (to reach your goal)")
        neg_counts = {
            "collections": int((df_work["status"] == "collection").sum()),
            "charge_offs": int((df_work["status"] == "charge_off").sum()),
            "lates": int((df_work["status"] == "late").sum()),
        }

        plan = []
        # 1) Utilization (if limits exist or user can set)
        plan.append("Set revolving utilization target (best quick-win): aim **<30%**, then **<10%**, then **1–9%** before rescore.")
        if neg_counts["collections"] > 0:
            plan.append("Attack **collections first**: prioritize PFD or deletion outcomes where possible, then paid updates.")
        if neg_counts["charge_offs"] > 0:
            plan.append("For **charge-offs**: settle/pay and ensure bureaus update balances; dispute inaccuracies if present.")
        if neg_counts["lates"] > 0:
            plan.append("For **lates**: try goodwill removals on otherwise good accounts; ensure everything is current/autopay.")
        plan.append("If file is thin: consider **authorized user** (strong, aged, low utilization) or a new revolving line (only if appropriate).")
        plan.append("After updates: run a **Rapid Rescore** with documentation (receipts/letters/statements) and confirm bureau refresh dates.")

        for i, p in enumerate(plan[:7], start=1):
            st.markdown(f"**{i}.** {p}")

        # Utilization helper
        st.markdown("### Utilization Helper")
        if total_rev_lim <= 0:
            st.info("To compute paydown needed, add **credit_limit** values for revolving accounts in **Parsed Accounts**.")
        else:
            util = (total_rev_bal / total_rev_lim) * 100.0 if total_rev_lim > 0 else 0.0
            st.write(f"Total revolving balance: **${total_rev_bal:,.2f}**")
            st.write(f"Total revolving limits: **${total_rev_lim:,.2f}**")
            st.write(f"Estimated utilization: **{util:.1f}%**")
            paydowns = utilization_paydown_helper(total_rev_bal, total_rev_lim)
            st.write("Paydown needed to reach targets:")
            st.json(paydowns)

        # Checklist export button
        st.markdown("### Rapid Rescore Checklist PDF")
        # top actions overall (simple summary)
        top_actions = [
            "Lower revolving utilization before rescore (30% → 10% → 1–9%).",
        ]
        if neg_counts["collections"] > 0:
            top_actions.append("Prioritize collections for deletion outcomes (PFD or verified disputes).")
        if neg_counts["charge_offs"] > 0:
            top_actions.append("Settle/pay charge-offs and ensure bureau updates to $0 balance.")
        if neg_counts["lates"] > 0:
            top_actions.append("Request goodwill removals for isolated lates; prevent new lates via autopay.")
        top_actions.append("Prepare documentation for rapid rescore submission (receipts/letters/statements).")

        # Build negative Tradeline list for PDF
        neg_tls = []
        for _, row in neg_df.iterrows():
            neg_tls.append(Tradeline(
                creditor_name=str(row.get("creditor_name", "")),
                account_number=str(row.get("account_number", "")),
                balance=coerce_float(row.get("balance", None)),
                last_reported=str(row.get("last_reported", "")),
                opened_date=str(row.get("opened_date", "")),
                status=str(row.get("status", "ok")).lower().strip(),
                bureaus=str(row.get("bureaus", "Unknown")),
                account_type=str(row.get("account_type", "unknown")),
                notes="",
            ))

        pdf_bytes = build_checklist_pdf_bytes(
            score_lo=score_lo, score_hi=score_hi,
            goal_score=int(goal_score),
            top_actions=top_actions,
            negative_lines=neg_tls,
        )

        st.download_button(
            "⬇️ Download Rapid Rescore Checklist (PDF)",
            data=pdf_bytes,
            file_name="Rapid_Rescore_Checklist.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ----------------------------
# What-If Simulator
# ----------------------------
with tab_sim:
    st.subheader("What-If Simulator")
    st.caption("Toggle actions to see a **range-based** estimated score change. Combined scenarios are **not additive**.")

    colA, colB = st.columns([1, 1])

    with colA:
        util_target = st.slider("Utilization target (%)", min_value=1, max_value=100, value=30, step=1)
        remove_collections = st.checkbox("Remove / delete collections (PFD or successful dispute)")
        remove_chargeoffs = st.checkbox("Remove / fix charge-offs (delete or major correction)")
        remove_lates = st.checkbox("Remove late payments (goodwill / correction)")

    with colB:
        add_authorized_user = st.checkbox("Add strong authorized user (aged + low utilization)")
        add_new_revolver = st.checkbox("Add new revolving account (if appropriate)")
        months_pass = st.slider("Time passing (months)", min_value=0, max_value=24, value=0, step=1)

    toggles = {
        "util_target": util_target,
        "remove_collections": remove_collections,
        "remove_chargeoffs": remove_chargeoffs,
        "remove_lates": remove_lates,
        "add_authorized_user": add_authorized_user,
        "add_new_revolver": add_new_revolver,
        "months_pass": months_pass,
    }

    sim_lo, sim_hi, sim_dbg = combined_scenario_adjustment(score_lo, score_hi, toggles)

    delta_lo = sim_lo - score_lo
    delta_hi = sim_hi - score_hi

    st.markdown("### Scenario Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Current Range", f"{score_lo}–{score_hi}")
    r2.metric("Scenario Range", f"{sim_lo}–{sim_hi}")
    r3.metric("Estimated Delta", f"+{delta_lo} to +{delta_hi}")

    st.markdown("### Why it’s not additive")
    st.write(
        "Real scoring models have **interaction effects** (e.g., removing collections and lowering utilization overlap in benefit). "
        "This simulator applies **diminishing returns** when multiple actions are selected."
    )
    with st.expander("Scenario math details (debuggable)"):
        st.json(sim_dbg)

# ----------------------------
# Debug
# ----------------------------
with tab_debug:
    st.subheader("Debug")
    st.caption("Use this to tune parsing across bureau formats: raw text, detected bureau(s), blocks preview, and parsed field counts.")

    st.markdown("### Detected Bureau(s)")
    st.write(st.session_state.get("bureaus", ["Unknown"]))

    st.markdown("### Extraction Metadata")
    st.json(st.session_state.get("extraction_meta", {}))

    st.markdown("### Parse Debug")
    st.json(st.session_state.get("parse_debug", {}))

    st.markdown("### Raw Extracted Text (preview)")
    st.code(safe_snip(st.session_state.get("raw_text", ""), 4000))

    # Blocks preview
    blocks_preview = st.session_state.get("parse_debug", {}).get("raw_blocks_preview", [])
    if blocks_preview:
        st.markdown("### Detected Blocks (preview)")
        for i, b in enumerate(blocks_preview, start=1):
            with st.expander(f"Block {i} (preview)", expanded=False):
                st.code(safe_snip(b, 2500))
