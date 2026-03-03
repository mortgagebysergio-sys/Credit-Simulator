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
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return f"••••{digits[-4:]}"
    # Birchwood sometimes uses alphanumeric IDs; still mask as unknown
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
# Date + Money helpers (Birchwood uses MM/YY often)
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
# Extraction pipeline (Text then OCR)
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
# Birchwood-specific parsing
# =============================
HEADER_NOISE_PREFIXES = (
    "FILE #", "SEND TO", "CUST.", "BIRCHWOOD CREDIT SERVICES",
    "ECOA KEY:", "PROPERTY ADDRESS", "APPLICANT", "SOC SEC",
    "MARITAL STATUS", "The information is furnished",
)

def is_header_noise(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    for p in HEADER_NOISE_PREFIXES:
        if s.startswith(p):
            return True
    if s.startswith("Page "):
        return True
    return False


def is_account_id_line(line: str) -> bool:
    """
    Birchwood account IDs can be digits or alphanumeric (e.g., M318R2G9).
    """
    s = (line or "").strip()
    if not s:
        return False
    # avoid MM/YY
    if re.fullmatch(r"(0?[1-9]|1[0-2])[/\-]\d{2,4}", s):
        return False
    # all digits
    if re.fullmatch(r"\d{7,}", s):
        return True
    # alphanumeric
    if re.fullmatch(r"[A-Z0-9]{6,}", s.upper()):
        if s.upper() in {"OPENED", "REPORTED", "BALANCE", "PAYMENT", "PAST", "DUE"}:
            return False
        return True
    return False


def detect_bureaus_from_block(block: str) -> str:
    t = (block or "").upper()
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


def extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    t = (text or "")
    si = t.upper().find(start_marker.upper())
    if si < 0:
        return ""
    sub = t[si:]
    if end_marker:
        ei = sub.upper().find(end_marker.upper())
        if ei > 0:
            sub = sub[:ei]
    return sub


def birchwood_negative_status(block: str) -> Tuple[bool, str]:
    """
    Birchwood negatives:
      - CHARGE OFF / COLLECTION
      - DELINQ 30/60/90
      - delinquency bucket counts 30-59 / 60-89 / 90+
    """
    t = (block or "").upper()

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

    # bucket counts
    def count_bucket(label: str) -> int:
        m2 = re.search(rf"(?i){re.escape(label)}\s*\n\s*(\d+)", block)
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

    # Other public record style keywords (rare but possible)
    if "BANKRUPT" in t or "CHAPTER 7" in t or "CHAPTER 13" in t:
        return True, "Bankruptcy"
    if "FORECLOS" in t:
        return True, "Foreclosure"
    if "REPOS" in t:
        return True, "Repossession"
    if "JUDG" in t:
        return True, "Judgment"

    return False, ""


def build_birchwood_tradeline_blocks(full_text: str) -> Tuple[List[str], List[str]]:
    """
    Build one block per tradeline from TRADELINES section.
    """
    notes = []
    tradelines_text = extract_section(full_text, "TRADELINES", "TRADE SUMMARY")
    if not tradelines_text:
        return [], ["Birchwood parser: TRADELINES section not found."]

    lines = [ln.rstrip() for ln in tradelines_text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    blocks = []
    i = 0

    def looks_like_start(idx: int) -> bool:
        if idx >= len(lines) - 1:
            return False
        a = lines[idx].strip()
        b = lines[idx + 1].strip()
        if is_header_noise(a):
            return False
        # creditor-ish line followed by account id soon
        if is_account_id_line(b):
            return True
        if idx + 2 < len(lines) and is_account_id_line(lines[idx + 2].strip()):
            return True
        return False

    while i < len(lines):
        if not looks_like_start(i):
            i += 1
            continue

        creditor_lines = []
        j = i
        while j < len(lines) and not is_account_id_line(lines[j].strip()):
            if not is_header_noise(lines[j]):
                creditor_lines.append(lines[j].strip())
            j += 1

        if j >= len(lines):
            break

        acct_line = lines[j].strip()
        j += 1

        # optional numeric suffix line
        suffix = ""
        if j < len(lines) and re.fullmatch(r"\d{1,3}", lines[j].strip()):
            suffix = lines[j].strip()
            j += 1

        k = j
        while k < len(lines) and not looks_like_start(k):
            k += 1

        blk_lines = creditor_lines + [acct_line] + ([suffix] if suffix else []) + lines[j:k]
        blk = "\n".join(blk_lines).strip()
        if blk:
            blocks.append(blk)
        i = k

    notes.append(f"Birchwood parser: built {len(blocks)} tradeline blocks from TRADELINES.")
    return blocks, notes


def parse_birchwood_tradeline(block: str) -> Tuple[NegativeAccount, List[str]]:
    notes = []
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]

    creditor_parts = []
    acct_raw = ""
    idx = 0
    while idx < len(lines):
        if is_account_id_line(lines[idx]):
            acct_raw = lines[idx]
            idx += 1
            if idx < len(lines) and re.fullmatch(r"\d{1,3}", lines[idx]):
                acct_raw = acct_raw + lines[idx]
                idx += 1
            break
        creditor_parts.append(lines[idx])
        idx += 1

    creditor = " ".join(creditor_parts).strip()
    creditor = re.sub(r"\s{2,}", " ", creditor)[:90]
    notes.append("Creditor: leading lines" if creditor else "Creditor: missing")

    if not acct_raw:
        m = ACCT_LIKE_RE.search(block or "")
        acct_raw = m.group(0) if m else ""
        notes.append("Account#: heuristic fallback" if acct_raw else "Account#: missing")
    else:
        notes.append("Account#: account-id line")

    masked_acct = mask_account_number(acct_raw)

    opened_dt = None
    reported_dt = None
    bal = None

    m = re.search(r"(?i)\bOpened\b\s*\n\s*([0-9]{1,2}[/\-][0-9]{2,4})", block)
    if m:
        opened_dt = parse_mm_yy(m.group(1))
        notes.append("Opened: label match")
    else:
        notes.append("Opened: missing")

    m = re.search(r"(?i)\bReported\b\s*\n\s*([0-9]{1,2}[/\-][0-9]{2,4})", block)
    if m:
        reported_dt = parse_mm_yy(m.group(1))
        notes.append("Reported/Last reported: label match")
    else:
        notes.append("Reported/Last reported: missing")

    m = re.search(r"(?i)\bBalance\b\s*\n\s*(\$?\s*[0-9,]+(?:\.[0-9]{2})?)", block)
    if m:
        bal = parse_money(m.group(1))
        notes.append("Balance: label match")
    else:
        m2 = re.search(r"(?i)\bBalance\b[^0-9$]{0,10}(\$?\s*[0-9,]+(?:\.[0-9]{2})?)", block)
        if m2:
            bal = parse_money(m2.group(1))
            notes.append("Balance: inline fallback")
        else:
            notes.append("Balance: missing")

    bureaus = detect_bureaus_from_block(block)
    notes.append(f"Bureaus: {bureaus}")

    is_neg, status = birchwood_negative_status(block)
    notes.append(f"Negative: {is_neg} ({status})")

    acct = NegativeAccount(
        creditor_name=creditor,
        masked_account_number=masked_acct,
        current_balance=bal,
        last_reported_date=reported_dt,
        date_opened=opened_dt,
        age_of_account=calc_age(opened_dt),
        negative_type_status=status,
        bureaus=bureaus,
        estimated_impact="",  # set later
        raw_block_snippet=mask_pii_in_snippet((block or "")[:1600]),
    )
    return acct, notes


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
# PDF export
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
        "Disclaimer: This report is an informational extraction from the uploaded credit bureau PDF. "
        "Estimated impact ranges are heuristic, not an exact score change, and vary by scoring model "
        "(e.g., FICO vs Vantage), credit file thickness, utilization, and reporting details. "
        "Always verify against the source report and/or a qualified professional."
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
st.caption("Birchwood-only parser. Uploads processed in-memory. Account numbers masked (last 4). OCR fallback enabled.")

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

with st.spinner("Building tradeline blocks + parsing negatives..."):
    blocks, build_notes = build_birchwood_tradeline_blocks(extracted_text)
    accounts: List[NegativeAccount] = []
    parse_notes: List[ParseNote] = []

    for i, blk in enumerate(blocks):
        is_neg, _ = birchwood_negative_status(blk)
        if not is_neg:
            continue
        acct, notes = parse_birchwood_tradeline(blk)
        if acct.negative_type_status:
            tier, rng = impact_range_for_account(acct.negative_type_status, acct.last_reported_date, acct.current_balance)
            acct.estimated_impact = f"{tier}: {rng}"
        accounts.append(acct)
        parse_notes.append(ParseNote(block_index=i, notes=notes))

    # dedupe
    deduped = []
    seen = set()
    for a in accounts:
        key = (
            (a.creditor_name or "").lower().strip()[:40],
            a.masked_account_number,
            (a.negative_type_status or "").lower(),
            int((a.current_balance or 0.0) // 10),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    accounts = deduped

confidence, conf_debug = compute_confidence(extracted_text, accounts)

state["extraction_mode"] = mode
state["confidence"] = confidence
state["blocks"] = blocks
state["build_notes"] = build_notes
state["parse_notes"] = parse_notes
state["conf_debug"] = conf_debug
state["base_accounts"] = accounts

if "manual_df" not in state:
    state["manual_df"] = build_manual_editor_df(accounts)

final_accounts = apply_manual_overrides(state["base_accounts"], state["manual_df"])
final_df = to_table_df(final_accounts)

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
                st.caption("Raw block snippet (PII-masked):")
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
    updated_df = to_table_df(updated_accounts)
    st.subheader("Preview: Updated Negative Accounts Table")
    st.dataframe(updated_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Debug")

    st.markdown("### Extraction Details")
    st.write({
        "Extraction Mode": mode,
        "Confidence": confidence,
        "Text Length": int(state["conf_debug"]["text_length"]),
        "Completion Rate": round(state["conf_debug"]["completion_rate"], 3),
        "Confidence Score": round(state["conf_debug"]["score"], 3),
        "Tradeline Blocks Built": len(state["blocks"]),
        "Negative Accounts Parsed": len(state["base_accounts"]),
    })

    st.markdown("### Block Builder Notes")
    for n in state.get("build_notes", []):
        st.info(n)

    st.markdown("### Field Extraction Notes (per parsed negative block)")
    note_rows = [{"Block Index": n.block_index, "Notes": " | ".join(n.notes)} for n in state["parse_notes"]]
    st.dataframe(pd.DataFrame(note_rows), use_container_width=True, hide_index=True)

    st.markdown("### Raw Extracted Text Preview (PII-masked best-effort)")
    st.code(mask_pii_in_snippet((extracted_text or "")[:12000]), language="text")

    st.markdown("### Tradeline Blocks (first 30, PII-masked)")
    for i, blk in enumerate(state["blocks"][:30]):
        with st.expander(f"Block #{i}"):
            st.code(mask_pii_in_snippet(blk[:2200]), language="text")
