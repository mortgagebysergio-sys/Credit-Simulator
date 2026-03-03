import re
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import streamlit as st
import pdfplumber

# -----------------------------
# Data models
# -----------------------------
@dataclass
class Tradeline:
    creditor: str
    account_type: str = "Unknown"
    balance: Optional[float] = None
    limit: Optional[float] = None
    status: str = "Unknown"
    past_due: Optional[float] = None
    remarks: str = ""
    opened: str = ""
    last_reported: str = ""

@dataclass
class Recommendation:
    action: str
    target: str
    estimated_points_low: int
    estimated_points_high: int
    why: str
    timeline_days: int

# -----------------------------
# Helpers
# -----------------------------
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def parse_money(s: str) -> Optional[float]:
    try:
        s = s.replace(",", "").replace("$", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def extract_text_from_pdf(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)

# -----------------------------
# VERY generic tradeline parsing
# (Works best when PDF has clear trade-line blocks)
# You will refine this once we see your PDF format.
# -----------------------------
def extract_tradelines_generic(text: str) -> List[Tradeline]:
    # Heuristic: split on common separators seen in many reports
    # and then look for key fields within each chunk.
    chunks = re.split(r"\n-{3,}\n|\n={3,}\n|\n\s*ACCOUNT\s+HISTORY\s*\n", text, flags=re.I)
    tradelines: List[Tradeline] = []

    # Patterns (very forgiving)
    creditor_pat = re.compile(r"(CREDITOR|FURNISHER|SUBSCRIBER)\s*:\s*(.+)", re.I)
    balance_pat  = re.compile(r"(BALANCE)\s*:\s*\$?([\d,]+(\.\d+)?)", re.I)
    limit_pat    = re.compile(r"(CREDIT\s*LIMIT|HIGH\s*CREDIT|LIMIT)\s*:\s*\$?([\d,]+(\.\d+)?)", re.I)
    status_pat   = re.compile(r"(STATUS)\s*:\s*(.+)", re.I)
    type_pat     = re.compile(r"(ACCOUNT\s*TYPE|TYPE)\s*:\s*(.+)", re.I)
    pastdue_pat  = re.compile(r"(PAST\s*DUE)\s*:\s*\$?([\d,]+(\.\d+)?)", re.I)
    remarks_pat  = re.compile(r"(REMARKS|COMMENTS)\s*:\s*(.+)", re.I)

    # Also catch “COLLECTION”, “CHARGE OFF”, etc.
    bad_words = ["COLLECTION", "CHARGE OFF", "CHARGEOFF", "LATE", "DELINQUENT", "DEROGATORY", "PAST DUE"]

    for c in chunks:
        c_strip = c.strip()
        if len(c_strip) < 250:
            continue

        creditor = None
        m = creditor_pat.search(c_strip)
        if m:
            creditor = m.group(2).strip()
        else:
            # fallback: pick a likely uppercase header line
            lines = [ln.strip() for ln in c_strip.splitlines() if ln.strip()]
            if lines:
                # choose first line if it looks like a name
                creditor = lines[0][:80]

        bal = None
        m = balance_pat.search(c_strip)
        if m:
            bal = parse_money(m.group(2))

        lim = None
        m = limit_pat.search(c_strip)
        if m:
            lim = parse_money(m.group(2))

        stat = "Unknown"
        m = status_pat.search(c_strip)
        if m:
            stat = m.group(2).strip()[:120]
        else:
            # infer from keywords
            upper = c_strip.upper()
            for w in bad_words:
                if w in upper:
                    stat = w.title()
                    break

        acct_type = "Unknown"
        m = type_pat.search(c_strip)
        if m:
            acct_type = m.group(2).strip()[:120]

        past_due = None
        m = pastdue_pat.search(c_strip)
        if m:
            past_due = parse_money(m.group(2))

        remarks = ""
        m = remarks_pat.search(c_strip)
        if m:
            remarks = m.group(2).strip()[:200]

        if creditor:
            tradelines.append(
                Tradeline(
                    creditor=creditor,
                    account_type=acct_type,
                    balance=bal,
                    limit=lim,
                    status=stat,
                    past_due=past_due,
                    remarks=remarks,
                )
            )

    # De-dupe by creditor + balance + limit
    seen = set()
    unique = []
    for t in tradelines:
        key = (t.creditor.lower(), t.balance, t.limit, t.status.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique

# -----------------------------
# Conservative scoring simulator (NOT FICO)
# -----------------------------
UTIL_TIERS = [
    (0.09, (12, 30)),   # <9% is best
    (0.29, (8, 22)),    # 9-29%
    (0.49, (4, 14)),    # 30-49%
    (0.69, (0, 8)),     # 50-69%
    (0.89, (-6, 2)),    # 70-89%
    (1.00, (-12, -2)),  # 90-100%+
]

def utilization_points(util: float) -> Tuple[int, int]:
    util = max(0.0, util)
    for cap, pts in UTIL_TIERS:
        if util <= cap:
            return pts
    return (-12, -2)

def estimate_overall_utilization(tradelines: List[Tradeline]) -> Optional[float]:
    total_bal = 0.0
    total_lim = 0.0
    found = False
    for t in tradelines:
        if t.balance is not None and t.limit is not None and t.limit > 0:
            total_bal += t.balance
            total_lim += t.limit
            found = True
    if not found or total_lim <= 0:
        return None
    return total_bal / total_lim

def generate_recommendations(tradelines: List[Tradeline]) -> List[Recommendation]:
    recs: List[Recommendation] = []

    # 1) Utilization recommendations
    util = estimate_overall_utilization(tradelines)
    if util is not None:
        # suggest tier drops
        current_tier_pts = utilization_points(util)
        targets = [0.69, 0.49, 0.29, 0.09]
        for target in targets:
            if util > target:
                # Conservative gain estimate: use difference between tier point ranges
                new_pts = utilization_points(target)
                low_gain = clamp(new_pts[0] - current_tier_pts[1], 0, 60)
                high_gain = clamp(new_pts[1] - current_tier_pts[0], 0, 80)
                # timeline: utilization updates as accounts report (often 30-45 days)
                recs.append(
                    Recommendation(
                        action=f"Pay down revolving utilization to ≤ {int(target*100)}%",
                        target="All revolving cards (overall)",
                        estimated_points_low=low_gain,
                        estimated_points_high=high_gain,
                        why="Crossing utilization tiers is one of the most reliable short-term movers in mortgage scoring models.",
                        timeline_days=30,
                    )
                )

    # 2) Collections / charge-offs / past-due
    for t in tradelines:
        s = (t.status or "").upper()
        r = (t.remarks or "").upper()

        is_collection = "COLLECTION" in s or "COLLECTION" in r
        is_chargeoff = "CHARGE" in s or "CHARGEOFF" in s or "CHARGE OFF" in s or "CHARGE" in r
        is_past_due  = (t.past_due or 0) > 0 or "PAST DUE" in s or "DELINQUENT" in s

        if is_past_due:
            recs.append(
                Recommendation(
                    action="Bring account current (cure past-due) ASAP",
                    target=t.creditor,
                    estimated_points_low=15,
                    estimated_points_high=45,
                    why="Active delinquencies are typically more damaging than older derogatories. Getting current can unlock approvals faster.",
                    timeline_days=30,
                )
            )

        if is_collection:
            recs.append(
                Recommendation(
                    action="Attempt pay-for-delete (PFD); if not possible, negotiate settlement and ensure reporting updates",
                    target=t.creditor,
                    estimated_points_low=5,
                    estimated_points_high=25,
                    why="Deletion often helps more than simply 'paid', but impacts vary by bureau data and file thickness. We estimate conservatively.",
                    timeline_days=60,
                )
            )

        if is_chargeoff:
            recs.append(
                Recommendation(
                    action="Negotiate 'delete' or 'remove remarks'; if not possible, settle and ensure balance updates to $0",
                    target=t.creditor,
                    estimated_points_low=3,
                    estimated_points_high=18,
                    why="Charge-offs can remain; the best-case is deletion or removal of derogatory language, but even balance updates can help DTI/underwriting and sometimes score modestly.",
                    timeline_days=60,
                )
            )

    # 3) Micro-optimizations: individual card utilization
    for t in tradelines:
        if t.balance is not None and t.limit is not None and t.limit > 0:
            u = t.balance / t.limit
            if u >= 0.9:
                recs.append(
                    Recommendation(
                        action="Pay this card down below 89%, then 49%, then 29% (tier steps)",
                        target=t.creditor,
                        estimated_points_low=6,
                        estimated_points_high=22,
                        why="Single-card maxing can hurt even if overall utilization is okay. Tier drops can help quickly when the card reports.",
                        timeline_days=30,
                    )
                )

    # Rank by highest potential
    recs.sort(key=lambda x: (x.estimated_points_high, x.estimated_points_low), reverse=True)

    # De-dupe similar actions for same target
    out = []
    seen = set()
    for r in recs:
        key = (r.action, r.target)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def project_scores(base_score: int, recs: List[Recommendation]) -> Dict[int, Tuple[int, int]]:
    """
    Returns conservative projections for 30/60/90 days based on actions likely to post by then.
    This is a heuristic, not FICO.
    """
    horizons = [30, 60, 90]
    proj = {}
    for h in horizons:
        low = 0
        high = 0
        for r in recs:
            if r.timeline_days <= h:
                low += r.estimated_points_low
                high += r.estimated_points_high
        proj[h] = (clamp(base_score + low, 300, 850), clamp(base_score + high, 300, 850))
    return proj

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Mortgage Credit Simulator (Conservative)", layout="wide")
st.title("Mortgage Credit Simulator (Conservative, Non-FICO)")

st.caption(
    "Uploads a credit report PDF, extracts accounts (best-effort), ranks actions by likely impact, "
    "and gives conservative 30/60/90-day score projections. Not an official FICO score product."
)

colA, colB = st.columns([2, 1], gap="large")

with colB:
    base_score = st.number_input("Starting middle score (estimate)", min_value=300, max_value=850, value=660, step=1)
    st.markdown("**Tip:** If you don’t know, enter the client’s middle score.")
    st.divider()
    st.markdown("### Privacy note")
    st.write("For a production deployment, we can disable logging, avoid saving PDFs, and add a client data purge button.")

with colA:
    pdf = st.file_uploader("Upload credit report PDF (text-based works best)", type=["pdf"])
    if not pdf:
        st.stop()

    with st.spinner("Reading PDF…"):
        text = extract_text_from_pdf(pdf)

    if not text.strip():
        st.error("No extractable text found. This PDF may be scanned (image-based). The MVP works best with text-based PDFs.")
        st.stop()

    st.success("PDF text extracted.")
    with st.expander("Preview extracted text (first 2,000 chars)"):
        st.code(text[:2000])

    with st.spinner("Extracting tradelines (best-effort)…"):
        tradelines = extract_tradelines_generic(text)

    st.subheader("Detected accounts (best-effort)")
    st.write(f"Found **{len(tradelines)}** possible tradelines.")
    st.dataframe([asdict(t) for t in tradelines], use_container_width=True)

    recs = generate_recommendations(tradelines)

    st.subheader("Ranked actions (highest impact first)")
    st.dataframe([asdict(r) for r in recs], use_container_width=True)

    proj = project_scores(int(base_score), recs)
    st.subheader("Conservative projections")
    pcols = st.columns(3)
    for i, h in enumerate([30, 60, 90]):
        lo, hi = proj[h]
        with pcols[i]:
            st.metric(f"{h}-day projection", f"{lo} – {hi}")

    st.divider()
    st.subheader("What to do next (client action plan)")
    for r in recs[:10]:
        st.markdown(
            f"**{r.target}** — {r.action}  \n"
            f"Estimated gain: **+{r.estimated_points_low} to +{r.estimated_points_high}** (posts ~{r.timeline_days} days)  \n"
            f"Why: {r.why}"
        )
