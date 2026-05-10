import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO

st.set_page_config(
    page_title="🎱 Bingo Manager",
    layout="wide",
    page_icon="🎱",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* General */
    .block-container { padding-top: 1rem; }

    /* Header */
    .bingo-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 22px 30px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 18px;
    }
    .bingo-header h1 { color: white; margin: 0; font-size: 38px; letter-spacing: 2px; }
    .round-badge {
        background: #e94560;
        color: white;
        padding: 5px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin-top: 8px;
    }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
        color: #1a1a2e;
        padding: 18px 24px;
        border-radius: 14px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.55);
        animation: none;
    }

    /* Called chip */
    .chip {
        display: inline-block;
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        margin: 2px;
        font-size: 13px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "cards": [],           # list[pd.DataFrame]
    "card_names": [],      # list[str]
    "called_values": [],   # list[str] – normalised to upper
    "winners": set(),      # set[int] – indices of winning cards
    "round": 1,
    "last_warning": None,
    "rules": {
        "check_rows": True,
        "check_cols": True,
        "check_diagonals": True,
        "check_full_card": False,
        "free_space": True,
        "free_space_row": 2,
        "free_space_col": 2,
    },
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Core Logic ───────────────────────────────────────────────────────────────
def _is_marked(val, r, c, called_set, rules):
    if rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]:
        return True
    return str(val).strip().upper() in called_set


def check_bingo(card_df, called_values, rules):
    called_set = {str(v).strip().upper() for v in called_values}
    nrows, ncols = card_df.shape

    def m(r, c):
        return _is_marked(card_df.iloc[r, c], r, c, called_set, rules)

    if rules.get("check_rows"):
        if any(all(m(r, c) for c in range(ncols)) for r in range(nrows)):
            return True
    if rules.get("check_cols"):
        if any(all(m(r, c) for r in range(nrows)) for c in range(ncols)):
            return True
    if rules.get("check_diagonals") and nrows == ncols:
        if all(m(i, i) for i in range(nrows)):
            return True
        if all(m(i, nrows - 1 - i) for i in range(nrows)):
            return True
    if rules.get("check_full_card"):
        if all(m(r, c) for r in range(nrows) for c in range(ncols)):
            return True
    return False


def recalc_winners():
    st.session_state.winners = {
        i
        for i, card in enumerate(st.session_state.cards)
        if check_bingo(card, st.session_state.called_values, st.session_state.rules)
    }


# ─── Rendering ────────────────────────────────────────────────────────────────
def render_card_html(idx):
    card = st.session_state.cards[idx]
    name = st.session_state.card_names[idx]
    called_values = st.session_state.called_values
    rules = st.session_state.rules
    is_winner = idx in st.session_state.winners

    called_set = {str(v).strip().upper() for v in called_values}
    nrows, ncols = card.shape

    border = "3px solid #ffd700" if is_winner else "2px solid #dee2e6"
    shadow = "0 0 22px rgba(255,215,0,0.45)" if is_winner else "0 2px 8px rgba(0,0,0,0.08)"
    bg = "#fffdf0" if is_winner else "#ffffff"
    name_color = "#c8870a" if is_winner else "#343a40"
    prefix = "🏆 " if is_winner else ""

    html = (
        f'<div style="border:{border};border-radius:13px;padding:12px;'
        f'box-shadow:{shadow};background:{bg};margin-bottom:4px;">'
        f'<div style="text-align:center;font-size:15px;font-weight:700;color:{name_color};margin-bottom:8px;">'
        f"{prefix}{name}"
        f"</div>"
        f'<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
    )

    # BINGO header row for 5-column cards
    if ncols == 5:
        html += "<tr>"
        for letter in "BINGO":
            html += (
                f'<th style="background:#1a1a2e;color:#ffd700;padding:8px 2px;'
                f"text-align:center;font-size:15px;font-weight:bold;"
                f'border:1px solid #333;">{letter}</th>'
            )
        html += "</tr>"

    for r in range(nrows):
        html += "<tr>"
        for c in range(ncols):
            val = card.iloc[r, c]
            free = rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]
            marked = free or str(val).strip().upper() in called_set
            display = "★" if free else str(val)

            if is_winner and marked:
                cell_bg, fg, fw = "#ffd700", "#1a1a2e", "bold"
            elif marked:
                cell_bg, fg, fw = "#28a745", "#ffffff", "bold"
            else:
                cell_bg, fg, fw = "#f8f9fa", "#495057", "normal"

            html += (
                f'<td style="background:{cell_bg};color:{fg};font-weight:{fw};'
                f"padding:10px 2px;text-align:center;border:1px solid #dee2e6;"
                f'font-size:14px;border-radius:3px;">{display}</td>'
            )
        html += "</tr>"

    html += "</table></div>"
    return html


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # --- Rules upload ---
    with st.expander("📜 Upload Rules (JSON)", expanded=False):
        rules_file = st.file_uploader("Rules JSON file", type=["json"], key="rules_upload")
        if rules_file:
            try:
                loaded = json.load(rules_file)
                st.session_state.rules.update(loaded)
                recalc_winners()
                st.success("Rules applied!")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        st.caption("Expected JSON format:")
        st.code(
            json.dumps(
                {
                    "check_rows": True,
                    "check_cols": True,
                    "check_diagonals": True,
                    "check_full_card": False,
                    "free_space": True,
                    "free_space_row": 2,
                    "free_space_col": 2,
                },
                indent=2,
            ),
            language="json",
        )

    st.markdown("---")
    st.markdown("### 📋 Winning Conditions")

    rules = st.session_state.rules
    prev = dict(rules)

    rules["check_rows"]      = st.checkbox("✅ Complete Row",        value=rules["check_rows"])
    rules["check_cols"]      = st.checkbox("✅ Complete Column",     value=rules["check_cols"])
    rules["check_diagonals"] = st.checkbox("✅ Diagonal",            value=rules["check_diagonals"])
    rules["check_full_card"] = st.checkbox("✅ Full Card (Blackout)", value=rules["check_full_card"])
    rules["free_space"]      = st.checkbox("⭐ Free Space (center)", value=rules["free_space"])

    if any(rules[k] != prev[k] for k in prev):
        recalc_winners()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔁 Round Management")

    if st.button("🔄 New Round  —  Keep Cards", use_container_width=True):
        st.session_state.called_values = []
        st.session_state.winners = set()
        st.session_state.last_warning = None
        st.session_state.round += 1
        st.rerun()

    if st.button("🆕 New Round  —  Upload New Cards", use_container_width=True):
        st.session_state.called_values = []
        st.session_state.winners = set()
        st.session_state.cards = []
        st.session_state.card_names = []
        st.session_state.last_warning = None
        st.session_state.round += 1
        st.rerun()

    st.markdown("---")

    # --- Called values history ---
    if st.session_state.called_values:
        st.markdown("### 📢 Called Values")
        chips = " ".join(
            f'<span class="chip">{v}</span>'
            for v in st.session_state.called_values
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"Total called: {len(st.session_state.called_values)}")

        if st.button("↩️ Undo Last Call", use_container_width=True):
            st.session_state.called_values.pop()
            st.session_state.last_warning = None
            recalc_winners()
            st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="bingo-header">
  <h1>🎱 Bingo Manager</h1>
  <span class="round-badge">Round {st.session_state.round}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ── Upload Expander ────────────────────────────────────────────────────────────
with st.expander(
    "📤 Upload Bingo Cards",
    expanded=not bool(st.session_state.cards),
):
    tab_csv, tab_excel = st.tabs(
        ["📄 CSV — one file per card", "📊 Excel — one sheet per card"]
    )

    # CSV tab
    with tab_csv:
        with st.expander("📖 CSV format guide", expanded=False):
            st.markdown(
                """
**No header row needed.** Each row of the CSV corresponds to one row of the bingo card.

Example for a standard 5×5 BINGO card:
"""
            )
            sample = pd.DataFrame(
                [
                    ["B1", "I16", "N31", "G46", "O61"],
                    ["B2", "I17", "N32", "G47", "O62"],
                    ["B3", "I18", "FREE", "G48", "O63"],
                    ["B4", "I19", "N34", "G49", "O64"],
                    ["B5", "I20", "N35", "G50", "O65"],
                ]
            )
            st.dataframe(sample, hide_index=True, use_container_width=True)
            st.caption(
                "Save as .csv without a header row. "
                'The FREE cell in the centre is handled automatically by the ⭐ Free Space rule — '
                "you can leave that cell blank or write anything."
            )

        csv_files = st.file_uploader(
            "Upload CSV card files",
            type=["csv"],
            accept_multiple_files=True,
            key="csv_up",
        )
        if csv_files and st.button("➕ Add These Cards", key="add_csv"):
            added = 0
            for f in csv_files:
                try:
                    df = pd.read_csv(StringIO(f.read().decode("utf-8")), header=None)
                    df = df.astype(str).applymap(str.strip)
                    name = f.name.rsplit(".", 1)[0]
                    st.session_state.cards.append(df)
                    st.session_state.card_names.append(name)
                    added += 1
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")
            recalc_winners()
            st.success(f"✅ {added} card(s) added!")
            st.rerun()

    # Excel tab
    with tab_excel:
        st.markdown("Each **sheet** in the workbook becomes one bingo card. No header row needed.")
        excel_file = st.file_uploader(
            "Upload Excel file", type=["xlsx", "xls"], key="excel_up"
        )
        if excel_file and st.button("➕ Add Cards from Excel", key="add_excel"):
            try:
                raw = excel_file.read()
                xl = pd.ExcelFile(BytesIO(raw))
                added = 0
                for sheet in xl.sheet_names:
                    df = pd.read_excel(BytesIO(raw), sheet_name=sheet, header=None)
                    df = df.astype(str).applymap(str.strip)
                    st.session_state.cards.append(df)
                    st.session_state.card_names.append(sheet)
                    added += 1
                recalc_winners()
                st.success(f"✅ {added} card(s) added!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not read Excel file: {e}")


# ── Game Area ─────────────────────────────────────────────────────────────────
if not st.session_state.cards:
    st.info(
        "👆 Upload bingo cards using the section above to get started!  \n"
        "You can upload CSV files (one per card) or an Excel file (one sheet per card)."
    )
    st.stop()

# — Call form —
st.markdown("### 📢 Call a Value")
with st.form("call_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        raw_input = st.text_input(
            "value",
            placeholder="Type a value and press Enter or click Call  (e.g.  B7 · 42 · STAR)",
            label_visibility="collapsed",
        )
    with c2:
        submitted = st.form_submit_button("📣 Call!", use_container_width=True, type="primary")

    if submitted:
        val = raw_input.strip().upper()
        if not val:
            st.session_state.last_warning = "Please enter a value before calling."
        elif val in [str(v).upper() for v in st.session_state.called_values]:
            st.session_state.last_warning = f"'{val}' was already called!"
        else:
            st.session_state.called_values.append(val)
            st.session_state.last_warning = None
            recalc_winners()

if st.session_state.last_warning:
    st.warning(st.session_state.last_warning)

# — Winner banners —
if st.session_state.winners:
    for win_idx in sorted(st.session_state.winners):
        win_name = st.session_state.card_names[win_idx]
        st.markdown(
            f'<div class="winner-banner">🎉 BINGO! &nbsp;&nbsp; {win_name} &nbsp;&nbsp; 🎉</div>',
            unsafe_allow_html=True,
        )

# — Called values strip —
if st.session_state.called_values:
    chips = " ".join(
        f'<span class="chip">{v}</span>' for v in st.session_state.called_values
    )
    st.markdown(
        f"<div style='margin:10px 0;'><strong>Called ({len(st.session_state.called_values)}):</strong>&nbsp; {chips}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# — Cards grid —
n = len(st.session_state.cards)
st.markdown(f"### 🃏 Cards &nbsp; <span style='color:#6c757d;font-size:16px;'>({n})</span>", unsafe_allow_html=True)

num_cols = min(n, 3)
cols = st.columns(num_cols, gap="medium")

for i in range(n):
    with cols[i % num_cols]:
        st.markdown(render_card_html(i), unsafe_allow_html=True)
        if st.button("🗑️ Remove", key=f"rm_{i}_{st.session_state.round}", use_container_width=True):
            st.session_state.cards.pop(i)
            st.session_state.card_names.pop(i)
            recalc_winners()
            st.rerun()
