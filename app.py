import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="🎱 Bingo Manager",
    layout="wide",
    page_icon="🎱",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .bingo-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 22px 30px; border-radius: 14px;
        text-align: center; margin-bottom: 18px;
    }
    .bingo-header h1 { color: white; margin: 0; font-size: 38px; letter-spacing: 2px; }
    .round-badge {
        background: #e94560; color: white;
        padding: 5px 18px; border-radius: 20px;
        font-weight: bold; font-size: 14px;
        display: inline-block; margin-top: 8px;
    }
    .winner-banner {
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
        color: #1a1a2e; padding: 18px 24px; border-radius: 14px;
        text-align: center; font-size: 22px; font-weight: bold;
        margin: 8px 0; box-shadow: 0 4px 20px rgba(255,215,0,0.55);
    }
    .chip {
        display: inline-block; background: #28a745; color: white;
        padding: 4px 12px; border-radius: 15px;
        margin: 2px; font-size: 13px; font-weight: bold;
    }
    .rules-box {
        background: #e8f4fd; border-left: 4px solid #0f3460;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
        font-size: 14px; color: #1a1a2e;
    }
    .grid-input input {
        text-align: center !important;
        font-weight: bold !important;
        font-size: 15px !important;
    }
    div[data-testid="stTextInput"] input {
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "cards": [],
    "card_names": [],
    "card_thumbs": [],      # base64 data-URLs
    "called_values": [],
    "winners": set(),
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
    # Entry state
    "entry_grid_size": 5,
    "entry_free_row": 2,
    "entry_free_col": 2,
    "entry_has_free": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Game Logic ───────────────────────────────────────────────────────────────
_BINGO = "BINGO"

def _marked(card_df, r, c, called_set, rules):
    """
    Flexible matching that handles three common situations:
      1. Cell "8"  vs called "B8"  → match  (number-only card, letter-prefixed call)
      2. Cell "B8" vs called "B8"  → match  (both have prefix)
      3. Cell "B8" vs called "8"   → match  (cell has prefix, call is bare number)
    Works for any column count; BINGO-letter logic activates only for 5-col cards.
    """
    if rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]:
        return True

    cell_val = str(card_df.iloc[r, c]).strip().upper()
    ncols    = card_df.shape[1]

    # Direct match (covers cases 2 and 3 when both sides are the same format)
    if cell_val in called_set:
        return True

    # BINGO-prefix bridge for 5-column cards
    if ncols == 5 and c < 5:
        letter = _BINGO[c]
        # Case 1 – cell is bare number, called has letter prefix (e.g. "8" vs "B8")
        if f"{letter}{cell_val}" in called_set:
            return True
        # Case 3 – cell has letter prefix, called is bare number (e.g. "B8" vs "8")
        if cell_val.startswith(letter) and cell_val[1:] in called_set:
            return True

    # Generic fallback – strip any single leading alpha from each called value and compare
    for called in called_set:
        if len(called) >= 2 and called[0].isalpha() and not called[0].isdigit():
            if called[1:] == cell_val:
                return True

    return False


def check_bingo(card_df, called_values, rules):
    called_set = {str(v).strip().upper() for v in called_values}
    nrows, ncols = card_df.shape
    m = lambda r, c: _marked(card_df, r, c, called_set, rules)

    if rules.get("check_rows"):
        if any(all(m(r, c) for c in range(ncols)) for r in range(nrows)):
            return True
    if rules.get("check_cols"):
        if any(all(m(r, c) for r in range(nrows)) for c in range(ncols)):
            return True
    if rules.get("check_diagonals") and nrows == ncols:
        if all(m(i, i) for i in range(nrows)) or all(m(i, nrows-1-i) for i in range(nrows)):
            return True
    if rules.get("check_full_card"):
        if all(m(r, c) for r in range(nrows) for c in range(ncols)):
            return True
    return False


def recalc_winners():
    st.session_state.winners = {
        i for i, card in enumerate(st.session_state.cards)
        if check_bingo(card, st.session_state.called_values, st.session_state.rules)
    }


# ─── Card Rendering ───────────────────────────────────────────────────────────
def render_card_html(idx: int) -> str:
    card      = st.session_state.cards[idx]
    name      = st.session_state.card_names[idx]
    thumb     = st.session_state.card_thumbs[idx]
    called    = st.session_state.called_values
    rules     = st.session_state.rules
    is_winner = idx in st.session_state.winners

    called_set   = {str(v).strip().upper() for v in called}
    nrows, ncols = card.shape

    border  = "3px solid #ffd700" if is_winner else "2px solid #dee2e6"
    shadow  = "0 0 22px rgba(255,215,0,0.45)" if is_winner else "0 2px 8px rgba(0,0,0,0.08)"
    bg      = "#fffdf0" if is_winner else "#ffffff"
    name_c  = "#c8870a" if is_winner else "#343a40"
    prefix  = "🏆 " if is_winner else ""

    thumb_html = (
        f'<img src="{thumb}" style="width:100%;max-height:90px;'
        f'object-fit:cover;border-radius:7px;margin-bottom:7px;opacity:0.85;" />'
        if thumb else ""
    )

    html = (
        f'<div style="border:{border};border-radius:13px;padding:12px;'
        f'box-shadow:{shadow};background:{bg};margin-bottom:4px;">'
        f'{thumb_html}'
        f'<div style="text-align:center;font-size:15px;font-weight:700;'
        f'color:{name_c};margin-bottom:8px;">{prefix}{name}</div>'
        f'<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
    )

    if ncols == 5:
        html += "<tr>"
        for letter in "BINGO":
            html += (
                f'<th style="background:#1a1a2e;color:#ffd700;padding:8px 2px;'
                f'text-align:center;font-size:15px;font-weight:bold;'
                f'border:1px solid #333;">{letter}</th>'
            )
        html += "</tr>"

    for r in range(nrows):
        html += "<tr>"
        for c in range(ncols):
            val     = card.iloc[r, c]
            free    = rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]
            marked  = free or str(val).strip().upper() in called_set
            display = "★" if free else str(val)

            if is_winner and marked:
                cbg, fg, fw = "#ffd700", "#1a1a2e", "bold"
            elif marked:
                cbg, fg, fw = "#28a745", "#ffffff", "bold"
            else:
                cbg, fg, fw = "#f8f9fa", "#495057", "normal"

            html += (
                f'<td style="background:{cbg};color:{fg};font-weight:{fw};'
                f'padding:10px 2px;text-align:center;border:1px solid #dee2e6;'
                f'font-size:14px;border-radius:3px;">{display}</td>'
            )
        html += "</tr>"

    html += "</table></div>"
    return html


# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Rules image ──────────────────────────────────────────────────────────
    st.markdown("### 📜 Rules Image")
    st.caption("Upload the rules image for reference, then configure below.")
    rules_img = st.file_uploader(
        "Rules image",
        type=["png", "jpg", "jpeg", "webp"],
        key="rules_img",
        label_visibility="collapsed",
    )
    if rules_img:
        st.image(rules_img, use_container_width=True)

    st.markdown("### 📋 Winning Conditions")
    rules = st.session_state.rules
    prev  = dict(rules)

    rules["check_rows"]      = st.checkbox("✅ Complete Row",         value=rules["check_rows"])
    rules["check_cols"]      = st.checkbox("✅ Complete Column",      value=rules["check_cols"])
    rules["check_diagonals"] = st.checkbox("✅ Diagonal",             value=rules["check_diagonals"])
    rules["check_full_card"] = st.checkbox("✅ Full Card (Blackout)", value=rules["check_full_card"])
    rules["free_space"]      = st.checkbox("⭐ Free Space (center)",  value=rules["free_space"])

    if any(rules[k] != prev[k] for k in prev):
        recalc_winners()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔁 Round Management")

    if st.button("🔄 New Round — Keep Cards", use_container_width=True):
        st.session_state.called_values  = []
        st.session_state.winners        = set()
        st.session_state.last_warning   = None
        st.session_state.round         += 1
        st.rerun()

    if st.button("🆕 New Round — Upload New Cards", use_container_width=True):
        st.session_state.called_values  = []
        st.session_state.winners        = set()
        st.session_state.cards          = []
        st.session_state.card_names     = []
        st.session_state.card_thumbs    = []
        st.session_state.last_warning   = None
        st.session_state.round         += 1
        st.rerun()

    st.markdown("---")

    if st.session_state.called_values:
        st.markdown("### 📢 Called Values")
        chips = " ".join(f'<span class="chip">{v}</span>' for v in st.session_state.called_values)
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"Total called: {len(st.session_state.called_values)}")

        if st.button("↩️ Undo Last Call", use_container_width=True):
            st.session_state.called_values.pop()
            st.session_state.last_warning = None
            recalc_winners()
            st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bingo-header">
  <h1>🎱 Bingo Manager</h1>
  <span class="round-badge">Round {st.session_state.round}</span>
</div>
""", unsafe_allow_html=True)


# ─── Card Upload & Entry ──────────────────────────────────────────────────────
with st.expander("➕ Add a Bingo Card", expanded=not bool(st.session_state.cards)):

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── Left: image upload ────────────────────────────────────────────────────
    with left_col:
        st.markdown("#### 📷 Card Image")
        st.caption("Upload the card photo — use it as reference while filling the grid.")

        card_img = st.file_uploader(
            "Card image",
            type=["png", "jpg", "jpeg", "webp"],
            key="card_img_upload",
            label_visibility="collapsed",
        )

        if card_img:
            st.image(card_img, use_container_width=True, caption=card_img.name)
        else:
            st.markdown(
                '<div style="border:2px dashed #ccc;border-radius:10px;padding:40px;'
                'text-align:center;color:#999;font-size:14px;">📷 Card image will appear here</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### ⚙️ Grid Options")
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            grid_size = st.selectbox("Grid size", [5, 4, 3, 6], index=0, key="entry_grid_size_sel")
        with cfg_col2:
            card_name_input = st.text_input(
                "Card name",
                value=f"Card {len(st.session_state.cards) + 1}",
                key="card_name_input",
            )

        has_free = st.checkbox("⭐ Has free space", value=True, key="entry_has_free_cb")
        if has_free:
            fc1, fc2 = st.columns(2)
            with fc1:
                free_row = st.number_input("Free row (0-indexed)", 0, grid_size - 1,
                                           value=min(2, grid_size - 1), key="entry_free_row_sel")
            with fc2:
                free_col = st.number_input("Free col (0-indexed)", 0, grid_size - 1,
                                           value=min(2, grid_size - 1), key="entry_free_col_sel")
        else:
            free_row, free_col = -1, -1

    # ── Right: grid entry form ─────────────────────────────────────────────────
    with right_col:
        st.markdown("#### ✏️ Fill in the Grid")
        st.caption("Type each value exactly as it appears on the card.")

        if grid_size == 5:
            col_labels = list("BINGO")
        else:
            col_labels = [str(i + 1) for i in range(grid_size)]

        # Column headers
        hdr_cols = st.columns(grid_size)
        for ci, lbl in enumerate(col_labels):
            with hdr_cols[ci]:
                st.markdown(
                    f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                    f'font-weight:bold;font-size:16px;padding:6px;border-radius:5px;">{lbl}</div>',
                    unsafe_allow_html=True,
                )

        # Input grid
        grid_values = []
        for r in range(grid_size):
            row_cols = st.columns(grid_size)
            row_vals = []
            for c in range(grid_size):
                with row_cols[c]:
                    is_free_cell = has_free and r == free_row and c == free_col
                    if is_free_cell:
                        st.markdown(
                            '<div style="background:#ffd700;color:#1a1a2e;text-align:center;'
                            'font-weight:bold;font-size:18px;padding:8px;border-radius:5px;'
                            'border:2px solid #ccc;">★</div>',
                            unsafe_allow_html=True,
                        )
                        row_vals.append("FREE")
                    else:
                        val = st.text_input(
                            f"r{r}c{c}",
                            key=f"cell_{r}_{c}_{len(st.session_state.cards)}",
                            label_visibility="collapsed",
                            placeholder="—",
                        )
                        row_vals.append(val.strip() if val.strip() else f"?{r}{c}")
            grid_values.append(row_vals)

        st.markdown("")
        if st.button("✅ Add This Card", type="primary", use_container_width=True):
            # Build DataFrame
            df = pd.DataFrame(grid_values).astype(str)

            # Build thumbnail data URL
            thumb = ""
            if card_img:
                import base64
                card_img.seek(0)
                raw = card_img.read()
                name_lower = card_img.name.lower()
                mt = ("image/png" if name_lower.endswith(".png")
                      else "image/webp" if name_lower.endswith(".webp")
                      else "image/jpeg")
                b64 = base64.standard_b64encode(raw).decode("utf-8")
                thumb = f"data:{mt};base64,{b64}"

            st.session_state.cards.append(df)
            st.session_state.card_names.append(card_name_input or f"Card {len(st.session_state.cards)}")
            st.session_state.card_thumbs.append(thumb)

            # Sync free space to rules
            if has_free:
                st.session_state.rules["free_space"]     = True
                st.session_state.rules["free_space_row"] = free_row
                st.session_state.rules["free_space_col"] = free_col
            else:
                st.session_state.rules["free_space"] = False

            recalc_winners()
            st.success(f"✅ '{card_name_input}' added!")
            st.rerun()


# ─── Game Area ────────────────────────────────────────────────────────────────
if not st.session_state.cards:
    st.info(
        "👆 Use the section above to add your first card.\n\n"
        "Upload the card image as a reference, fill in the grid values, and click **Add This Card**."
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
for win_idx in sorted(st.session_state.winners):
    st.markdown(
        f'<div class="winner-banner">'
        f'🎉 BINGO! &nbsp;&nbsp; {st.session_state.card_names[win_idx]} &nbsp;&nbsp; 🎉'
        f'</div>',
        unsafe_allow_html=True,
    )

# — Called values strip —
if st.session_state.called_values:
    chips = " ".join(f'<span class="chip">{v}</span>' for v in st.session_state.called_values)
    st.markdown(
        f"<div style='margin:10px 0;'>"
        f"<strong>Called ({len(st.session_state.called_values)}):</strong>&nbsp; {chips}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# — Cards grid —
n = len(st.session_state.cards)
st.markdown(
    f"### 🃏 Cards &nbsp;<span style='color:#6c757d;font-size:16px;'>({n})</span>",
    unsafe_allow_html=True,
)

num_cols  = min(n, 3)
grid_cols = st.columns(num_cols, gap="medium")

for i in range(n):
    with grid_cols[i % num_cols]:
        st.markdown(render_card_html(i), unsafe_allow_html=True)
        if st.button("🗑️ Remove", key=f"rm_{i}_{st.session_state.round}", use_container_width=True):
            for lst in ("cards", "card_names", "card_thumbs"):
                st.session_state[lst].pop(i)
            recalc_winners()
            st.rerun()
