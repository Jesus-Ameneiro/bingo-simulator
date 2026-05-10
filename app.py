import streamlit as st
import pandas as pd
import base64

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

    /* ── Header ── */
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

    /* ── Winner banner ── */
    .winner-banner {
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
        color: #1a1a2e; padding: 16px 24px; border-radius: 12px;
        text-align: center; font-size: 20px; font-weight: bold;
        margin: 6px 0; box-shadow: 0 4px 20px rgba(255,215,0,0.55);
    }

    /* ── Called value chips ── */
    .chip {
        display: inline-block; background: #28a745; color: white;
        padding: 4px 12px; border-radius: 15px;
        margin: 2px; font-size: 13px; font-weight: bold;
    }

    /* ── MARKED cell buttons — primary type → green ── */
    button[data-testid="baseButton-primary"] {
        background-color: #28a745 !important;
        border-color:     #28a745 !important;
        color:            white   !important;
        font-weight:      bold    !important;
        font-size:        14px    !important;
        min-height:       44px   !important;
        border-radius:    6px     !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #218838 !important;
        border-color:     #1e7e34 !important;
    }

    /* ── Unmarked cell buttons — secondary type ── */
    button[data-testid="baseButton-secondary"] {
        font-size:    14px !important;
        min-height:   44px !important;
        border-radius: 6px !important;
        color: #343a40 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "cards":         [],   # list[pd.DataFrame]
    "card_names":    [],   # list[str]
    "card_thumbs":   [],   # list[str]  – base64 data-URLs
    "called_values": [],
    "manual_marks":  {},   # {card_idx: set of (r, c)}
    "winners":       set(),
    "round":         1,
    "last_warning":  None,
    "rules": {
        "check_rows":      True,
        "check_cols":      True,
        "check_diagonals": True,
        "check_full_card": False,
        "free_space":      True,
        "free_space_row":  2,
        "free_space_col":  2,
    },
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Matching helper ──────────────────────────────────────────────────────────
_BINGO = "BINGO"

def _value_matches(cell_val: str, c: int, ncols: int, called_set: set) -> bool:
    """True if cell_val is covered by any entry in called_set.
    Handles BINGO-prefix bridging (called 'B8' matches cell '8' in col 0, etc.)
    """
    v = cell_val.upper()
    if v in called_set:
        return True
    if ncols == 5 and c < 5:
        letter = _BINGO[c]
        if f"{letter}{v}" in called_set:          # cell "8"  vs called "B8"
            return True
        if v.startswith(letter) and v[1:] in called_set:  # cell "B8" vs called "8"
            return True
    # Generic: strip any single leading letter from called values
    for cv in called_set:
        if len(cv) >= 2 and cv[0].isalpha() and cv[1:] == v:
            return True
    return False

# ─── Game Logic ───────────────────────────────────────────────────────────────
def is_cell_marked(idx: int, card_df, r: int, c: int, called_set: set, rules: dict) -> bool:
    if rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]:
        return True
    cell_val = str(card_df.iloc[r, c]).strip()
    if _value_matches(cell_val, c, card_df.shape[1], called_set):
        return True
    return (r, c) in st.session_state.manual_marks.get(idx, set())


def toggle_mark(idx: int, r: int, c: int):
    marks = st.session_state.manual_marks.setdefault(idx, set())
    if (r, c) in marks:
        marks.discard((r, c))
    else:
        marks.add((r, c))


def check_bingo(card_df, idx: int, called_values, rules) -> bool:
    called_set = {str(v).strip().upper() for v in called_values}
    nrows, ncols = card_df.shape
    m = lambda r, c: is_cell_marked(idx, card_df, r, c, called_set, rules)

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
        if check_bingo(card, i, st.session_state.called_values, st.session_state.rules)
    }

# ─── Card Rendering — fully interactive button grid ───────────────────────────
def render_card(idx: int):
    card      = st.session_state.cards[idx]
    name      = st.session_state.card_names[idx]
    thumb     = st.session_state.card_thumbs[idx]
    rules     = st.session_state.rules
    is_winner = idx in st.session_state.winners
    called_set = {str(v).strip().upper() for v in st.session_state.called_values}
    nrows, ncols = card.shape

    # Winner banner above the card
    if is_winner:
        st.markdown(
            f'<div class="winner-banner">🏆 BINGO! &nbsp; {name} &nbsp; 🏆</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        # Thumbnail
        if thumb:
            st.markdown(
                f'<img src="{thumb}" style="width:100%;max-height:82px;'
                f'object-fit:cover;border-radius:7px;margin-bottom:4px;" />',
                unsafe_allow_html=True,
            )

        # Card name
        name_color = "#c8870a" if is_winner else "#343a40"
        st.markdown(
            f'<div style="text-align:center;font-weight:700;font-size:15px;'
            f'color:{name_color};margin-bottom:4px;">{name}</div>',
            unsafe_allow_html=True,
        )

        # BINGO column header
        if ncols == 5:
            hcols = st.columns(5, gap="small")
            for ci, letter in enumerate("BINGO"):
                with hcols[ci]:
                    st.markdown(
                        f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                        f'font-weight:bold;font-size:14px;padding:6px 0;'
                        f'border-radius:4px;margin-bottom:2px;">{letter}</div>',
                        unsafe_allow_html=True,
                    )

        # Cell grid — each cell is a clickable button
        for r in range(nrows):
            rcols = st.columns(ncols, gap="small")
            for c in range(ncols):
                with rcols[c]:
                    val     = str(card.iloc[r, c])
                    is_free = (rules["free_space"]
                               and r == rules["free_space_row"]
                               and c == rules["free_space_col"])

                    if is_free:
                        # Static gold star — no button needed
                        st.markdown(
                            '<div style="background:#ffd700;color:#1a1a2e;text-align:center;'
                            'font-weight:bold;font-size:18px;min-height:44px;line-height:44px;'
                            'border-radius:6px;border:1px solid #ccc;">★</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        marked     = is_cell_marked(idx, card, r, c, called_set, rules)
                        btn_type   = "primary" if marked else "secondary"
                        btn_label  = val  # always show the value

                        if st.button(
                            btn_label,
                            key=f"cell_{idx}_{r}_{c}_{st.session_state.round}",
                            type=btn_type,
                            use_container_width=True,
                        ):
                            toggle_mark(idx, r, c)
                            recalc_winners()
                            st.rerun()

    # Remove card
    if st.button("🗑️ Remove card", key=f"rm_{idx}_{st.session_state.round}",
                 use_container_width=True):
        for lst in ("cards", "card_names", "card_thumbs"):
            st.session_state[lst].pop(idx)
        st.session_state.manual_marks.pop(idx, None)
        recalc_winners()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 📜 Rules Image")
    st.caption("Upload for visual reference, then configure below.")
    rules_img = st.file_uploader(
        "Rules image", type=["png","jpg","jpeg","webp"],
        key="rules_img", label_visibility="collapsed",
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
        st.session_state.manual_marks   = {}
        st.session_state.winners        = set()
        st.session_state.last_warning   = None
        st.session_state.round         += 1
        st.rerun()

    if st.button("🆕 New Round — Upload New Cards", use_container_width=True):
        st.session_state.called_values  = []
        st.session_state.manual_marks   = {}
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

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bingo-header">
  <h1>🎱 Bingo Manager</h1>
  <span class="round-badge">Round {st.session_state.round}</span>
</div>
""", unsafe_allow_html=True)

# ─── Add Card ─────────────────────────────────────────────────────────────────
with st.expander("➕ Add a Bingo Card", expanded=not bool(st.session_state.cards)):
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("#### 📷 Card Image")
        st.caption("Upload the card photo — use it as reference while filling the grid.")

        card_img = st.file_uploader(
            "Card image", type=["png","jpg","jpeg","webp"],
            key="card_img_upload", label_visibility="collapsed",
        )
        if card_img:
            st.image(card_img, use_container_width=True, caption=card_img.name)
        else:
            st.markdown(
                '<div style="border:2px dashed #ccc;border-radius:10px;padding:40px;'
                'text-align:center;color:#999;">📷 Card image will appear here</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### ⚙️ Options")
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            grid_size = st.selectbox("Grid size", [5, 4, 3, 6], index=0)
        with cfg2:
            card_name_input = st.text_input(
                "Card name",
                value=f"Card {len(st.session_state.cards) + 1}",
            )

        has_free = st.checkbox("⭐ Has free space", value=True)
        if has_free:
            fc1, fc2 = st.columns(2)
            with fc1:
                free_row = st.number_input(
                    "Free row (0-indexed)", 0, grid_size-1,
                    value=min(2, grid_size-1),
                )
            with fc2:
                free_col = st.number_input(
                    "Free col (0-indexed)", 0, grid_size-1,
                    value=min(2, grid_size-1),
                )
        else:
            free_row, free_col = -1, -1

    with right_col:
        st.markdown("#### ✏️ Fill in the Grid")
        st.caption("Type each value exactly as printed on the card.")

        col_labels = list("BINGO") if grid_size == 5 else [str(i+1) for i in range(grid_size)]
        hdr_cols = st.columns(grid_size)
        for ci, lbl in enumerate(col_labels):
            with hdr_cols[ci]:
                st.markdown(
                    f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                    f'font-weight:bold;font-size:15px;padding:6px;border-radius:5px;">{lbl}</div>',
                    unsafe_allow_html=True,
                )

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
                            key=f"entry_{r}_{c}_{len(st.session_state.cards)}",
                            label_visibility="collapsed",
                            placeholder="—",
                        )
                        row_vals.append(val.strip() if val.strip() else f"?{r}{c}")
            grid_values.append(row_vals)

        st.markdown("")
        if st.button("✅ Add This Card", use_container_width=True):
            df = pd.DataFrame(grid_values).astype(str)

            thumb = ""
            if card_img:
                card_img.seek(0)
                raw  = card_img.read()
                name = card_img.name.lower()
                mt   = ("image/png"  if name.endswith(".png")
                        else "image/webp" if name.endswith(".webp")
                        else "image/jpeg")
                b64  = base64.standard_b64encode(raw).decode("utf-8")
                thumb = f"data:{mt};base64,{b64}"

            st.session_state.cards.append(df)
            st.session_state.card_names.append(card_name_input or f"Card {len(st.session_state.cards)}")
            st.session_state.card_thumbs.append(thumb)

            if has_free:
                st.session_state.rules["free_space"]     = True
                st.session_state.rules["free_space_row"] = int(free_row)
                st.session_state.rules["free_space_col"] = int(free_col)
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

# — Called values strip —
if st.session_state.called_values:
    chips = " ".join(
        f'<span class="chip">{v}</span>'
        for v in st.session_state.called_values
    )
    st.markdown(
        f"<div style='margin:10px 0;'>"
        f"<strong>Called ({len(st.session_state.called_values)}):</strong>&nbsp; {chips}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='margin:6px 0 4px;font-size:13px;color:#6c757d;'>"
    "💡 Click any cell to manually mark or unmark it."
    "</div>",
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
outer_cols = st.columns(num_cols, gap="large")

for i in range(n):
    with outer_cols[i % num_cols]:
        render_card(i)
