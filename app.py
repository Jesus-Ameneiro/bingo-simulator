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

    .bingo-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 22px 30px; border-radius: 14px;
        text-align: center; margin-bottom: 18px;
    }
    .bingo-header h1  { color: white; margin: 0; font-size: 36px; letter-spacing: 2px; }
    .round-badge {
        background: #e94560; color: white;
        padding: 5px 18px; border-radius: 20px;
        font-weight: bold; font-size: 14px;
        display: inline-block; margin-top: 8px;
    }

    .winner-banner {
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
        color: #1a1a2e; padding: 16px 24px; border-radius: 12px;
        text-align: center; font-size: 20px; font-weight: bold;
        margin: 6px 0; box-shadow: 0 4px 20px rgba(255,215,0,0.55);
    }

    /* ── Green = marked cell (primary button) ── */
    button[data-testid="baseButton-primary"] {
        background-color: #28a745 !important;
        border-color:     #28a745 !important;
        color:            white   !important;
        font-weight:      bold    !important;
        font-size:        15px    !important;
        min-height:       46px    !important;
        border-radius:    6px     !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #218838 !important;
        border-color:     #1e7e34 !important;
    }

    /* ── Gray = unmarked cell (secondary button) ── */
    button[data-testid="baseButton-secondary"] {
        font-size:     15px !important;
        min-height:    46px !important;
        border-radius: 6px  !important;
        color: #343a40      !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Default card data ────────────────────────────────────────────────────────
def _default_cards():
    grids = [
        [["5","19","38","50","73"],
         ["8","20","37","56","65"],
         ["10","25","FREE","49","68"],
         ["15","18","32","53","72"],
         ["4","23","35","58","75"]],

        [["5","28","35","48","73"],
         ["10","22","32","46","66"],
         ["4","27","FREE","60","72"],
         ["15","25","38","57","64"],
         ["11","20","33","58","70"]],

        [["15","24","43","50","63"],
         ["7","25","45","49","65"],
         ["6","26","FREE","60","70"],
         ["11","29","38","54","61"],
         ["9","28","35","48","67"]],
    ]
    names  = ["Card 1478", "Card 1479", "Card 1480"]
    return [pd.DataFrame(g).astype(str) for g in grids], names

# ─── Session State ────────────────────────────────────────────────────────────
if "cards" not in st.session_state:
    _cards, _names = _default_cards()
    st.session_state.cards       = _cards
    st.session_state.card_names  = _names
    st.session_state.card_thumbs = [""] * len(_cards)

_DEFAULTS = {
    "manual_marks":  {},      # {card_idx: set of (r, c)}
    "winners":       set(),
    "round":         1,
    "round_pattern": set(),   # set of (r, c) — required cells to win
    "pattern_size":  5,
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
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Game Logic ───────────────────────────────────────────────────────────────
def is_marked(idx: int, r: int, c: int) -> bool:
    rules = st.session_state.rules
    if (rules["free_space"]
            and r == rules["free_space_row"]
            and c == rules["free_space_col"]):
        return True
    return (r, c) in st.session_state.manual_marks.get(idx, set())


def toggle_mark(idx: int, r: int, c: int):
    marks = st.session_state.manual_marks.setdefault(idx, set())
    marks.discard((r, c)) if (r, c) in marks else marks.add((r, c))


def check_bingo(idx: int) -> bool:
    card  = st.session_state.cards[idx]
    rules = st.session_state.rules
    nrows, ncols = card.shape
    m = lambda r, c: is_marked(idx, r, c)

    pattern = st.session_state.round_pattern
    if pattern:
        return all(m(r, c) for r, c in pattern)

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
        i for i in range(len(st.session_state.cards))
        if check_bingo(i)
    }


# ─── Round Pattern Card ───────────────────────────────────────────────────────
def render_pattern_card():
    rules   = st.session_state.rules
    pattern = st.session_state.round_pattern
    size    = st.session_state.pattern_size
    status  = f"Custom — {len(pattern)} cell(s) marked" if pattern else "Using sidebar rules"

    # ── Title bar ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f3460,#16213e);'
        'border-radius:10px 10px 0 0;padding:12px 18px;display:flex;'
        'align-items:center;justify-content:space-between;">'
        '<span style="color:#ffd700;font-weight:700;font-size:17px;">'
        '🎯 Round Winning Pattern</span>'
        f'<span style="background:#e94560;color:white;padding:3px 14px;'
        f'border-radius:14px;font-size:13px;font-weight:bold;">{status}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # ── Controls row ─────────────────────────────────────────────────────
        desc_col, sz_col, clr_col = st.columns([5, 2, 1], gap="small")
        with desc_col:
            st.caption(
                "Click a cell to add it to the winning pattern (✅ = required, ⬜ = not required, ★ = free space always counts)."
            )
        with sz_col:
            new_size = st.radio(
                "Grid size", [3, 4, 5], index=[3, 4, 5].index(size),
                horizontal=True, label_visibility="collapsed",
                key="pat_size_radio",
            )
            if new_size != size:
                st.session_state.pattern_size  = new_size
                st.session_state.round_pattern = set()
                recalc_winners()
                st.rerun()
        with clr_col:
            if st.button("🗑️ Clear", key="clr_pat", use_container_width=True):
                st.session_state.round_pattern = set()
                recalc_winners()
                st.rerun()

        # ── Centered grid ────────────────────────────────────────────────────
        _, grid_col, _ = st.columns([2, 1, 2])
        with grid_col:
            # BINGO column headers
            if size == 5:
                hc = st.columns(5, gap="small")
                for ci, letter in enumerate("BINGO"):
                    with hc[ci]:
                        st.markdown(
                            f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                            f'font-weight:bold;font-size:14px;padding:6px 0;'
                            f'border-radius:4px;margin-bottom:3px;">{letter}</div>',
                            unsafe_allow_html=True,
                        )

            # Clickable cells
            for r in range(size):
                rc = st.columns(size, gap="small")
                for c in range(size):
                    with rc[c]:
                        is_free = (rules["free_space"] and size == 5
                                   and r == rules["free_space_row"]
                                   and c == rules["free_space_col"])
                        if is_free:
                            st.markdown(
                                '<div style="background:#ffd700;color:#1a1a2e;text-align:center;'
                                'font-weight:bold;font-size:18px;min-height:46px;line-height:46px;'
                                'border-radius:6px;border:1px solid #ccc;">★</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            in_pat = (r, c) in pattern
                            if st.button(
                                "✅" if in_pat else "⬜",
                                key=f"pat_{r}_{c}_{st.session_state.round}",
                                type="primary" if in_pat else "secondary",
                                use_container_width=True,
                            ):
                                pat = st.session_state.round_pattern
                                pat.discard((r, c)) if (r, c) in pat else pat.add((r, c))
                                recalc_winners()
                                st.rerun()


# ─── Playing Card ─────────────────────────────────────────────────────────────
def render_card(idx: int):
    card      = st.session_state.cards[idx]
    name      = st.session_state.card_names[idx]
    thumb     = st.session_state.card_thumbs[idx]
    is_winner = idx in st.session_state.winners
    nrows, ncols = card.shape

    if is_winner:
        st.markdown(
            f'<div class="winner-banner">🏆 BINGO! &nbsp; {name} &nbsp; 🏆</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        if thumb:
            st.markdown(
                f'<img src="{thumb}" style="width:100%;max-height:80px;'
                f'object-fit:cover;border-radius:7px;margin-bottom:4px;" />',
                unsafe_allow_html=True,
            )

        name_color = "#c8870a" if is_winner else "#343a40"
        st.markdown(
            f'<div style="text-align:center;font-weight:700;font-size:15px;'
            f'color:{name_color};margin-bottom:4px;">{name}</div>',
            unsafe_allow_html=True,
        )

        # BINGO column headers
        if ncols == 5:
            hc = st.columns(5, gap="small")
            for ci, letter in enumerate("BINGO"):
                with hc[ci]:
                    st.markdown(
                        f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                        f'font-weight:bold;font-size:14px;padding:6px 0;'
                        f'border-radius:4px;margin-bottom:2px;">{letter}</div>',
                        unsafe_allow_html=True,
                    )

        # Clickable cell grid
        rules = st.session_state.rules
        for r in range(nrows):
            rc = st.columns(ncols, gap="small")
            for c in range(ncols):
                with rc[c]:
                    val     = str(card.iloc[r, c])
                    is_free = (rules["free_space"]
                               and r == rules["free_space_row"]
                               and c == rules["free_space_col"])

                    if is_free:
                        st.markdown(
                            '<div style="background:#ffd700;color:#1a1a2e;text-align:center;'
                            'font-weight:bold;font-size:18px;min-height:46px;line-height:46px;'
                            'border-radius:6px;border:1px solid #ccc;margin:1px 0;">★</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        marked = is_marked(idx, r, c)
                        if st.button(
                            val,
                            key=f"cell_{idx}_{r}_{c}_{st.session_state.round}",
                            type="primary" if marked else "secondary",
                            use_container_width=True,
                        ):
                            toggle_mark(idx, r, c)
                            recalc_winners()
                            st.rerun()

    # Remove button
    if st.button("🗑️ Remove card", key=f"rm_{idx}_{st.session_state.round}",
                 use_container_width=True):
        st.session_state.cards.pop(idx)
        st.session_state.card_names.pop(idx)
        st.session_state.card_thumbs.pop(idx)
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
        "Rules image", type=["png", "jpg", "jpeg", "webp"],
        key="rules_img_up", label_visibility="collapsed",
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
        st.session_state.manual_marks  = {}
        st.session_state.winners       = set()
        st.session_state.round_pattern = set()
        st.session_state.round        += 1
        st.rerun()

    if st.button("🔄 New Round — Reset to Default Cards", use_container_width=True):
        _cards, _names = _default_cards()
        st.session_state.cards         = _cards
        st.session_state.card_names    = _names
        st.session_state.card_thumbs   = [""] * len(_cards)
        st.session_state.manual_marks  = {}
        st.session_state.winners       = set()
        st.session_state.round_pattern = set()
        st.session_state.round        += 1
        st.rerun()

    if st.button("🆕 New Round — Clear All Cards", use_container_width=True):
        st.session_state.cards         = []
        st.session_state.card_names    = []
        st.session_state.card_thumbs   = []
        st.session_state.manual_marks  = {}
        st.session_state.winners       = set()
        st.session_state.round_pattern = set()
        st.session_state.round        += 1
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bingo-header">
  <h1>🎱 Bingo Manager</h1>
  <span class="round-badge">Round {st.session_state.round}</span>
</div>
""", unsafe_allow_html=True)


# ─── Add a Card ───────────────────────────────────────────────────────────────
with st.expander("➕ Add a New Card", expanded=False):
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("#### 📷 Card Image *(optional)*")
        card_img = st.file_uploader(
            "Card image", type=["png", "jpg", "jpeg", "webp"],
            key="new_card_img", label_visibility="collapsed",
        )
        if card_img:
            st.image(card_img, use_container_width=True)
        else:
            st.markdown(
                '<div style="border:2px dashed #ccc;border-radius:10px;padding:30px;'
                'text-align:center;color:#aaa;">📷 Optional photo reference</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### ⚙️ Options")
        c1, c2 = st.columns(2)
        with c1:
            grid_size = st.selectbox("Grid size", [5, 4, 3, 6], index=0, key="new_card_size")
        with c2:
            new_name = st.text_input(
                "Card name",
                value=f"Card {len(st.session_state.cards) + 1}",
                key="new_card_name",
            )
        has_free = st.checkbox("⭐ Has free space", value=True, key="new_has_free")
        if has_free:
            f1, f2 = st.columns(2)
            with f1:
                free_row = int(st.number_input("Free row (0-index)", 0, grid_size-1,
                                               value=min(2, grid_size-1), key="new_fr"))
            with f2:
                free_col = int(st.number_input("Free col (0-index)", 0, grid_size-1,
                                               value=min(2, grid_size-1), key="new_fc"))
        else:
            free_row, free_col = -1, -1

    with right_col:
        st.markdown("#### ✏️ Fill in the Grid")
        st.caption("Type each value exactly as printed on the card.")

        labels = list("BINGO") if grid_size == 5 else [str(i+1) for i in range(grid_size)]
        hc = st.columns(grid_size)
        for ci, lbl in enumerate(labels):
            with hc[ci]:
                st.markdown(
                    f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                    f'font-weight:bold;font-size:15px;padding:6px;border-radius:5px;">{lbl}</div>',
                    unsafe_allow_html=True,
                )

        grid_vals = []
        for r in range(grid_size):
            rc = st.columns(grid_size)
            row = []
            for c in range(grid_size):
                with rc[c]:
                    is_fc = has_free and r == free_row and c == free_col
                    if is_fc:
                        st.markdown(
                            '<div style="background:#ffd700;color:#1a1a2e;text-align:center;'
                            'font-weight:bold;font-size:18px;padding:8px;border-radius:5px;'
                            'border:2px solid #ccc;">★</div>',
                            unsafe_allow_html=True,
                        )
                        row.append("FREE")
                    else:
                        v = st.text_input(
                            f"r{r}c{c}",
                            key=f"nc_{r}_{c}_{len(st.session_state.cards)}",
                            label_visibility="collapsed",
                            placeholder="—",
                        )
                        row.append(v.strip() if v.strip() else f"?{r}{c}")
            grid_vals.append(row)

        st.markdown("")
        if st.button("✅ Add This Card", use_container_width=True, key="add_card_btn"):
            df    = pd.DataFrame(grid_vals).astype(str)
            thumb = ""
            if card_img:
                card_img.seek(0)
                raw = card_img.read()
                mt  = ("image/png" if card_img.name.lower().endswith(".png")
                       else "image/webp" if card_img.name.lower().endswith(".webp")
                       else "image/jpeg")
                thumb = "data:{};base64,{}".format(
                    mt, base64.standard_b64encode(raw).decode()
                )

            st.session_state.cards.append(df)
            st.session_state.card_names.append(new_name or f"Card {len(st.session_state.cards)}")
            st.session_state.card_thumbs.append(thumb)

            if has_free:
                st.session_state.rules["free_space"]     = True
                st.session_state.rules["free_space_row"] = free_row
                st.session_state.rules["free_space_col"] = free_col
            else:
                st.session_state.rules["free_space"] = False

            recalc_winners()
            st.success(f"✅ '{new_name}' added!")
            st.rerun()


# ─── Pattern card + Playing cards ────────────────────────────────────────────
if not st.session_state.cards:
    st.info("Use **➕ Add a New Card** above or reset to default cards from the sidebar.")
    st.stop()

# Winner banners
for wi in sorted(st.session_state.winners):
    st.markdown(
        f'<div class="winner-banner">🏆 BINGO! &nbsp;&nbsp; '
        f'{st.session_state.card_names[wi]} &nbsp;&nbsp; 🏆</div>',
        unsafe_allow_html=True,
    )

# ── Round Pattern card — full width ───────────────────────────────────────────
render_pattern_card()

st.markdown(
    '<div style="font-size:13px;color:#6c757d;margin:16px 0 8px;">'
    '💡 Click any cell on the cards below to mark or unmark it.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Playing cards — fixed 3 columns ──────────────────────────────────────────
st.markdown(
    f"### 🃏 Cards &nbsp;<span style='color:#6c757d;font-size:16px;'>"
    f"({len(st.session_state.cards)})</span>",
    unsafe_allow_html=True,
)

cols = st.columns(3, gap="large")
for i in range(len(st.session_state.cards)):
    with cols[i % 3]:
        render_card(i)
