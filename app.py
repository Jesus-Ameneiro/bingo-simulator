import streamlit as st
import pandas as pd
import base64
import numpy as np
from PIL import Image
import io

# ── OCR imports (graceful fallback if not installed) ──────────────────────────
try:
    import cv2
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

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
        font-size:        14px    !important;
        min-height:       46px    !important;
        border-radius:    6px     !important;
        padding:          4px 0px !important;
        width:            100%    !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #218838 !important;
        border-color:     #1e7e34 !important;
    }
    button[data-testid="baseButton-primary"] p {
        white-space: nowrap   !important;
        overflow:    visible  !important;
        font-size:   14px     !important;
        margin:      0        !important;
        line-height: 1.2      !important;
    }

    /* ── Gray = unmarked cell (secondary button) ── */
    button[data-testid="baseButton-secondary"] {
        font-size:     14px    !important;
        min-height:    46px    !important;
        border-radius: 6px     !important;
        color:         #343a40 !important;
        padding:       4px 0px !important;
        width:         100%    !important;
    }
    button[data-testid="baseButton-secondary"] p {
        white-space: nowrap   !important;
        overflow:    visible  !important;
        font-size:   14px     !important;
        margin:      0        !important;
        line-height: 1.2      !important;
    }

    /* ── Squeeze column gaps inside card containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"]
        div[data-testid="stHorizontalBlock"] {
        gap: 4px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]
        div[data-testid="stHorizontalBlock"]
        div[data-testid="column"] {
        padding-left:  1px !important;
        padding-right: 1px !important;
        min-width:     0   !important;
        overflow:      visible !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── OCR Engine ───────────────────────────────────────────────────────────────
def _ocr_cell(cell_img: np.ndarray) -> str:
    """Run tesseract on a single cropped cell. Returns cleaned number string."""
    # Upscale small cells for better OCR accuracy
    h, w = cell_img.shape[:2]
    if h < 60:
        cell_img = cv2.resize(cell_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # Grayscale + denoise
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold — white text on dark or dark text on white both handled
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    # Tesseract: digits only, single-line mode
    cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(thresh, config=cfg).strip()
    raw = "".join(ch for ch in raw if ch.isdigit())
    return raw if raw else "?"


def scan_card_from_image(pil_img: Image.Image, grid_size: int = 5) -> list[list[str]]:
    """
    Extract bingo card grid from a PIL image using OpenCV + tesseract.
    Strategy:
      1. Find the largest white/light rectangular region (the card body).
      2. Detect the BINGO header row and crop it out.
      3. Divide remaining area evenly into grid_size × grid_size cells.
      4. OCR each cell; auto-detect the centre free space.
    Returns a list[list[str]] grid.
    """
    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # ── Step 1: isolate the card grid area ───────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Find the white grid body by looking for the largest white rectangle
    # Threshold for light pixels (card background is white/near-white)
    _, light_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest contour that is reasonably square
    best_rect = None
    best_area = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / ch if ch > 0 else 0
        if area > best_area and 0.5 < aspect < 2.0 and area > (w * h * 0.1):
            best_area = area
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        # Fallback: use full image with small margin
        margin = int(min(h, w) * 0.05)
        best_rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    gx, gy, gw, gh = best_rect
    card_crop = img_bgr[gy: gy + gh, gx: gx + gw]
    ch2, cw2 = card_crop.shape[:2]

    # ── Step 2: remove BINGO header row (top ~15% is usually the header) ─────
    # Find the first horizontal band that contains mostly white (grid starts)
    gray_card = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
    row_brightness = gray_card.mean(axis=1)  # mean brightness per row
    # Header band ends where brightness drops (darker grid lines appear)
    header_end = 0
    threshold_brightness = row_brightness.max() * 0.7
    for i, b in enumerate(row_brightness):
        if i > ch2 * 0.05 and b < threshold_brightness:
            header_end = i
            break
    if header_end < ch2 * 0.05:
        header_end = int(ch2 * 0.18)  # safe default

    grid_crop = card_crop[header_end:, :]
    gh2, gw2 = grid_crop.shape[:2]

    # ── Step 3: divide into grid_size × grid_size cells ──────────────────────
    cell_h = gh2 // grid_size
    cell_w = gw2 // grid_size
    padding = max(4, int(min(cell_h, cell_w) * 0.08))  # slight inner padding

    grid: list[list[str]] = []
    mid = grid_size // 2

    for r in range(grid_size):
        row: list[str] = []
        for c in range(grid_size):
            y1 = r * cell_h + padding
            y2 = (r + 1) * cell_h - padding
            x1 = c * cell_w + padding
            x2 = (c + 1) * cell_w - padding
            cell = grid_crop[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

            # Centre cell → FREE space
            if r == mid and c == mid:
                row.append("FREE")
                continue

            val = _ocr_cell(cell) if cell.size > 0 else "?"
            row.append(val)
        grid.append(row)

    return grid


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
    "manual_marks":   {},
    "winners":        set(),
    "round":          1,
    "round_pattern":  set(),
    "pattern_size":   5,
    "ocr_grid":       None,   # list[list[str]] pending review
    "ocr_thumb":      "",     # base64 data-URL of scanned image
    "ocr_name":       "",
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


def toggle_mark_global(source_idx: int, r: int, c: int):
    """
    Toggle the clicked cell, then find the same value in every other card
    and apply the same mark/unmark action — keeping all cards in sync.
    """
    source_card = st.session_state.cards[source_idx]
    cell_val    = str(source_card.iloc[r, c]).strip().upper()

    # Decide direction: are we marking or unmarking?
    source_marks = st.session_state.manual_marks.setdefault(source_idx, set())
    marking = (r, c) not in source_marks   # True → mark, False → unmark

    # Apply to the source cell
    source_marks.add((r, c)) if marking else source_marks.discard((r, c))

    # Skip syncing FREE / empty placeholder values
    if cell_val in ("FREE", "") or cell_val.startswith("?"):
        return

    # Propagate to every other card
    for other_idx, other_card in enumerate(st.session_state.cards):
        if other_idx == source_idx:
            continue
        other_marks = st.session_state.manual_marks.setdefault(other_idx, set())
        nrows, ncols = other_card.shape
        for rr in range(nrows):
            for cc in range(ncols):
                if str(other_card.iloc[rr, cc]).strip().upper() == cell_val:
                    other_marks.add((rr, cc)) if marking else other_marks.discard((rr, cc))


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
def _pattern_presets(size: int) -> dict:
    """Return common bingo winning patterns for the given grid size."""
    mid = size // 2
    p = {}
    p["Row 1 (top)"]    = {(0, c) for c in range(size)}
    if size > 2:
        p[f"Row {mid + 1} (mid)"] = {(mid, c) for c in range(size)}
    p[f"Row {size} (bot)"] = {(size - 1, c) for c in range(size)}
    p["Col 1 (left)"]   = {(r, 0) for r in range(size)}
    if size > 2:
        p[f"Col {mid + 1} (mid)"] = {(r, mid) for r in range(size)}
    p[f"Col {size} (right)"] = {(r, size - 1) for r in range(size)}
    p["Diagonal ↘"]     = {(i, i) for i in range(size)}
    p["Diagonal ↗"]     = {(i, size - 1 - i) for i in range(size)}
    p["X Shape"]        = {(i, i) for i in range(size)} | {(i, size - 1 - i) for i in range(size)}
    p["Blackout"]       = {(r, c) for r in range(size) for c in range(size)}
    return p


def render_pattern_card():
    rules   = st.session_state.rules
    pattern = st.session_state.round_pattern
    size    = st.session_state.pattern_size
    n_marked = len(pattern)
    status   = f"Custom — {n_marked} cell(s) required" if pattern else "No pattern — click cells below"

    # ── Title bar ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f3460,#16213e);'
        'border-radius:10px 10px 0 0;padding:14px 20px;display:flex;'
        'align-items:center;justify-content:space-between;">'
        '<span style="color:#ffd700;font-weight:700;font-size:18px;letter-spacing:0.5px;">'
        '🎯 Round Winning Pattern</span>'
        f'<span style="background:{"#28a745" if pattern else "#555"};color:white;'
        f'padding:4px 16px;border-radius:14px;font-size:13px;font-weight:600;">'
        f'{status}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        # ── 3-panel layout: Controls | Grid | Presets ─────────────────────────
        ctrl_col, grid_col, preset_col = st.columns([1, 1.4, 1.6], gap="large")

        # ── LEFT: controls ─────────────────────────────────────────────────────
        with ctrl_col:
            st.markdown(
                '<p style="color:#aaa;font-size:13px;margin-bottom:16px;">'
                'Click cells on the grid to define which positions a card must '
                'complete in order to win this round.</p>',
                unsafe_allow_html=True,
            )

            st.markdown("**Grid size**")
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

            st.markdown("")
            if st.button("🗑️ Clear Pattern", key="clr_pat", use_container_width=True):
                st.session_state.round_pattern = set()
                recalc_winners()
                st.rerun()

            st.markdown(
                '<div style="margin-top:20px;padding:10px;background:#111;'
                'border-radius:8px;font-size:12px;color:#888;line-height:1.7;">'
                '✅ Required cell<br>'
                '⬜ Not required<br>'
                '★ Free space (always counts)'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── CENTER: interactive grid ────────────────────────────────────────────
        with grid_col:
            # BINGO column headers
            labels = list("BINGO") if size == 5 else [str(i + 1) for i in range(size)]
            hc = st.columns(size, gap="small")
            for ci, lbl in enumerate(labels):
                with hc[ci]:
                    st.markdown(
                        f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                        f'font-weight:bold;font-size:16px;padding:8px 0;'
                        f'border-radius:6px;margin-bottom:4px;">{lbl}</div>',
                        unsafe_allow_html=True,
                    )

            for r in range(size):
                rc = st.columns(size, gap="small")
                for c in range(size):
                    with rc[c]:
                        is_free = (rules["free_space"] and size == 5
                                   and r == rules["free_space_row"]
                                   and c == rules["free_space_col"])
                        if is_free:
                            st.markdown(
                                '<div style="background:#ffd700;color:#1a1a2e;'
                                'text-align:center;font-weight:bold;font-size:20px;'
                                'min-height:50px;line-height:50px;border-radius:8px;'
                                'border:2px solid #e6c200;margin:1px 0;">★</div>',
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

        # ── RIGHT: quick presets ────────────────────────────────────────────────
        with preset_col:
            st.markdown("**⚡ Quick Patterns**")
            st.caption("Click a preset to load it instantly.")

            presets = _pattern_presets(size)
            preset_names = list(presets.keys())

            # Display in 2 columns
            pc1, pc2 = st.columns(2, gap="small")
            for i, name in enumerate(preset_names):
                target_col = pc1 if i % 2 == 0 else pc2
                with target_col:
                    is_active = presets[name] == pattern
                    if st.button(
                        f"{'✓ ' if is_active else ''}{name}",
                        key=f"preset_{name}_{st.session_state.round}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.round_pattern = set(presets[name])
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
                            toggle_mark_global(idx, r, c)
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

    tab_scan, tab_manual = st.tabs(["📷 Scan from Image", "✏️ Enter Manually"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — SCAN FROM IMAGE
    # ════════════════════════════════════════════════════════════════════════
    with tab_scan:
        if not _OCR_AVAILABLE:
            st.warning(
                "⚠️ OCR libraries not installed. "
                "Make sure `packages.txt` contains `tesseract-ocr` and "
                "`requirements.txt` contains `pytesseract`, `opencv-python-headless`, "
                "`Pillow`, and `numpy`, then redeploy."
            )
        else:
            scan_img_col, scan_ctrl_col = st.columns([1, 1], gap="large")

            with scan_img_col:
                st.markdown("#### 📷 Upload Card Image")
                st.caption("Upload a clear photo or screenshot of the bingo card.")
                scan_file = st.file_uploader(
                    "Card image", type=["png", "jpg", "jpeg", "webp"],
                    key="scan_img_upload", label_visibility="collapsed",
                )
                if scan_file:
                    st.image(scan_file, use_container_width=True)

            with scan_ctrl_col:
                st.markdown("#### ⚙️ Scan Options")
                scan_size = st.selectbox("Grid size", [5, 4, 3, 6], index=0, key="scan_size")
                scan_name = st.text_input(
                    "Card name",
                    value=f"Card {len(st.session_state.cards) + 1}",
                    key="scan_name_input",
                )
                st.markdown("")

                if scan_file and st.button(
                    "🔍 Scan Card", type="primary",
                    use_container_width=True, key="do_scan",
                ):
                    with st.spinner("Reading card…"):
                        try:
                            scan_file.seek(0)
                            pil_img = Image.open(scan_file)
                            grid    = scan_card_from_image(pil_img, scan_size)

                            # Build thumbnail data-URL
                            scan_file.seek(0)
                            raw = scan_file.read()
                            mt  = ("image/png" if scan_file.name.lower().endswith(".png")
                                   else "image/webp" if scan_file.name.lower().endswith(".webp")
                                   else "image/jpeg")
                            thumb = "data:{};base64,{}".format(
                                mt, base64.standard_b64encode(raw).decode()
                            )
                            st.session_state.ocr_grid  = grid
                            st.session_state.ocr_thumb = thumb
                            st.session_state.ocr_name  = scan_name
                        except Exception as e:
                            st.error(f"Scan failed: {e}")

            # ── Review & edit scanned result ─────────────────────────────────
            if st.session_state.ocr_grid is not None:
                st.divider()
                st.markdown(
                    "#### ✅ Review Scanned Values"
                )
                st.caption(
                    "The grid below was read from your image. "
                    "Fix any errors before adding the card."
                )

                raw_grid = st.session_state.ocr_grid
                size     = len(raw_grid)
                mid      = size // 2

                # BINGO column headers
                labels = list("BINGO") if size == 5 else [str(i+1) for i in range(size)]
                hc = st.columns(size)
                for ci, lbl in enumerate(labels):
                    with hc[ci]:
                        st.markdown(
                            f'<div style="background:#1a1a2e;color:#ffd700;text-align:center;'
                            f'font-weight:bold;font-size:15px;padding:6px;border-radius:5px;">'
                            f'{lbl}</div>',
                            unsafe_allow_html=True,
                        )

                edited_grid = []
                for r in range(size):
                    rc   = st.columns(size)
                    row  = []
                    for c in range(size):
                        with rc[c]:
                            is_free = (r == mid and c == mid)
                            if is_free:
                                st.markdown(
                                    '<div style="background:#ffd700;color:#1a1a2e;'
                                    'text-align:center;font-weight:bold;font-size:18px;'
                                    'padding:8px;border-radius:5px;border:2px solid #ccc;">'
                                    '★</div>',
                                    unsafe_allow_html=True,
                                )
                                row.append("FREE")
                            else:
                                cur = raw_grid[r][c] if raw_grid[r][c] != "?" else ""
                                # Highlight uncertain reads in red
                                if raw_grid[r][c] in ("?", ""):
                                    st.markdown(
                                        '<div style="background:#5c1010;height:4px;'
                                        'border-radius:2px;margin-bottom:2px;"></div>',
                                        unsafe_allow_html=True,
                                    )
                                v = st.text_input(
                                    f"sr{r}c{c}",
                                    value=cur,
                                    key=f"scan_cell_{r}_{c}",
                                    label_visibility="collapsed",
                                    placeholder="?",
                                )
                                row.append(v.strip() if v.strip() else f"?{r}{c}")
                    edited_grid.append(row)

                bad_cells = sum(
                    1 for r in range(size) for c in range(size)
                    if edited_grid[r][c].startswith("?") and not (r == mid and c == mid)
                )
                if bad_cells:
                    st.warning(
                        f"⚠️ {bad_cells} cell(s) could not be read — shown with a red bar. "
                        "Please fill them in manually above."
                    )

                confirm_col, discard_col = st.columns([2, 1])
                with confirm_col:
                    if st.button(
                        "✅ Add This Card", type="primary",
                        use_container_width=True, key="confirm_scan",
                    ):
                        df = pd.DataFrame(edited_grid).astype(str)
                        st.session_state.cards.append(df)
                        st.session_state.card_names.append(
                            st.session_state.ocr_name or f"Card {len(st.session_state.cards)}"
                        )
                        st.session_state.card_thumbs.append(st.session_state.ocr_thumb)
                        st.session_state.ocr_grid  = None
                        st.session_state.ocr_thumb = ""
                        st.session_state.ocr_name  = ""
                        recalc_winners()
                        st.success("Card added!")
                        st.rerun()
                with discard_col:
                    if st.button("🗑️ Discard Scan", use_container_width=True, key="discard_scan"):
                        st.session_state.ocr_grid = None
                        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — MANUAL ENTRY
    # ════════════════════════════════════════════════════════════════════════
    with tab_manual:
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
                    '<div style="border:2px dashed #444;border-radius:10px;padding:30px;'
                    'text-align:center;color:#666;">📷 Optional photo reference</div>',
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
                    free_row = int(st.number_input(
                        "Free row (0-index)", 0, grid_size-1,
                        value=min(2, grid_size-1), key="new_fr",
                    ))
                with f2:
                    free_col = int(st.number_input(
                        "Free col (0-index)", 0, grid_size-1,
                        value=min(2, grid_size-1), key="new_fc",
                    ))
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
                        f'font-weight:bold;font-size:15px;padding:6px;border-radius:5px;">'
                        f'{lbl}</div>',
                        unsafe_allow_html=True,
                    )

            grid_vals = []
            for r in range(grid_size):
                rc  = st.columns(grid_size)
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
                st.session_state.card_names.append(
                    new_name or f"Card {len(st.session_state.cards)}"
                )
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
    '💡 Click any cell on the cards below to mark or unmark it. '
    'The same value will be marked automatically on all other cards.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Globally marked values strip ──────────────────────────────────────────────
_BALL_COLORS = {
    "B": "linear-gradient(145deg,#1a6fc4,#0d47a1)",
    "I": "linear-gradient(145deg,#e53935,#b71c1c)",
    "N": "linear-gradient(145deg,#8e24aa,#4a148c)",
    "G": "linear-gradient(145deg,#2e7d32,#1b5e20)",
    "O": "linear-gradient(145deg,#ef6c00,#e65100)",
    "":  "linear-gradient(145deg,#546e7a,#263238)",
}

def _bingo_letter(val: str) -> str:
    try:
        n = int(val)
        if  1 <= n <= 15: return "B"
        if 16 <= n <= 30: return "I"
        if 31 <= n <= 45: return "N"
        if 46 <= n <= 60: return "G"
        if 61 <= n <= 75: return "O"
    except ValueError:
        pass
    return ""

def _ball_html(val: str) -> str:
    letter   = _bingo_letter(val)
    gradient = _BALL_COLORS.get(letter, _BALL_COLORS[""])
    num_size = "13px" if len(val) >= 3 else "16px"
    return (
        f'<div style="display:flex;justify-content:center;padding:4px 0 2px;">'
        f'<div style="'
        f'  width:58px;height:58px;border-radius:50%;'
        f'  background:{gradient};'
        f'  border:3px solid rgba(255,255,255,0.22);'
        f'  box-shadow:2px 5px 16px rgba(0,0,0,0.55),'
        f'             inset 0 2px 6px rgba(255,255,255,0.18);'
        f'  display:flex;flex-direction:column;'
        f'  align-items:center;justify-content:center;'
        f'  position:relative;overflow:hidden;'
        f'">'
        # gloss highlight
        f'  <div style="'
        f'    position:absolute;top:6px;left:12px;'
        f'    width:22px;height:10px;border-radius:50%;'
        f'    background:rgba(255,255,255,0.28);'
        f'    transform:rotate(-30deg);">'
        f'  </div>'
        # letter label
        f'  <span style="color:rgba(255,255,255,0.75);font-size:9px;'
        f'    font-weight:800;letter-spacing:1.5px;line-height:1;">{letter}</span>'
        # number
        f'  <span style="color:#fff;font-size:{num_size};'
        f'    font-weight:900;line-height:1.25;">{val}</span>'
        f'</div></div>'
    )


def get_global_marked_values() -> list[str]:
    seen = set()
    for idx, card in enumerate(st.session_state.cards):
        for (r, c) in st.session_state.manual_marks.get(idx, set()):
            val = str(card.iloc[r, c]).strip().upper()
            if val not in ("FREE", "") and not val.startswith("?"):
                seen.add(val)
    def sort_key(v):
        try:    return (0, int(v))
        except: return (1, v)
    return sorted(seen, key=sort_key)


def unmark_global_value(val: str):
    val = val.strip().upper()
    for idx, card in enumerate(st.session_state.cards):
        marks = st.session_state.manual_marks.setdefault(idx, set())
        nrows, ncols = card.shape
        for r in range(nrows):
            for c in range(ncols):
                if str(card.iloc[r, c]).strip().upper() == val:
                    marks.discard((r, c))


marked_vals = get_global_marked_values()

hdr_col, clr_col = st.columns([5, 1])
with hdr_col:
    st.markdown(
        f'<div style="font-size:14px;font-weight:600;color:#aaa;padding:6px 0 4px;">'
        f'🎱 Marked values ({len(marked_vals)})'
        + ('&nbsp;<span style="font-size:12px;font-weight:400;color:#555;">'
           '— click ✕ under a ball to unmark it from all cards</span>'
           if marked_vals else '')
        + '</div>',
        unsafe_allow_html=True,
    )
with clr_col:
    if marked_vals and st.button("🗑️ Clear All", key="clear_all_marks",
                                  use_container_width=True):
        st.session_state.manual_marks = {}
        recalc_winners()
        st.rerun()

if marked_vals:
    MAX_PER_ROW = 12
    for row_start in range(0, len(marked_vals), MAX_PER_ROW):
        row_vals  = marked_vals[row_start : row_start + MAX_PER_ROW]
        ball_cols = st.columns(len(row_vals))
        for col, val in zip(ball_cols, row_vals):
            with col:
                # Bingo ball (visual)
                st.markdown(_ball_html(val), unsafe_allow_html=True)
                # Remove button beneath the ball
                if st.button(
                    "✕",
                    key=f"chip_{val}_{st.session_state.round}",
                    use_container_width=True,
                ):
                    unmark_global_value(val)
                    recalc_winners()
                    st.rerun()
else:
    st.markdown(
        '<div style="color:#555;font-size:13px;padding:6px 0 10px;">'
        'No values marked yet — click any cell on the cards below to start.'
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

cols = st.columns(3, gap="small")
for i in range(len(st.session_state.cards)):
    with cols[i % 3]:
        render_card(i)
