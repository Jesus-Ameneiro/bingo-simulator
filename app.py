import streamlit as st
import pandas as pd
import base64
import json
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

# ─── Thumbnail helper ─────────────────────────────────────────────────────────
def _make_thumb(pil_or_bytes, max_width: int = 200) -> str:
    """
    Compress any image down to ≤ max_width px wide, encode as JPEG base64.
    Keeps session-state size tiny (~5–8 KB per card instead of 300–500 KB).
    """
    import io as _io
    if isinstance(pil_or_bytes, (bytes, bytearray)):
        img = Image.open(_io.BytesIO(pil_or_bytes)).convert("RGB")
    else:
        img = pil_or_bytes.convert("RGB")
    # Resize proportionally
    w, h = img.size
    if w > max_width:
        img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

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
def _ocr_cell(cell_bgr: np.ndarray) -> str:
    """
    OCR a single bingo card cell.
    Tries multiple thresholds and PSM modes; returns majority-vote result.
    Enforces standard bingo range (1–75).
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return "?"
    h2, w2 = cell_bgr.shape[:2]
    if h2 < 10 or w2 < 10:
        return "?"

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale very small cells
    if h2 < 100:
        scale = max(2, 100 // h2)
        gray  = cv2.resize(gray, (w2 * scale, h2 * scale), cv2.INTER_CUBIC)

    results: list[int] = []
    for thr in (127, 100, 150):
        _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        binary = cv2.copyMakeBorder(binary, 25, 25, 25, 25,
                                    cv2.BORDER_CONSTANT, value=255)
        for psm in (7, 6, 8):
            cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
            try:
                raw    = pytesseract.image_to_string(binary, config=cfg).strip()
                digits = "".join(ch for ch in raw if ch.isdigit())
                if digits:
                    n = int(digits)
                    if 1 <= n <= 75:          # enforce bingo range
                        results.append(n)
            except Exception:
                pass

    if not results:
        return "?"
    from collections import Counter
    return str(Counter(results).most_common(1)[0][0])


def _find_white_bands(sat_1d: np.ndarray, y_start: int, y_end: int,
                      threshold: float = 25.0, min_width: int = 12) -> list[tuple[int, int]]:
    """Return list of (start, end) spans where saturation < threshold."""
    bands: list[tuple[int, int]] = []
    in_band = False
    band_start = y_start
    for y in range(y_start, y_end):
        if sat_1d[y] < threshold and not in_band:
            in_band    = True
            band_start = y
        elif sat_1d[y] >= threshold and in_band:
            in_band = False
            if y - band_start >= min_width:
                bands.append((band_start, y))
    if in_band and y_end - band_start >= min_width:
        bands.append((band_start, y_end))
    return bands


def scan_card_from_image(pil_img: Image.Image, grid_size: int = 5) -> list[list[str]]:
    """
    Robust bingo card reader that handles two card designs automatically:

    Style A — Clean green-border cards (one solid white band per row):
      • Detects exactly grid_size large white bands with small inter-band gaps.
      • Finds actual column dividers via vertical saturation scan.
      • Uses tight inner padding to keep full digits in frame.

    Style B — Pink-stripe cards (two white sub-bands per row):
      • Each data row has a thick decorative pink stripe splitting digits in half.
      • Pairs the two white sub-bands (connected by large gap > 50 px) into one row.
      • Uses even column division from outer green-border edges with overshoot.

    Pipeline (both styles):
      1. Work at original resolution — larger cells → better OCR.
      2. Detect card extent and BINGO header via row-saturation profile.
      3. Classify card style from white-band gaps.
      4. Build row & column boundaries accordingly.
      5. OCR each cell: multi-threshold × multi-PSM, majority vote, range 1–75.
    """
    # ── 0. Load at original resolution ────────────────────────────────────────
    orig    = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── 1. Row saturation profile ──────────────────────────────────────────────
    cx1, cx2 = int(w * 0.12), int(w * 0.88)
    row_sat  = np.convolve(
        hsv[:, cx1:cx2, 1].mean(axis=1), np.ones(3) / 3, mode="same"
    )

    # Card top / bottom from high-saturation (green/pink) border rows
    card_top = 0
    for y in range(h):
        if row_sat[y] > 60:
            card_top = y
            break
    card_bottom = h
    for y in range(h - 1, card_top, -1):
        if row_sat[y] > 60:
            card_bottom = y
            break

    # Header end: first sustained low-saturation zone after the coloured header
    header_end = card_top
    for y in range(card_top + 10, h - 20):
        if row_sat[y] < 25 and row_sat[min(y + 10, h - 1)] < 40:
            header_end = y
            break

    # ── 2. Find white horizontal bands ────────────────────────────────────────
    all_bands = _find_white_bands(row_sat, header_end, card_bottom)
    gaps = ([all_bands[i + 1][0] - all_bands[i][1]
             for i in range(len(all_bands) - 1)]
            if len(all_bands) > 1 else [])
    n_large_gaps = sum(1 for g in gaps if g > 50)

    # ── 3. Route to appropriate strategy ──────────────────────────────────────

    # ── Style A: clean cards — one large white band per row ───────────────────
    if len(all_bands) == grid_size and all(g < 30 for g in gaps):
        # Use white band extents directly as row boundaries
        rows: list[int] = [max(header_end, b[0] - 5) for b in all_bands]
        rows.append(min(card_bottom, all_bands[-1][1] + 5))

        # Detect column dividers via vertical saturation in the data zone
        y1c = all_bands[1][0]
        y2c = all_bands[-2][1]
        col_sat = np.convolve(
            hsv[y1c:y2c, :, 1].mean(axis=0), np.ones(5) / 5, mode="same"
        )
        dividers: list[int] = []
        in_div = False
        d_start = 0
        for x in range(w):
            if col_sat[x] >= 60 and not in_div:
                in_div = True
                d_start = x
            elif col_sat[x] < 60 and in_div:
                in_div = False
                if x - d_start >= 3:
                    dividers.append(int((d_start + x) / 2))
        if in_div and w - d_start >= 3:
            dividers.append(int((d_start + w) / 2))

        if len(dividers) >= grid_size + 1:
            cols_x = dividers[:grid_size + 1]
        else:
            lx = dividers[0] if dividers else int(w * 0.05)
            rx = dividers[-1] if len(dividers) > 1 else int(w * 0.95)
            cols_x = [int(lx + i * (rx - lx) / grid_size)
                      for i in range(grid_size + 1)]

        # Tight inner pad (2 px) avoids clipping leading digits
        mid  = grid_size // 2
        PAD  = 2
        grid: list[list[str]] = []
        for r in range(grid_size):
            row: list[str] = []
            y1 = rows[r]  + PAD
            y2 = rows[r + 1] - PAD
            for c in range(grid_size):
                if r == mid and c == mid:
                    row.append("FREE")
                    continue
                x1 = max(0, cols_x[c]     + PAD)
                x2 = min(w, cols_x[c + 1] - PAD)
                cell = img_bgr[y1:y2, x1:x2]
                row.append(_ocr_cell(cell) if cell.size > 0 else "?")
            grid.append(row)
        return grid

    # ── Style B: pink-stripe cards — two sub-bands per row ────────────────────
    if n_large_gaps >= grid_size - 1:
        pairs: list[tuple[int, int, int, int]] = []
        i = 0
        while i < len(all_bands):
            s1, e1 = all_bands[i]
            if i + 1 < len(all_bands):
                s2, e2 = all_bands[i + 1]
                if s2 - e1 > 50:
                    pairs.append((s1, e1, s2, e2))
                    i += 2
                    continue
            i += 1

        rows = []
        if len(pairs) >= grid_size:
            for s1, e1, s2, e2 in pairs[:grid_size]:
                rows.append(max(header_end, s1 - 5))
            rows.append(min(card_bottom, pairs[grid_size - 1][3] + 5))
        else:
            bh = (card_bottom - header_end) / grid_size
            rows = [int(header_end + j * bh) for j in range(grid_size + 1)]

        # Even column division from outer green-border edges
        col_sat = hsv[header_end:card_bottom, :, 1].mean(axis=0)
        left_x = 0
        for x in range(w):
            if col_sat[x] > 50:
                left_x = x
                break
        right_x = w
        for x in range(w - 1, 0, -1):
            if col_sat[x] > 50:
                right_x = x
                break
        cols_x = [int(left_x + i * (right_x - left_x) / grid_size)
                  for i in range(grid_size + 1)]

        mid = grid_size // 2
        COL_OVERSHOOT = 10
        grid = []
        for r in range(grid_size):
            row = []
            y1 = rows[r]     + 6
            y2 = rows[r + 1] - 6
            for c in range(grid_size):
                if r == mid and c == mid:
                    row.append("FREE")
                    continue
                x1 = max(0, cols_x[c]     - COL_OVERSHOOT)
                x2 = min(w, cols_x[c + 1] + COL_OVERSHOOT)
                cell = img_bgr[y1:y2, x1:x2]
                row.append(_ocr_cell(cell) if cell.size > 0 else "?")
            grid.append(row)
        return grid

    # ── Fallback: even division using last white band as bottom ────────────────
    data_end = all_bands[-1][1] if all_bands else card_bottom
    bh       = (data_end - header_end) / grid_size
    rows     = [int(header_end + j * bh) for j in range(grid_size + 1)]

    col_sat = hsv[header_end:card_bottom, :, 1].mean(axis=0)
    left_x = 0
    for x in range(w):
        if col_sat[x] > 50:
            left_x = x
            break
    right_x = w
    for x in range(w - 1, 0, -1):
        if col_sat[x] > 50:
            right_x = x
            break
    cols_x = [int(left_x + i * (right_x - left_x) / grid_size)
              for i in range(grid_size + 1)]

    mid  = grid_size // 2
    grid = []
    for r in range(grid_size):
        row = []
        y1 = rows[r]     + 6
        y2 = rows[r + 1] - 6
        for c in range(grid_size):
            if r == mid and c == mid:
                row.append("FREE")
                continue
            x1 = max(0, cols_x[c]     - 10)
            x2 = min(w, cols_x[c + 1] + 10)
            cell = img_bgr[y1:y2, x1:x2]
            row.append(_ocr_cell(cell) if cell.size > 0 else "?")
        grid.append(row)
    return grid


# ─── Session State ────────────────────────────────────────────────────────────
if "cards" not in st.session_state:
    st.session_state.cards       = []
    st.session_state.card_names  = []
    st.session_state.card_thumbs = []

_DEFAULTS = {
    "manual_marks":   {},
    "winners":        set(),
    "round":          1,
    "round_pattern":  set(),
    "pattern_size":   5,
    "ocr_grid":       None,
    "ocr_thumb":      "",
    "ocr_name":       "",
    "ocr_pending":    [],
    "lang":           "es",
    "load_key":       0,    # incremented after each successful .bingo import to reset uploader
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

# ─── i18n Strings ─────────────────────────────────────────────────────────────
_T = {
    # ── Sidebar ──────────────────────────────────────────────────────────────
    "settings":             {"es": "⚙️ Configuración",          "en": "⚙️ Settings"},
    "language":             {"es": "Idioma / Language",          "en": "Idioma / Language"},
    "rules_image":          {"es": "### 📜 Imagen de Reglas",    "en": "### 📜 Rules Image"},
    "rules_img_caption":    {"es": "Carga una imagen de referencia para las reglas.",
                             "en": "Upload for visual reference."},
    "round_mgmt":           {"es": "### 🔁 Gestión de Ronda",   "en": "### 🔁 Round Management"},
    "keep_cards":           {"es": "🔄 Nueva Ronda — Mantener Tarjetas",
                             "en": "🔄 New Round — Keep Cards"},
    "clear_cards":          {"es": "🆕 Nueva Ronda — Borrar Todas las Tarjetas",
                             "en": "🆕 New Round — Clear All Cards"},
    # ── Header ───────────────────────────────────────────────────────────────
    "round_badge":          {"es": "Ronda",                      "en": "Round"},
    # ── Expander / tabs ──────────────────────────────────────────────────────
    "load_cards":           {"es": "📥 Cargar Tarjetas",         "en": "📥 Load Cards"},
    "add_more_cards":       {"es": "➕ Agregar Más Tarjetas",    "en": "➕ Add More Cards"},
    "tab_scan":             {"es": "📷 Escanear Imagen",         "en": "📷 Scan from Image"},
    "tab_manual":           {"es": "✏️ Ingresar Manualmente",    "en": "✏️ Enter Manually"},
    # ── Scan tab ─────────────────────────────────────────────────────────────
    "ocr_unavailable":      {"es": "⚠️ Librerías OCR no instaladas. Asegúrate de que "
                                   "`packages.txt` tenga `tesseract-ocr` y `requirements.txt` "
                                   "tenga `pytesseract`, `opencv-python-headless`, `Pillow` y "
                                   "`numpy`, luego vuelve a desplegar.",
                             "en": "⚠️ OCR libraries not installed. Make sure `packages.txt` "
                                   "contains `tesseract-ocr` and `requirements.txt` contains "
                                   "`pytesseract`, `opencv-python-headless`, `Pillow`, and "
                                   "`numpy`, then redeploy."},
    "welcome_banner":       {"es": "👋 <strong>¡Bienvenido!</strong> Carga todas las imágenes "
                                   "de tus tarjetas de bingo abajo para comenzar. Puedes "
                                   "seleccionar varios archivos a la vez.",
                             "en": "👋 <strong>Welcome!</strong> Upload all your bingo card "
                                   "images below to get started. You can select multiple files "
                                   "at once."},
    "grid_size":            {"es": "Tamaño de cuadrícula",       "en": "Grid size"},
    "scan_caption":         {"es": "Cada archivo se convertirá en una tarjeta. El OCR lee los "
                                   "números automáticamente — puedes corregir errores antes de "
                                   "confirmar.",
                             "en": "Each file will become one card. OCR reads the numbers "
                                   "automatically — you can fix errors before confirming."},
    "upload_label":         {"es": "Cargar imágenes de tarjetas", "en": "Upload card images"},
    "scan_all_btn":         {"es": "🔍 Escanear las {} Tarjeta(s)",
                             "en": "🔍 Scan All {} Card(s)"},
    "scanning":             {"es": "Escaneando {}…",             "en": "Scanning {}…"},
    "scan_failed":          {"es": "⚠️ No se pudo escanear **{}**: {}",
                             "en": "⚠️ Could not scan **{}**: {}"},
    "scan_complete":        {"es": "¡Escaneo completo!",         "en": "Scan complete!"},
    # ── Review section ───────────────────────────────────────────────────────
    "review_title":         {"es": "#### ✅ Revisar {} Tarjeta(s) Escaneada(s)",
                             "en": "#### ✅ Review {} Scanned Card(s)"},
    "review_caption":       {"es": "La cuadrícula de cada tarjeta se muestra abajo. "
                                   "Corrige los valores incorrectos (barra roja = OCR incierto), "
                                   "luego haz clic en **Confirmar Todas las Tarjetas**.",
                             "en": "Each card grid is shown below. Fix any incorrect values "
                                   "(red bar = OCR was uncertain), then click "
                                   "**Confirm All Cards**."},
    "card_name_label":      {"es": "Nombre de tarjeta",          "en": "Card name"},
    "image_reference":      {"es": "📷 Imagen de referencia",    "en": "📷 Reference image"},
    "cells_need_input":     {"es": "⚠️ {} celda(s) requieren entrada manual arriba.",
                             "en": "⚠️ {} cell(s) need manual input above."},
    "confirm_all":          {"es": "✅ Confirmar Todas las {} Tarjetas",
                             "en": "✅ Confirm All {} Cards"},
    "discard_all":          {"es": "🗑️ Descartar Todo",          "en": "🗑️ Discard All"},
    "cards_added":          {"es": "✅ {} tarjeta(s) agregada(s)!", "en": "✅ {} card(s) added!"},
    # ── Manual entry ─────────────────────────────────────────────────────────
    "manual_ref_title":     {"es": "#### 🖼️ Imagen de Referencia (Opcional)",
                             "en": "#### 🖼️ Reference Image (Optional)"},
    "manual_ref_caption":   {"es": "Carga una foto de tu tarjeta para usarla como guía visual.",
                             "en": "Upload a photo of your card to use as a visual guide."},
    "manual_grid_title":    {"es": "#### ✏️ Llenar la Cuadrícula",
                             "en": "#### ✏️ Fill in the Grid"},
    "manual_grid_caption":  {"es": "Escribe cada valor exactamente como está impreso en la tarjeta.",
                             "en": "Type each value exactly as printed on the card."},
    "has_free":             {"es": "La tarjeta tiene espacio libre",
                             "en": "Card has a free space"},
    "free_row":             {"es": "Fila del espacio libre",     "en": "Free space row"},
    "free_col":             {"es": "Col del espacio libre",      "en": "Free space col"},
    "new_card_name":        {"es": "Nombre de la nueva tarjeta", "en": "New card name"},
    "card_default_name":    {"es": "Tarjeta {}",                 "en": "Card {}"},
    "add_card_btn":         {"es": "✅ Agregar Tarjeta",         "en": "✅ Add Card"},
    "card_added_ok":        {"es": "✅ '{}' agregada!",          "en": "✅ '{}' added!"},
    # ── Pattern card ─────────────────────────────────────────────────────────
    "pattern_title":        {"es": "🎯 Patrón Ganador de la Ronda",
                             "en": "🎯 Round Winning Pattern"},
    "pattern_status_set":   {"es": "Personalizado — {} celda(s) requeridas",
                             "en": "Custom — {} cell(s) required"},
    "pattern_status_none":  {"es": "Sin patrón — haz clic en celdas abajo",
                             "en": "No pattern — click cells below"},
    "pattern_ctrl_caption": {"es": "Haz clic en las celdas de la cuadrícula para definir "
                                   "qué posiciones debe completar una tarjeta para ganar. "
                                   "Usar presets o borrar.",
                             "en": "Click cells on the grid to define which positions a card "
                                   "must complete to win. Use presets or clear."},
    "clear_pattern":        {"es": "🗑️ Borrar patrón",          "en": "🗑️ Clear pattern"},
    "presets_title":        {"es": "**Presets rápidos**",        "en": "**Quick Presets**"},
    # ── Playing cards ────────────────────────────────────────────────────────
    "mark_hint":            {"es": "💡 Haz clic en cualquier celda para marcarla o desmarcarla. "
                                   "El mismo valor se marcará automáticamente en todas las demás tarjetas.",
                             "en": "💡 Click any cell on the cards below to mark or unmark it. "
                                   "The same value will be marked automatically on all other cards."},
    "marked_values":        {"es": "Valores Marcados",           "en": "Marked Values"},
    "clear_all_marks":      {"es": "🗑️ Borrar Todo",            "en": "🗑️ Clear All"},
    "winner_banner":        {"es": "🏆 ¡BINGO! &nbsp;&nbsp; {} &nbsp;&nbsp; 🏆",
                             "en": "🏆 BINGO! &nbsp;&nbsp; {} &nbsp;&nbsp; 🏆"},
    "remove_card":          {"es": "🗑️ Eliminar tarjeta",        "en": "🗑️ Remove card"},
    # ── Save / Load session ───────────────────────────────────────────────────
    "save_load":            {"es": "### 💾 Guardar / Cargar",     "en": "### 💾 Save / Load"},
    "save_btn":             {"es": "⬇️ Descargar sesión (.bingo)",
                             "en": "⬇️ Download session (.bingo)"},
    "save_caption":         {"es": "Guarda todas las tarjetas y cuadrículas en un archivo.",
                             "en": "Saves all cards and grids to a file."},
    "load_caption":         {"es": "Carga un archivo .bingo guardado previamente.",
                             "en": "Load a previously saved .bingo file."},
    "load_success":         {"es": "✅ {} tarjeta(s) cargadas desde el archivo.",
                             "en": "✅ {} card(s) loaded from file."},
    "load_error":           {"es": "⚠️ El archivo no es válido o está dañado.",
                             "en": "⚠️ File is invalid or corrupted."},
    "no_cards_to_save":     {"es": "No hay tarjetas para guardar.",
                             "en": "No cards to save yet."},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def t(key: str, *args) -> str:
    """Return localised string for current language, optionally format with args."""
    lang = st.session_state.get("lang", "es")
    text = _T.get(key, {}).get(lang, _T.get(key, {}).get("en", key))
    return text.format(*args) if args else text

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
    status   = (t("pattern_status_set", n_marked) if pattern
                else t("pattern_status_none"))

    # ── Title bar ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f3460,#16213e);'
        'border-radius:10px 10px 0 0;padding:14px 20px;display:flex;'
        'align-items:center;justify-content:space-between;">'
        f'<span style="color:#ffd700;font-weight:700;font-size:18px;letter-spacing:0.5px;">'
        f'{t("pattern_title")}</span>'
        f'<span style="background:{"#28a745" if pattern else "#555"};color:white;'
        f'padding:4px 16px;border-radius:14px;font-size:13px;font-weight:600;">'
        f'{status}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        ctrl_col, grid_col, preset_col = st.columns([1, 1.4, 1.6], gap="large")

        with ctrl_col:
            st.markdown(
                f'<p style="color:#aaa;font-size:13px;margin-bottom:16px;">'
                f'{t("pattern_ctrl_caption")}</p>',
                unsafe_allow_html=True,
            )

            st.markdown(f"**{t('grid_size')}**")
            new_size = st.radio(
                t("grid_size"), [3, 4, 5], index=[3, 4, 5].index(size),
                horizontal=True, label_visibility="collapsed",
                key="pat_size_radio",
            )
            if new_size != size:
                st.session_state.pattern_size  = new_size
                st.session_state.round_pattern = set()
                recalc_winners()
                st.rerun()

            st.markdown("")
            if st.button(t("clear_pattern"), key="clr_pat", use_container_width=True):
                st.session_state.round_pattern = set()
                recalc_winners()
                st.rerun()

            legend = {"es": "✅ Celda requerida<br>⬜ No requerida<br>★ Espacio libre (siempre cuenta)",
                      "en": "✅ Required cell<br>⬜ Not required<br>★ Free space (always counts)"}
            st.markdown(
                f'<div style="margin-top:20px;padding:10px;background:#111;'
                f'border-radius:8px;font-size:12px;color:#888;line-height:1.7;">'
                f'{legend[st.session_state.get("lang","es")]}</div>',
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
            presets_hdr = {"es": "**⚡ Patrones Rápidos**", "en": "**⚡ Quick Patterns**"}
            presets_cap = {"es": "Haz clic en un preset para cargarlo.",
                           "en": "Click a preset to load it instantly."}
            st.markdown(presets_hdr[st.session_state.get("lang", "es")])
            st.caption(presets_cap[st.session_state.get("lang", "es")])

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
            f'<div class="winner-banner">{t("winner_banner", name)}</div>',
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
    if st.button(t("remove_card"), key=f"rm_{idx}_{st.session_state.round}",
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
    st.markdown(f"## {t('settings')}")

    # ── Language toggle ────────────────────────────────────────────────────────
    lang_choice = st.radio(
        t("language"),
        options=["Español", "English"],
        index=0 if st.session_state.lang == "es" else 1,
        horizontal=True,
    )
    new_lang = "es" if lang_choice == "Español" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.markdown("---")
    st.markdown(t("rules_image"))
    st.caption(t("rules_img_caption"))
    rules_img = st.file_uploader(
        "Rules image", type=["png", "jpg", "jpeg", "webp"],
        key="rules_img_up", label_visibility="collapsed",
    )
    if rules_img:
        st.image(rules_img, use_container_width=True)

    # ── Save / Load ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(t("save_load"))
    st.caption(t("save_caption"))

    # ── SAVE ──────────────────────────────────────────────────────────────────
    if st.session_state.cards:
        session_data = {
            "version": 1,
            "cards": [
                {
                    "name":  st.session_state.card_names[i],
                    "grid":  st.session_state.cards[i].values.tolist(),
                    "thumb": st.session_state.card_thumbs[i],
                }
                for i in range(len(st.session_state.cards))
            ],
        }
        st.download_button(
            label    = t("save_btn"),
            data     = json.dumps(session_data, ensure_ascii=False, indent=2),
            file_name= "bingo_session.bingo",
            mime     = "application/json",
            use_container_width=True,
        )
    else:
        st.caption(t("no_cards_to_save"))

    # ── LOAD ──────────────────────────────────────────────────────────────────
    st.caption(t("load_caption"))
    bingo_file = st.file_uploader(
        "Load .bingo", type=["bingo", "json"],
        key=f"bingo_load_up_{st.session_state.load_key}",
        label_visibility="collapsed",
    )
    if bingo_file is not None:
        try:
            raw_data = json.loads(bingo_file.read().decode("utf-8"))
            cards_in = raw_data.get("cards", [])
            if not cards_in:
                raise ValueError("empty")

            new_cards  = []
            new_names  = []
            new_thumbs = []
            for entry in cards_in:
                df = pd.DataFrame(entry["grid"]).astype(str)
                new_cards.append(df)
                new_names.append(entry.get("name", f"Card {len(new_cards)}"))
                new_thumbs.append(entry.get("thumb", ""))

            st.session_state.cards         = new_cards
            st.session_state.card_names    = new_names
            st.session_state.card_thumbs   = new_thumbs
            st.session_state.manual_marks  = {}
            st.session_state.winners       = set()
            st.session_state.round_pattern = set()
            # Increment key → uploader resets to empty on next render (breaks the loop)
            st.session_state.load_key     += 1
            recalc_winners()
            st.success(t("load_success", len(new_cards)))
            st.rerun()
        except Exception:
            st.error(t("load_error"))

    st.markdown("---")
    st.markdown(t("round_mgmt"))

    if st.button(t("keep_cards"), use_container_width=True):
        st.session_state.manual_marks  = {}
        st.session_state.winners       = set()
        st.session_state.round_pattern = set()
        st.session_state.round        += 1
        st.rerun()

    if st.button(t("clear_cards"), use_container_width=True):
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
  <span class="round-badge">{t('round_badge')} {st.session_state.round}</span>
</div>
""", unsafe_allow_html=True)


# ─── Add a Card ───────────────────────────────────────────────────────────────
no_cards = not bool(st.session_state.cards)
with st.expander(
    t("load_cards") if no_cards else t("add_more_cards"),
    expanded=no_cards,
):

    tab_scan, tab_manual = st.tabs([t("tab_scan"), t("tab_manual")])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — SCAN FROM IMAGE (multiple at once)
    # ════════════════════════════════════════════════════════════════════════
    with tab_scan:
        if not _OCR_AVAILABLE:
            st.warning(t("ocr_unavailable"))
        else:
            if no_cards:
                st.markdown(
                    f'<div style="background:#0f3460;border-radius:8px;padding:12px 16px;'
                    f'margin-bottom:14px;color:#e0e0e0;font-size:14px;">'
                    f'{t("welcome_banner")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            opt1, opt2 = st.columns([1, 2])
            with opt1:
                scan_size = st.selectbox(t("grid_size"), [5, 4, 3, 6], index=0, key="scan_size")
            with opt2:
                st.caption(t("scan_caption"))

            scan_files = st.file_uploader(
                t("upload_label"),
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="scan_img_upload",
                label_visibility="collapsed",
            )

            if scan_files:
                prev_cols = st.columns(min(len(scan_files), 6))
                for i, f in enumerate(scan_files):
                    with prev_cols[i % 6]:
                        st.image(f, use_container_width=True,
                                 caption=f.name.rsplit(".", 1)[0])

                st.markdown("")
                if st.button(
                    t("scan_all_btn", len(scan_files)),
                    type="primary", use_container_width=True, key="do_scan_all",
                ):
                    progress = st.progress(0, text=t("scanning", "…"))
                    pending  = []

                    for i, f in enumerate(scan_files):
                        progress.progress(i / len(scan_files), text=t("scanning", f.name))
                        try:
                            f.seek(0)
                            pil_img = Image.open(f)
                            grid    = scan_card_from_image(pil_img, scan_size)
                            # Store a small compressed thumbnail — not the full image
                            thumb   = _make_thumb(pil_img)
                            pending.append({
                                "name":  f.name.rsplit(".", 1)[0],
                                "grid":  grid,
                                "thumb": thumb,
                            })
                        except Exception as e:
                            st.warning(t("scan_failed", f.name, e))

                    progress.progress(1.0, text=t("scan_complete"))
                    st.session_state.ocr_pending = pending
                    st.rerun()

            # ── Review all scanned cards ──────────────────────────────────────
            pending = st.session_state.get("ocr_pending", [])
            if pending:
                st.divider()
                st.markdown(t("review_title", len(pending)))
                st.caption(t("review_caption"))

                all_edited = []
                for card_idx, card_data in enumerate(pending):
                    raw_grid = card_data["grid"]
                    size     = len(raw_grid)
                    mid      = size // 2
                    thumb    = card_data["thumb"]

                    with st.container(border=True):
                        # ── Card name row ─────────────────────────────────────
                        edited_name = st.text_input(
                            t("card_name_label"),
                            value=card_data["name"],
                            key=f"ocr_name_{card_idx}",
                            label_visibility="visible",
                        )

                        # ── Full reference image + grid side by side ──────────
                        img_col, grid_col = st.columns([1, 2], gap="medium")

                        with img_col:
                            if thumb:
                                st.markdown(
                                    f'<p style="color:#aaa;font-size:12px;margin-bottom:4px;">'
                                    f'{t("image_reference")}</p>'
                                    f'<img src="{thumb}" style="width:100%;border-radius:8px;'
                                    f'border:1px solid #444;" />',
                                    unsafe_allow_html=True,
                                )

                        with grid_col:
                            # Column headers
                            labels = list("BINGO") if size == 5 else [str(i+1) for i in range(size)]
                            hc = st.columns(size)
                            for ci, lbl in enumerate(labels):
                                with hc[ci]:
                                    st.markdown(
                                        f'<div style="background:#1a1a2e;color:#ffd700;'
                                        f'text-align:center;font-weight:bold;font-size:14px;'
                                        f'padding:5px;border-radius:5px;">{lbl}</div>',
                                        unsafe_allow_html=True,
                                    )

                            edited_grid = []
                            for r in range(size):
                                rc  = st.columns(size)
                                row = []
                                for c in range(size):
                                    with rc[c]:
                                        if r == mid and c == mid:
                                            st.markdown(
                                                '<div style="background:#ffd700;color:#1a1a2e;'
                                                'text-align:center;font-weight:bold;font-size:18px;'
                                                'padding:7px;border-radius:5px;">★</div>',
                                                unsafe_allow_html=True,
                                            )
                                            row.append("FREE")
                                        else:
                                            cur = raw_grid[r][c] if raw_grid[r][c] != "?" else ""
                                            uncertain = raw_grid[r][c] in ("?", "")
                                            if uncertain:
                                                st.markdown(
                                                    '<div style="background:#8b0000;height:3px;'
                                                    'border-radius:2px;margin-bottom:2px;"></div>',
                                                    unsafe_allow_html=True,
                                                )
                                            v = st.text_input(
                                                f"ocr_{card_idx}_{r}_{c}",
                                                value=cur,
                                                key=f"ocr_cell_{card_idx}_{r}_{c}",
                                                label_visibility="collapsed",
                                                placeholder="?",
                                            )
                                            row.append(v.strip() if v.strip() else f"?{r}{c}")
                                edited_grid.append(row)

                            bad = sum(
                                1 for r in range(size) for c in range(size)
                                if edited_grid[r][c].startswith("?") and not (r == mid and c == mid)
                            )
                            if bad:
                                st.warning(t("cells_need_input", bad))

                    all_edited.append({"name": edited_name, "grid": edited_grid,
                                       "thumb": thumb})

                st.markdown("")
                confirm_col, discard_col = st.columns([3, 1])
                with confirm_col:
                    if st.button(
                        t("confirm_all", len(all_edited)),
                        type="primary", use_container_width=True, key="confirm_all_scans",
                    ):
                        for item in all_edited:
                            df = pd.DataFrame(item["grid"]).astype(str)
                            st.session_state.cards.append(df)
                            st.session_state.card_names.append(item["name"])
                            st.session_state.card_thumbs.append(item["thumb"])
                        st.session_state.ocr_pending = []
                        recalc_winners()
                        st.success(t("cards_added", len(all_edited)))
                        st.rerun()
                with discard_col:
                    if st.button(t("discard_all"), use_container_width=True,
                                 key="discard_all_scans"):
                        st.session_state.ocr_pending = []
                        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — MANUAL ENTRY
    # ════════════════════════════════════════════════════════════════════════
    with tab_manual:
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown(t("manual_ref_title"))
            st.caption(t("manual_ref_caption"))
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
                grid_size = st.selectbox(t("grid_size"), [5, 4, 3, 6], index=0, key="new_card_size")
            with c2:
                new_name = st.text_input(
                    t("new_card_name"),
                    value=t("card_default_name", len(st.session_state.cards) + 1),
                    key="new_card_name_input",
                )
            has_free = st.checkbox(f"⭐ {t('has_free')}", value=True, key="new_has_free")
            if has_free:
                f1, f2 = st.columns(2)
                with f1:
                    free_row = int(st.number_input(
                        t("free_row"), 0, grid_size-1,
                        value=min(2, grid_size-1), key="new_fr",
                    ))
                with f2:
                    free_col = int(st.number_input(
                        t("free_col"), 0, grid_size-1,
                        value=min(2, grid_size-1), key="new_fc",
                    ))
            else:
                free_row, free_col = -1, -1

        with right_col:
            st.markdown(t("manual_grid_title"))
            st.caption(t("manual_grid_caption"))

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
            if st.button(t("add_card_btn"), use_container_width=True, key="add_card_btn"):
                df    = pd.DataFrame(grid_vals).astype(str)
                thumb = ""
                if card_img:
                    card_img.seek(0)
                    thumb = _make_thumb(Image.open(card_img))

                st.session_state.cards.append(df)
                st.session_state.card_names.append(
                    new_name or t("card_default_name", len(st.session_state.cards))
                )
                st.session_state.card_thumbs.append(thumb)

                if has_free:
                    st.session_state.rules["free_space"]     = True
                    st.session_state.rules["free_space_row"] = free_row
                    st.session_state.rules["free_space_col"] = free_col
                else:
                    st.session_state.rules["free_space"] = False

                recalc_winners()
                st.success(t("card_added_ok", new_name))
                st.rerun()


# ─── Pattern card + Playing cards ────────────────────────────────────────────
if not st.session_state.cards:
    st.stop()

# Winner banners
for wi in sorted(st.session_state.winners):
    st.markdown(
        f'<div class="winner-banner">'
        f'{t("winner_banner", st.session_state.card_names[wi])}</div>',
        unsafe_allow_html=True,
    )

# ── Round Pattern card — full width ───────────────────────────────────────────
render_pattern_card()

st.markdown(
    f'<div style="font-size:13px;color:#6c757d;margin:16px 0 8px;">'
    f'{t("mark_hint")}'
    f'</div>',
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
        f'🎱 {t("marked_values")} ({len(marked_vals)})'
        + ('&nbsp;<span style="font-size:12px;font-weight:400;color:#555;">'
           '— click ✕ under a ball to unmark it from all cards</span>'
           if marked_vals else '')
        + '</div>',
        unsafe_allow_html=True,
    )
with clr_col:
    if marked_vals and st.button(t("clear_all_marks"), key="clear_all_marks",
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
                st.markdown(_ball_html(val), unsafe_allow_html=True)
                if st.button(
                    "✕",
                    key=f"chip_{val}_{st.session_state.round}",
                    use_container_width=True,
                ):
                    unmark_global_value(val)
                    recalc_winners()
                    st.rerun()
else:
    no_marked_msg = {"es": "Sin valores marcados — haz clic en cualquier celda para comenzar.",
                     "en": "No values marked yet — click any cell on the cards below to start."}
    st.markdown(
        f'<div style="color:#555;font-size:13px;padding:6px 0 10px;">'
        f'{no_marked_msg[st.session_state.get("lang","es")]}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Playing cards — fixed 3 columns ──────────────────────────────────────────
cards_label = {"es": "Tarjetas", "en": "Cards"}
st.markdown(
    f"### 🃏 {cards_label[st.session_state.get('lang','es')]}"
    f"&nbsp;<span style='color:#6c757d;font-size:16px;'>"
    f"({len(st.session_state.cards)})</span>",
    unsafe_allow_html=True,
)

cols = st.columns(3, gap="small")
for i in range(len(st.session_state.cards)):
    with cols[i % 3]:
        render_card(i)
