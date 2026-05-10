import streamlit as st
import pandas as pd
import json
import base64
import os
import anthropic

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
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "cards": [],
    "card_names": [],
    "card_thumbs": [],      # base64 data-URLs for thumbnail display
    "called_values": [],
    "winners": set(),
    "round": 1,
    "last_warning": None,
    "rules_description": None,
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


# ─── Anthropic client ─────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def require_client():
    client = get_client()
    if client is None:
        st.error(
            "⚠️ **ANTHROPIC_API_KEY not found.**\n\n"
            "Add it to your Streamlit secrets:\n"
            "- Streamlit Cloud → App settings → Secrets → `ANTHROPIC_API_KEY = \"sk-ant-...\"`\n"
            "- Locally: create `.streamlit/secrets.toml` with the same key."
        )
        st.stop()
    return client


# ─── Image + Vision helpers ───────────────────────────────────────────────────
def file_to_b64(f) -> tuple:
    """Return (base64_string, media_type) from an UploadedFile."""
    raw = f.read()
    name = f.name.lower()
    if name.endswith(".png"):
        mt = "image/png"
    elif name.endswith(".gif"):
        mt = "image/gif"
    elif name.endswith(".webp"):
        mt = "image/webp"
    else:
        mt = "image/jpeg"
    return base64.standard_b64encode(raw).decode("utf-8"), mt


def make_data_url(b64: str, mt: str) -> str:
    return f"data:{mt};base64,{b64}"


def call_vision(client, b64: str, mt: str, prompt: str, max_tokens: int = 1200) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = resp.content[0].text.strip()
    # Strip markdown code fences the model may have added
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                return part
    return text


# ─── Analysis prompts ─────────────────────────────────────────────────────────
CARD_PROMPT = """You are reading a printed or digital bingo card.
Carefully extract every cell value from the grid, row by row, left to right.

Return ONLY a raw JSON object — no markdown, no explanation, nothing else outside the JSON:
{
  "grid": [
    ["B1",  "I16", "N31", "G46", "O61"],
    ["B2",  "I17", "N32", "G47", "O62"],
    ["B3",  "I18", "FREE","G48", "O63"],
    ["B4",  "I19", "N34", "G49", "O64"],
    ["B5",  "I20", "N35", "G50", "O65"]
  ],
  "has_free_space": true,
  "free_space_row": 2,
  "free_space_col": 2
}

Rules:
- grid: rectangular 2-D array — each inner array is one row of the card.
- Copy values EXACTLY as printed (letters, numbers, hyphens, etc.).
- If a cell is blank or is a free/wild space, write "FREE".
- has_free_space: true if any cell is a free/wild space.
- free_space_row / free_space_col: zero-indexed position of that cell (usually 2, 2 for 5x5)."""

RULES_PROMPT = """You are reading a bingo rules card or sheet.
Determine the winning conditions described or shown in this image.

Return ONLY a raw JSON object — no markdown, no explanation, nothing else:
{
  "check_rows":      true,
  "check_cols":      true,
  "check_diagonals": false,
  "check_full_card": false,
  "free_space":      true,
  "free_space_row":  2,
  "free_space_col":  2,
  "description":     "Complete any full row or column to win. Center cell is a free space."
}

Interpretation guide:
- check_rows:      true if completing any horizontal row wins.
- check_cols:      true if completing any vertical column wins.
- check_diagonals: true if completing a diagonal wins.
- check_full_card: true if ALL cells must be marked (blackout / coverall).
- free_space:      true if there is a free/wild center space.
- description:     one plain-language sentence summarising the rules."""


# ─── Game Logic ───────────────────────────────────────────────────────────────
def _marked(card_df, r, c, called_set, rules):
    if rules["free_space"] and r == rules["free_space_row"] and c == rules["free_space_col"]:
        return True
    return str(card_df.iloc[r, c]).strip().upper() in called_set


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
        if all(m(i, i) for i in range(nrows)) or all(m(i, nrows - 1 - i) for i in range(nrows)):
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

    called_set  = {str(v).strip().upper() for v in called}
    nrows, ncols = card.shape

    border = "3px solid #ffd700" if is_winner else "2px solid #dee2e6"
    shadow = "0 0 22px rgba(255,215,0,0.45)" if is_winner else "0 2px 8px rgba(0,0,0,0.08)"
    bg     = "#fffdf0" if is_winner else "#ffffff"
    name_c = "#c8870a" if is_winner else "#343a40"
    prefix = "🏆 " if is_winner else ""

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

    # BINGO header for 5-column cards
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

    if st.session_state.rules_description:
        st.markdown(
            f'<div class="rules-box">📜 <strong>Active Rules</strong><br>'
            f'{st.session_state.rules_description}</div>',
            unsafe_allow_html=True,
        )

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
        st.session_state.called_values   = []
        st.session_state.winners         = set()
        st.session_state.last_warning    = None
        st.session_state.round          += 1
        st.rerun()

    if st.button("🆕 New Round — Upload New Cards", use_container_width=True):
        for key in ("called_values", "cards", "card_names", "card_thumbs"):
            st.session_state[key] = [] if key != "called_values" else []
        st.session_state.winners          = set()
        st.session_state.last_warning     = None
        st.session_state.rules_description = None
        st.session_state.round           += 1
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


# ─── Upload & Analysis ────────────────────────────────────────────────────────
with st.expander("📤 Upload & Analyze Images", expanded=not bool(st.session_state.cards)):

    # ── Rules image ──────────────────────────────────────────────────────────
    st.markdown("#### 📜 Step 1 — Rules Image *(optional)*")
    st.caption("Upload a photo or screenshot of the bingo rules. Claude will read and apply them automatically.")

    rules_col, rules_prev_col = st.columns([2, 1])
    with rules_col:
        rules_img = st.file_uploader(
            "Rules image",
            type=["png", "jpg", "jpeg", "webp"],
            key="rules_img_upload",
            label_visibility="collapsed",
        )
    with rules_prev_col:
        if rules_img:
            st.image(rules_img, caption="Rules preview", use_container_width=True)

    if rules_img:
        if st.button("🔍 Analyze Rules Image", use_container_width=True):
            client = require_client()
            rules_img.seek(0)
            b64, mt = file_to_b64(rules_img)
            with st.spinner("Claude is reading the rules…"):
                try:
                    raw    = call_vision(client, b64, mt, RULES_PROMPT, max_tokens=600)
                    parsed = json.loads(raw)
                    for key in ("check_rows", "check_cols", "check_diagonals",
                                "check_full_card", "free_space",
                                "free_space_row", "free_space_col"):
                        if key in parsed:
                            st.session_state.rules[key] = parsed[key]
                    st.session_state.rules_description = parsed.get("description", "")
                    recalc_winners()
                    st.success(f"✅ Rules applied: *{st.session_state.rules_description}*")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not parse rules image: {e}\n\nRaw response:\n```\n{raw}\n```")

    st.markdown("---")

    # ── Card images ──────────────────────────────────────────────────────────
    st.markdown("#### 🃏 Step 2 — Bingo Card Images")
    st.caption("Upload one image per card. Claude will extract the grid values from each one.")

    card_imgs = st.file_uploader(
        "Bingo card images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="card_img_upload",
        label_visibility="collapsed",
    )

    if card_imgs:
        st.markdown(f"**{len(card_imgs)} card image(s) ready:**")
        prev_cols = st.columns(min(len(card_imgs), 4))
        for i, f in enumerate(card_imgs):
            with prev_cols[i % 4]:
                st.image(f, caption=f.name, use_container_width=True)

        if st.button("🔍 Analyze & Add All Cards", type="primary", use_container_width=True):
            client   = require_client()
            progress = st.progress(0, text="Starting…")
            added, errors = 0, []

            for idx, f in enumerate(card_imgs):
                progress.progress(idx / len(card_imgs), text=f"Analyzing {f.name}…")
                try:
                    f.seek(0)
                    b64, mt = file_to_b64(f)
                    raw     = call_vision(client, b64, mt, CARD_PROMPT, max_tokens=1200)
                    parsed  = json.loads(raw)

                    grid = parsed.get("grid", [])
                    if not grid or not grid[0]:
                        raise ValueError("Model returned an empty grid.")

                    df    = pd.DataFrame(grid).astype(str).applymap(str.strip)
                    thumb = make_data_url(b64, mt)
                    name  = f.name.rsplit(".", 1)[0]

                    # Use free-space position detected in card if no rules image was used
                    if parsed.get("has_free_space") and not st.session_state.rules_description:
                        st.session_state.rules["free_space"]     = True
                        st.session_state.rules["free_space_row"] = parsed.get("free_space_row", 2)
                        st.session_state.rules["free_space_col"] = parsed.get("free_space_col", 2)

                    st.session_state.cards.append(df)
                    st.session_state.card_names.append(name)
                    st.session_state.card_thumbs.append(thumb)
                    added += 1

                except Exception as e:
                    errors.append(f"**{f.name}**: {e}")

            progress.progress(1.0, text="Done!")
            recalc_winners()

            if added:
                st.success(f"✅ {added} card(s) added!")
            for err in errors:
                st.error(err)
            if added:
                st.rerun()


# ─── Game Area ────────────────────────────────────────────────────────────────
if not st.session_state.cards:
    st.info(
        "👆 Upload your bingo card images above and click **Analyze & Add All Cards**.\n\n"
        "Optionally upload a rules image first so Claude can configure the winning conditions automatically."
    )
    st.stop()

# — Rules summary —
if st.session_state.rules_description:
    st.markdown(
        f'<div class="rules-box">📜 {st.session_state.rules_description}</div>',
        unsafe_allow_html=True,
    )

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
