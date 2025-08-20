import os
import re
import json
from typing import Dict, List, Tuple, Optional
import html as htmlmod

from press_release.html_utils import ensure_allowed_tags

# Prefer OpenAI if available
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


def _truncate_thai(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


def _slugify_english(text: str) -> str:
    # very simple slugify: lowercase, alnum and hyphens only
    t = re.sub(r"[^A-Za-z0-9\s-]", "", text or "").strip().lower()
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"-+", "-", t)
    return t


def _title_candidates_from_html(html: str, main_keyword: str) -> List[str]:
    # Naive fallback: derive from first <h2> or keyword variations
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    h2 = soup.find("h2")
    base = h2.get_text(strip=True) if h2 else main_keyword
    cands = [
        base,
        f"{main_keyword} ข่าวพรีเซลมาแรง ดึงดูดนักลงทุนรุ่นใหม่",
        f"อัปเดตพรีเซล {main_keyword} กระแสแรง น่าจับตา"
    ]
    # enforce max 65 chars to match UI constraint
    seen = set()
    out: List[str] = []
    for t in cands:
        tt = _truncate_thai(t, 65)
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    # keep exactly 3 options (pad if needed)
    while len(out) < 3:
        out.append(_truncate_thai(f"{main_keyword} พรีเซลคึกคัก ดันความสนใจเพิ่มขึ้น", 65))
    return out[:3]


def _meta_candidates_from_html(html: str, main_keyword: str) -> List[str]:
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    p = soup.find("p")
    base = p.get_text(" ", strip=True) if p else f"อัปเดตพรีเซล {main_keyword} สำหรับนักลงทุนไทย"
    # ensure keyword early and <=155 chars, end with full sentence
    m1 = _truncate_thai(f"{main_keyword}: {base}", 155)
    if not m1.endswith(".") and not m1.endswith("?") and not m1.endswith("!" ):
        m1 = m1.rstrip("… ")
        if len(m1) <= 152:
            m1 += "…"
    cands = [m1]
    alt1 = _truncate_thai(f"จับตา {main_keyword} พรีเซลมาแรง พร้อมรายละเอียดครบถ้วนสำหรับนักลงทุนไทย", 155)
    alt2 = _truncate_thai(f"{main_keyword} กำลังเป็นกระแส—สรุปทุกไฮไลต์พรีเซล และสิ่งที่นักลงทุนควรรู้", 155)
    return [cands[0], alt1, alt2]


def _extract_text_for_llm(html: str, max_chars: int = 4000) -> str:
    """Extract a concise text snapshot from HTML for LLM prompting."""
    import bs4
    soup = bs4.BeautifulSoup(html or "", "html.parser")
    # prefer headings + first paragraphs
    texts: List[str] = []
    for tag in soup.find_all(["h2", "h3", "p", "li"], limit=200):
        t = tag.get_text(" ", strip=True)
        if t:
            texts.append(t)
        if sum(len(x) for x in texts) > max_chars:
            break
    combined = "\n".join(texts)
    return combined[:max_chars]


def _llm_generate_seo(html: str, main_keyword: str, project_name: str, site_key: str, english_headline_seed: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Use OpenRouter to generate titles, meta descriptions, and slug based on content. Returns None on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or OpenAI is None:
        return None


def llm_should_replace_last_section(
    last_section_text: str,
    project_name: str,
    main_keyword: str,
    site_key: str,
) -> Optional[dict]:
    """Ask the LLM whether the LAST section of the original article is already a 'how to buy / CTA' section.
    Returns a dict like {"decision": "replace_last" | "append_new", "reason": str}. Returns None on failure.
    Avoids rigid heuristics; relies on model judgment based on semantics.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    system = "You are a careful Thai/English editor. Classify whether content is effectively a 'How to Buy / Call-to-Action' section. The content may be in ENGLISH or THAI; apply the same criteria in either language."
    user = (
        "Given the TRAILING content of a press release (usually the last section), decide if it functions as a 'how to buy / call-to-action' section.\n"
        "Say YES (choose replace_last) not only for explicit step-by-step guides, but ALSO for 'weak or half CTA' endings such as:\n"
        "- THAI examples: 'ซื้อ [PROJECT] ได้ที่นี่', 'อ่านวิธีซื้อ [PROJECT]', 'ไปยังเว็บไซต์ทางการ', 'เว็บไซต์พรีเซล', 'ลงทะเบียนพรีเซล', 'ติดตาม X/Telegram'.\n"
        "- ENGLISH examples: 'Buy [PROJECT] here', 'How to buy [PROJECT]', 'Visit the official website', 'Go to the presale website', 'Register for presale', 'Follow on X/Telegram/Instagram'.\n"
        "- Affiliate/redirect links (e.g., /visit/..., ?transfer=1, ?ref=, utm_...) pointing readers to buy/register/follow.\n"
        "If any of the above intent exists, respond that we should REPLACE the last section with our standardized CTA. Otherwise, APPEND a new CTA at the end.\n"
        "Return a compact JSON with keys: decision (replace_last|append_new) and reason (short).\n"
        f"Project: {project_name} | Keyword: {main_keyword} | Site: {site_key}\n"
        f"TRAILING TEXT (last section or tail paragraphs):\n{last_section_text[:4000]}\n"
        "Respond ONLY with JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content if resp and resp.choices else None
        if not raw:
            return None
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw)
        decision = data.get("decision", "append_new")
        reason = data.get("reason", "")
        if decision not in {"replace_last", "append_new"}:
            decision = "append_new"
        return {"decision": decision, "reason": str(reason)}
    except Exception:
        return None

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    content_text = _extract_text_for_llm(html)

    system = (
        "You are an expert Thai crypto news editor. Generate SEO options based strictly on the provided article content."
    )
    seed_line = f"English headline seed (mandatory for semantic alignment): {english_headline_seed}\n\n" if english_headline_seed else ""
    user = (
        "Generate JSON with keys: "
        "titles (3 THAI options, EACH <=65 characters, click-worthy, news-style; HOOK-FIRST: start with a short 3–5 word emotional/power Thai hook, THEN follow with the primary keyword and complete the thought; MUST include the MAIN KEYWORD; keep meaning aligned with the provided English seed), "
        "meta_descriptions (3 THAI options, each 145–155 chars, complete sentences, no trailing '...'), "
        "slug (English, lowercase, hyphenated, 7–10 WORDS, REFLECTS THE HEADLINE MEANING, and includes the main keyword if natural).\n"
        f"Main keyword: {main_keyword}\n"
        f"Project name: {project_name}\n"
        f"Target site: {site_key}\n\n"
        f"{seed_line}"
        f"Article content (Thai/HTML excerpt converted to text):\n{content_text}\n\n"
        "Rules:\n"
        "- Titles must be derived from the content (not templates) and read like a Thai news headline. Preserve the core meaning of the English seed (numbers, timeframe, claim).\n"
        "- The MAIN KEYWORD must appear in each title and in each meta description (prefer near the beginning of meta).\n"
        "- HOOK-FIRST: Begin each title with a natural 3–5 word emotional/power hook, then immediately mention the PRIMARY KEYWORD and complete the statement.\n"
        "- Include concrete numbers/names and timeframes from the content when available (e.g., '$1M in 17 days', 'target $2M next week').\n"
        "- No clickbait, no incomplete phrases.\n"
        "- Meta descriptions must summarize the article concisely; 145–155 chars ideal to avoid truncation.\n"
        "- Slug must be English, concise, hyphenated, 7–10 words, and reflect the actual article headline/topic.\n"
        "Respond ONLY with a compact JSON object."
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content if resp and resp.choices else None
        if not raw:
            return None
        # Attempt to parse JSON in the reply
        raw = raw.strip()
        # Remove possible code fences
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("````"):
            # extremely rare quadruple backtick; trim one and proceed
            raw = raw[1:]
        elif raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw.strip("`")
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw)
        titles = [str(t).strip() for t in data.get("titles", []) if str(t).strip()][:3]
        metas = [str(m).strip() for m in data.get("meta_descriptions", []) if str(m).strip()][:3]
        slug = str(data.get("slug", "")).strip()
        # Basic validations and truncations
        titles = [_truncate_thai(t, 65) for t in titles][:3]
        metas = [_truncate_thai(m, 155) for m in metas][:3]
        # Ensure every title contains the main keyword; if missing, append with a dash
        ensured_titles: List[str] = []
        for t in titles:
            if str(main_keyword) and str(main_keyword) not in t:
                t = f"{t} – {main_keyword}"
                t = _truncate_thai(t, 65)
                ensured_titles.append(t)
            else:
                ensured_titles.append(t)
        titles = ensured_titles[:3]

        # Slug: enforce 7–10 words, reflect headline meaning
        slug_clean = _slugify_english(slug)
        def _limit_words(s: str, min_w: int = 7, max_w: int = 10) -> str:
            parts = [p for p in s.split('-') if p]
            if len(parts) < min_w:
                return '-'.join(parts)
            return '-'.join(parts[:max_w])

        if not slug_clean:
            base = english_headline_seed or f"{project_name} {main_keyword} press release"
            slug_clean = _slugify_english(base)
        slug_clean = _limit_words(slug_clean)
        # If still too short, pad with main keyword tokens
        parts = [p for p in slug_clean.split('-') if p]
        if len(parts) < 7 and main_keyword:
            pad = _slugify_english(main_keyword).split('-')
            # ensure up to 7 min words
            need = 7 - len(parts)
            slug_clean = '-'.join(parts + pad[:max(0, need)])

        if titles and metas and slug_clean:
            return {"titles": titles, "meta_descriptions": metas, "slug": slug_clean}
        return None
    except Exception:
        return None


def _pick_url(cta_row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = str(cta_row.get(k) or "").strip()
        if v.lower() in {"", "n/a", "na", "none", "null", "-"}:
            continue
        if v.startswith("http://") or v.startswith("https://"):
            return v
    return None


def llm_generate_thai_cta(
    project_name: str,
    site_key: str,
    main_keyword: str,
    cta_row: dict,
) -> Optional[str]:
    """Generate a Thai CTA section with heading + paragraphs using OpenRouter.
    Must include correct hyperlinks derived from the CTA sheet: official, presale, price prediction, how to buy,
    and encourage using Best Wallet's Upcoming Token feature with download links when available in the sheet.
    Returns sanitized HTML or None on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Collect known URLs from the sheet
    official_url = _pick_url(cta_row, ["official_url", "website", "site", "homepage"])
    presale_url = _pick_url(cta_row, ["presale_url", "token_sale_url", "sale_url"]) or official_url
    price_pred_url = _pick_url(cta_row, ["price_prediction_url", "price_pred_url"]) or official_url
    how_to_buy_url = _pick_url(cta_row, ["how_to_buy_url", "guide_url"]) or official_url
    twitter_url = _pick_url(cta_row, ["twitter_url", "x_url"]) or None
    telegram_url = _pick_url(cta_row, ["telegram_url"]) or None
    audit_1_url = _pick_url(cta_row, ["audit_url", "audit_1_url"]) or None
    audit_2_url = _pick_url(cta_row, ["audit_2_url"]) or None

    # Optional Best Wallet app links if provided in the sheet
    bw_play_url = _pick_url(cta_row, ["best_wallet_google_play", "best_wallet_play_url", "google_play_url"]) or None
    bw_appstore_url = _pick_url(cta_row, ["best_wallet_app_store", "best_wallet_appstore_url", "app_store_url"]) or None

    # Optional prebuilt CTA shortcode/HTML
    su_button_raw = str(cta_row.get("su_button_url") or "").strip()

    # Build a compact JSON-like context for the model
    link_context = {
        "official": official_url,
        "presale": presale_url,
        "price_prediction": price_pred_url,
        "how_to_buy": how_to_buy_url,
        "twitter": twitter_url,
        "telegram": telegram_url,
        "audit_1": audit_1_url,
        "audit_2": audit_2_url,
        "best_wallet_google_play": bw_play_url,
        "best_wallet_app_store": bw_appstore_url,
        "su_button_raw": su_button_raw,
    }

    # System/User instructions
    system = "You are a Thai crypto copywriter creating compelling CTAs for press releases."
    user = (
        "Create a CTA block in THAI that will be used as the FINAL section of a press release. "
        "Use only these HTML tags: <h2>, <h3>, <p>, <a>, <strong>, <em>, <ul>, <li>, <br>. If a prebuilt [su_button] shortcode is provided, include it VERBATIM; otherwise use a normal <a> link.\n"
        "- Start with a short, clear CTA heading that includes the project name.\n"
        "- Write 2–4 paragraphs encouraging readers to: visit the official/presale website, read the price prediction article, and read the how-to-buy guide.\n"
        "- If X (Twitter) and/or Telegram links are provided, invite readers to follow those channels.\n"
        "- Explicitly mention that the token can be purchased via the Best Wallet app's 'Upcoming Token' feature. Encourage downloading Best Wallet to discover presale tokens early (only mention app store links if provided).\n"
        "- Insert hyperlinks STRICTLY from the URLs provided below. If a URL is missing, omit that specific link gracefully (do not invent).\n"
        "- Tone: professional, supportive, informative — avoid hype or promises (no 'รับประกัน', 'รวยเร็ว', etc.).\n"
        "- Keep link anchor texts natural in Thai, e.g., 'เว็บไซต์พรีเซลทางการ', 'บทวิเคราะห์ราคา', 'วิธีซื้อเหรียญ'.\n"
        "- STYLE EXAMPLE (use as guidance only; DO NOT copy; vary wording each time):\n"
        "  หากคุณสนใจ สามารถอ่านบทวิเคราะห์คาดการราคา [PROJECT] ว่าจะมีโอกาสพุ่งถึง 100x หรือไม่ หรืออ่านวิธีซื้อ [PROJECT] ซึ่งให้ขั้นตอนไว้อย่างละเอียด หรือติดตามความเคลื่อนไหวได้ที่ X (Twitter) และ Telegram\n"
        "  [su_button ... เยี่ยมชมเว็บไซต์ [PROJECT]]\n"
        f"Project name: {project_name}\n"
        f"Main keyword: {main_keyword}\n"
        f"Target site: {site_key}\n"
        f"Links (JSON): {json.dumps(link_context, ensure_ascii=False)}\n\n"
        "Output valid HTML only (no code fences)."
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.45,
            max_tokens=800,
        )
        out = resp.choices[0].message.content if resp and resp.choices else None
        if not out:
            return None
        out = out.strip()
        if out.startswith("```html"):
            out = out[7:]
        if out.endswith("```"):
            out = out[:-3]
        return ensure_allowed_tags(out.strip())
    except Exception:
        return None

def _llm_rewrite_html_to_thai(html: str, main_keyword: str, project_name: str, site_key: str) -> Optional[str]:
    """Rewrite content to Thai, news-style, WordPress-ready HTML while PRESERVING detail and hyperlinks.
    Output may use: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>, <blockquote>, <br>.
    Returns None on failure.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    # Provide trimmed HTML so the model can preserve anchor labels/URLs and structure
    safe_html = ensure_allowed_tags(html)
    content_html = safe_html[:35000]
    system = "You are a professional Thai crypto journalist and editor."
    user = (
        "Rewrite the following ARTICLE HTML into Thai, news style, preserving ALL sections, headings, paragraphs, bullet points, images, and hyperlinks. DO NOT shorten or omit sections.\n"
        "Output clean WordPress-ready HTML using ONLY: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>, <blockquote>, <br>, <img>.\n"
        "DO NOT include <style>, <script>, <head>, <link>, <meta>, or inline CSS. Keep paragraph granularity similar (do NOT compress multiple sentences into one short line).\n"
        "Preserve <a href> URLs and anchor text where present; do not invent new links. Keep YouTube links as plain URLs on their own line or as anchors.\n"
        f"Main keyword: {main_keyword}\nProject: {project_name}\nSite: {site_key}\n\n"
        "Keyword integration requirements (balance readability):\n"
        "- Integrate the MAIN KEYWORD naturally throughout the body with at least 10 total mentions across the entire article.\n"
        "- Avoid keyword stuffing: maximum 2 mentions of the exact keyword per paragraph; use synonyms/related terms elsewhere.\n"
        "- Preserve the essence of the English headline and the original content’s structure and details.\n\n"
        "ARTICLE HTML TO REWRITE:\n" + content_html
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.35,
            max_tokens=32768,
        )
        out = resp.choices[0].message.content if resp and resp.choices else None
        if not out:
            return None
        out = out.strip()
        if out.startswith("```html"):
            out = out[7:]
        if out.endswith("```"):
            out = out[:-3]
        return out.strip()
    except Exception:
        return None


def generate_press_release_outputs(
    pasted_html_or_text: str,
    main_keyword: str,
    project_name: str,
    site_key: str,
    date_string: str = "ณ วันที่ 13 สิงหาคม 2025",
    english_headline_seed: Optional[str] = None,
    rewrite_to_thai: bool = False,
) -> Dict[str, object]:
    """
    Lightweight generator that preserves structure, sanitizes HTML, and provides SEO options without external APIs.
    If input is plain text, it will wrap paragraphs.
    """
    html = pasted_html_or_text or ""
    rewrite_attempted = False
    rewrite_applied = False

    # If looks like plain text (no tags), convert double newlines to <p> and linkify URLs
    if "<" not in html and ">" not in html:
        def _linkify(text: str) -> str:
            # Basic URL linker for http(s) URLs
            url_re = re.compile(r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)")
            def repl(m):
                u = m.group(1)
                return f"<a href=\"{u}\" target=\"_blank\" rel=\"nofollow noopener\">{u}</a>"
            return url_re.sub(repl, text)

        paras = [x.strip() for x in html.split("\n\n") if x.strip()]
        esc_paras = [htmlmod.escape(p) for p in paras]
        linked = [_linkify(p) for p in esc_paras]
        parts = [f"<p>{p}</p>" for p in linked]
        html = "\n".join(parts)

    html = ensure_allowed_tags(html)

    # Optionally rewrite to Thai news-style HTML
    if rewrite_to_thai:
        rewrite_attempted = True
        rewritten = _llm_rewrite_html_to_thai(html, main_keyword, project_name, site_key)
        if rewritten:
            html = ensure_allowed_tags(rewritten)
            rewrite_applied = True

    # Try LLM-based generation first
    llm = _llm_generate_seo(html, main_keyword, project_name, site_key, english_headline_seed=english_headline_seed)
    if llm:
        titles = llm["titles"]  # type: ignore
        metas = llm["meta_descriptions"]  # type: ignore
        slug = llm["slug"]  # type: ignore
    else:
        # Fallback heuristics
        titles = _title_candidates_from_html(html, main_keyword)
        metas = _meta_candidates_from_html(html, main_keyword)
        # Derive slug from English headline seed when available, 7–10 words
        base = english_headline_seed or f"{project_name} {main_keyword} press release"
        slug_raw = _slugify_english(base)
        def _limit_words(s: str, min_w: int = 7, max_w: int = 10) -> str:
            parts = [p for p in s.split('-') if p]
            if len(parts) < min_w:
                return '-'.join(parts)
            return '-'.join(parts[:max_w])
        slug_limited = _limit_words(slug_raw)
        parts = [p for p in slug_limited.split('-') if p]
        if len(parts) < 7 and main_keyword:
            pad = _slugify_english(main_keyword).split('-')
            need = 7 - len(parts)
            slug_limited = '-'.join(parts + pad[:max(0, need)])
        slug = slug_limited

    return {
        "titles": titles,
        "meta_descriptions": metas,
        "slug": slug,
        "html": html,
        "rewrite_attempted": rewrite_attempted,
        "rewrite_applied": rewrite_applied,
    }
