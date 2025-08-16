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
    # enforce max 80 chars, prefer ~70
    seen = set()
    out: List[str] = []
    for t in cands:
        tt = _truncate_thai(t, 80)
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    # keep exactly 3 options (pad if needed)
    while len(out) < 3:
        out.append(_truncate_thai(f"{main_keyword} พรีเซลคึกคัก ดันความสนใจเพิ่มขึ้น", 80))
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
    """Use OpenAI to generate titles, meta descriptions, and slug based on content. Returns None on failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    content_text = _extract_text_for_llm(html)

    system = (
        "You are an expert Thai crypto news editor. Generate SEO options based strictly on the provided article content."
    )
    seed_line = f"English headline seed (optional): {english_headline_seed}\n\n" if english_headline_seed else ""
    user = (
        "Generate JSON with keys: titles (3 Thai options, each <=80 chars, ~70 ideal, click-worthy, news-style), "
        "meta_descriptions (3 Thai options, each <=155 chars, complete sentences, no trailing '...'), "
        "and slug (English, lowercase, hyphenated, includes a concise English phrase reflecting the content).\n"
        f"Main keyword: {main_keyword}\n"
        f"Project name: {project_name}\n"
        f"Target site: {site_key}\n\n"
        f"{seed_line}"
        f"Article content (Thai/HTML excerpt converted to text):\n{content_text}\n\n"
        "Rules:\n"
        "- Titles must be derived from the content (not templates) and read like a Thai news headline.\n"
        "- Prefer strong verbs, concrete facts, entity-first structure. Avoid clickbait and vague phrasing.\n"
        "- Include concrete numbers/names from the content when available.\n"
        "- No clickbait, no incomplete phrases.\n"
        "- Meta descriptions must summarize the article concisely; <=155 chars.\n"
        "- Slug must be English, concise, hyphenated, and reflect the actual article topic.\n"
        "Respond ONLY with a compact JSON object."
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.5,
            max_tokens=600,
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
        titles = [_truncate_thai(t, 80) for t in titles][:3]
        metas = [_truncate_thai(m, 155) for m in metas][:3]
        slug = _slugify_english(slug) or _slugify_english(f"{project_name}-{main_keyword}")
        if titles and metas and slug:
            return {"titles": titles, "meta_descriptions": metas, "slug": slug}
        return None
    except Exception:
        return None


def _llm_rewrite_html_to_thai(html: str, main_keyword: str, project_name: str, site_key: str) -> Optional[str]:
    """Rewrite content to Thai, news-style, WordPress-ready HTML while PRESERVING detail and hyperlinks.
    Output may use: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>, <blockquote>, <br>.
    Returns None on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    # Provide trimmed HTML so the model can preserve anchor labels/URLs and structure
    safe_html = ensure_allowed_tags(html)
    content_html = safe_html[:35000]
    system = "You are a professional Thai crypto journalist and editor."
    user = (
        "Rewrite the following ARTICLE HTML into Thai, news style, preserving details and all meaningful hyperlinks.\n"
        "Output clean WordPress-ready HTML using ONLY: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>, <blockquote>, <br>.\n"
        "DO NOT include <style>, <script>, <head>, <link>, <meta>, or inline CSS. Keep paragraph granularity similar (do NOT compress multiple sentences into one short line).\n"
        "Preserve <a href> URLs and anchor text where present; do not invent new links.\n"
        f"Main keyword: {main_keyword}\nProject: {project_name}\nSite: {site_key}\n\n"
        "ARTICLE HTML TO REWRITE:\n" + content_html
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=4500,
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
        rewritten = _llm_rewrite_html_to_thai(html, main_keyword, project_name, site_key)
        if rewritten:
            html = ensure_allowed_tags(rewritten)

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
        slug = _slugify_english(f"{project_name}-{main_keyword}-press-release")

    return {
        "titles": titles,
        "meta_descriptions": metas,
        "slug": slug,
        "html": html,
    }
