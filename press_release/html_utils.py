import re
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urlparse, parse_qs, unquote, urlencode, urlunparse

ALLOWED_TAGS = {
    "p", "h2", "h3", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "img", "a",
    "strong", "em", "blockquote", "br"
}
ALLOWED_ATTRS = {
    "a": {"href", "title", "target", "rel"},
    "img": {"src", "alt", "width", "height"},
    # other tags: keep no attributes
}

SHORTCODE_PATTERN = re.compile(
    r"""
    \[
      (?P<name>su_note|su_button)   # shortcode name
      [^\]]*                        # attributes
    \]
    (?:                             # optional content + closing tag
      .*?
      \[/(?P=name)\]
    )?
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

MASK_TOKEN = "__SHORTCODE_MASK_{}__"


def mask_shortcodes(html: str) -> Tuple[str, List[str]]:
    masks: List[str] = []
    def _repl(match):
        masks.append(match.group(0))
        return MASK_TOKEN.format(len(masks)-1)
    return SHORTCODE_PATTERN.sub(_repl, html), masks


def unmask_shortcodes(html: str, masks: List[str]) -> str:
    for i, code in enumerate(masks):
        html = html.replace(MASK_TOKEN.format(i), code)
    return html


def sanitize_html_keep_structure(html: str) -> str:
    if not html:
        return ""
    masked, masks = mask_shortcodes(html)
    soup = BeautifulSoup(masked, "html.parser")

    # Normalize formatting tags
    for b in soup.find_all("b"):
        b.name = "strong"
    for i in soup.find_all("i"):
        i.name = "em"

    # Remove disallowed tags but keep their text/children
    # First, fully remove tags whose text content should not leak
    for bad in soup.find_all(["style", "script", "head", "link", "meta", "title", "noscript"]):
        bad.decompose()

    for tag in list(soup.find_all(True)):
        if tag.name not in ALLOWED_TAGS:
            # If it's a simple div with only text, convert to <p>
            if tag.name == "div":
                text = tag.get_text(" ", strip=False)
                p = soup.new_tag("p")
                p.string = text
                tag.replace_with(p)
                continue
            tag.unwrap()
            continue
        # Clean attributes
        allowed = ALLOWED_ATTRS.get(tag.name, set())
        for attr in list(tag.attrs.keys()):
            if attr not in allowed:
                del tag.attrs[attr]

        # A-tag safety
        if tag.name == "a" and tag.has_attr("href"):
            href = str(tag["href"]).strip()
            # Rewrite Google redirector to direct URL if present
            if href.startswith("https://www.google.com/url?"):
                try:
                    qs = parse_qs(urlparse(href).query)
                    q = qs.get("q", [""])[0]
                    if q:
                        direct = unquote(q)
                        if direct.startswith("http://") or direct.startswith("https://"):
                            tag["href"] = direct
                            href = direct
                except Exception:
                    pass
            # Ensure http(s)
            if not href.startswith("http://") and not href.startswith("https://"):
                # drop unsafe hrefs
                del tag.attrs["href"]
            else:
                # Strip unwanted query parameters like `referrer`
                try:
                    parsed = urlparse(href)
                    qd = parse_qs(parsed.query, keep_blank_values=True)
                    if "referrer" in qd:
                        qd.pop("referrer", None)
                        new_query = urlencode(qd, doseq=True)
                        new_href = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
                        tag["href"] = new_href
                except Exception:
                    pass

    # Drop CSS-like garbage paragraphs that sometimes get pasted from Google Docs
    css_markers = ["@import", ".lst-kix_", "{", "}", "themes.googleusercontent.com/fonts/css"]
    for p in list(soup.find_all("p")):
        txt = p.get_text(" ", strip=True)
        if txt and any(m in txt for m in css_markers):
            p.decompose()

    clean_html = str(soup)
    return unmask_shortcodes(clean_html, masks)


def split_sections_by_headings(html: str) -> List[Tag]:
    """
    Return a list of top-level section containers split by H2.
    Each section is a <div data-section> wrapper starting with its H2 (if any) and including following siblings until next H2.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Ensure no H1
    for h1 in soup.find_all("h1"):
        h1.name = "h2"

    body_children = list(soup.contents)
    # If soup has a single root like <html>, descend to body
    if soup.find("body"):
        body_children = list(soup.body.children)

    sections: List[Tag] = []
    current: Optional[Tag] = None

    def start_section(node: Tag):
        nonlocal current
        current = soup.new_tag("div")
        current["data-section"] = "1"
        current.append(node)
        sections.append(current)

    for node in body_children:
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if not text:
                continue
            # wrap stray text into <p>
            p = soup.new_tag("p")
            p.string = text
            node = p
        if isinstance(node, Tag):
            if node.name.lower() == "h2":
                start_section(node)
            else:
                if current is None:
                    # create a preface section
                    current = soup.new_tag("div")
                    current["data-section"] = "1"
                    sections.append(current)
                current.append(node)

    # Fallback: if nothing, create one section with entire HTML
    if not sections:
        wrapper = BeautifulSoup("<div data-section='1'></div>", "html.parser").div
        wrapper.append(BeautifulSoup(html, "html.parser"))
        sections = [wrapper]

    return sections


def find_promotional_section_index(sections: List[Tag], project_name: str, main_keyword: str) -> int:
    key_terms = [project_name.strip().lower(), main_keyword.strip().lower()]
    best_idx, best_score = 0, -1
    for i, sec in enumerate(sections):
        text = sec.get_text(" ", strip=True).lower()
        score = sum(1 for t in key_terms if t and t in text)
        # small boost if section mentions presale/bonus/official
        if any(w in text for w in ["presale", "พรีเซล", "โบนัส", "official", "เว็บไซต์ทางการ", "offering", "token sale"]):
            score += 1
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def insert_su_note_into_section(section: Tag, note_html: str) -> None:
    """Insert su_note after 2nd <p> if exists; else after first block."""
    if not note_html:
        return
    soup = section if isinstance(section, BeautifulSoup) else section
    ps = soup.find_all("p", recursive=False)
    insert_after = None
    if len(ps) >= 2:
        insert_after = ps[1]
    elif ps:
        insert_after = ps[0]
    else:
        first_block = next((child for child in section.children if isinstance(child, Tag)), None)
        insert_after = first_block

    if insert_after is not None:
        marker = BeautifulSoup(note_html, "html.parser")
        insert_after.insert_after(marker)
    else:
        section.append(BeautifulSoup(note_html, "html.parser"))


def join_sections_to_html(sections: List[Tag]) -> str:
    soup = BeautifulSoup("", "html.parser")
    for sec in sections:
        for child in list(sec.children):
            soup.append(child)
    return str(soup)


def ensure_allowed_tags(html: str) -> str:
    """Full sanitize pipeline while preserving shortcodes and structure."""
    return sanitize_html_keep_structure(html)


def prettify_html_for_display(html: str) -> str:
    """Return a multi-line, human-readable HTML for UI display and copying.
    Does NOT change the underlying structure; only formatting with indentation/newlines.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    # Use minimal formatter to avoid escaping quotes unnecessarily
    try:
        return soup.prettify(formatter="minimal")
    except Exception:
        return str(soup)


def make_copyable_html(html: str) -> str:
    """Return a compact, copy-safe HTML string with no prettify-induced line breaks.
    - Collapses excessive whitespace inside <p>, ensuring inline <a> stays in sentences.
    - Removes empty paragraphs.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    # Remove empty <p> tags and normalize whitespace inside paragraphs
    for p in list(soup.find_all("p")):
        # Drop paragraphs that have no text and no non-whitespace children
        has_text = (p.get_text(strip=True) or "") != ""
        has_elements = any(isinstance(ch, Tag) for ch in p.contents)
        if not has_text and not has_elements:
            p.decompose()
            continue
        # Collapse whitespace in text nodes to prevent line breaks between inline elements
        for node in list(p.descendants):
            if isinstance(node, NavigableString):
                compact = re.sub(r"\s+", " ", str(node))
                if compact != str(node):
                    node.replace_with(compact)
    # Ensure there are no stray newlines between inline anchors and text
    out = str(soup)
    # Remove newlines that are immediately inside paragraph boundaries
    out = re.sub(r"<p>\s+", "<p>", out)
    out = re.sub(r"\s+</p>", "</p>", out)
    return out


def build_cta_paragraphs(
    project_name: str,
    site_key: str,
    cta_row: dict,
    ticker: Optional[str] = None
) -> str:
    def _is_na(val: Optional[str]) -> bool:
        v = (val or "").strip().lower()
        return v in {"n/a", "na", "none", "null", "-", ""}

    def _is_http_url(val: str) -> bool:
        return val.startswith("http://") or val.startswith("https://")

    def _a(label: str, url_key: str) -> Optional[str]:
        url_raw = (cta_row.get(url_key) or "").strip()
        if _is_na(url_raw):
            return None
        if _is_http_url(url_raw):
            return f"<a href=\"{url_raw}\">{label}</a>"
        return None

    parts: List[str] = []
    price_pred = _a(f"บทวิเคราะห์อนาคตราคา {project_name}", "price_prediction_url")
    how_to_buy = _a(f"วิธีซื้อเหรียญ {project_name}{f' ({ticker})' if ticker else ''}", "how_to_buy_url")

    sentence_bits: List[str] = []
    if price_pred:
        sentence_bits.append(f"สามารถอ่าน{price_pred}")
    if how_to_buy:
        if sentence_bits:
            sentence_bits.append("หรือสามารถเรียนรู้" + how_to_buy)
        else:
            sentence_bits.append("สามารถเรียนรู้" + how_to_buy)

    paragraph = "สำหรับนักลงทุนที่สนใจ " + (" และ ".join(sentence_bits) if sentence_bits else "ติดตามความคืบหน้าได้ที่เว็บไซต์ทางการ")
    parts.append(f"<p>{paragraph}</p>")

    # su_button handling with N/A detection and prebuilt shortcode/HTML pass-through
    raw_su = (cta_row.get("su_button_url") or "").strip()
    raw_official = (cta_row.get("official_url") or "").strip()

    # If su_button_url contains a full shortcode or HTML, and is not N/A, use as-is
    su_lower = raw_su.lower()
    has_prebuilt = ("[su_button" in su_lower) or ("<a" in su_lower) or ("href=" in su_lower)
    if has_prebuilt and not _is_na(raw_su):
        parts.append(raw_su)
    else:
        # Choose a valid URL for shortcode: prefer su_button_url, else official_url
        chosen_url = None
        if not _is_na(raw_su) and _is_http_url(raw_su):
            chosen_url = raw_su
        elif not _is_na(raw_official) and _is_http_url(raw_official):
            chosen_url = raw_official

        if chosen_url:
            su_button_text = (cta_row.get("su_button_text") or f"เว็บไซต์ทางการ {project_name}").strip()
            if _is_na(su_button_text):
                su_button_text = f"เว็บไซต์ทางการ {project_name}"

            def _val(key: str, default: str) -> str:
                v = (cta_row.get(key) or "").strip()
                return default if _is_na(v) else v

            bg = _val("button_background", "#fabc2c")
            color = _val("button_color", "#000000")
            size = _val("button_size", "8")
            center = _val("button_center", "yes")
            radius = _val("button_radius", "0")
            rel = _val("button_rel", "sponsored noopener nofollow")

            button = (
                f"[su_button url=\"{chosen_url}\" target=\"_new\" style=\"flat\" "
                f"background=\"{bg}\" color=\"{color}\" size=\"{size}\" center=\"{center}\" radius=\"{radius}\" rel=\"{rel}\"]"
                f"{su_button_text}[/su_button]"
            )
            parts.append(button)
        # else: no valid URL -> skip button entirely

    return "\n".join(parts)
