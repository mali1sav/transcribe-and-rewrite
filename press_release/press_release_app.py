import os
import sys
import io
import json
import csv
from typing import Optional

# Ensure project root is on sys.path so 'press_release' package is importable when run via Streamlit
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
import requests

from press_release.press_release_llm import (
    generate_press_release_outputs,
    llm_generate_thai_cta,
    llm_should_replace_last_section,
)
from press_release.html_utils import (
    ensure_allowed_tags,
    split_sections_by_headings,
    find_promotional_section_index,
    insert_su_note_into_section,
    join_sections_to_html,
    build_cta_paragraphs,
    prettify_html_for_display,
    make_copyable_html,
)
from press_release.cta_templates import render_templates

# Reuse uploader utilities if available
try:
    # Prefer the more complete uploader from search2 if present
    from search2.search2 import submit_article_to_wordpress as submit_article_to_wp
except Exception:
    try:
        from wordpress_uploader import submit_article_to_wordpress as submit_article_to_wp
    except Exception:
        submit_article_to_wp = None  # type: ignore

# Prefer centralized CTA links provider from search2 if available
try:
    from search2.project_cta_links import get_project_cta_links, reload_cta_cache  # type: ignore
except Exception:
    get_project_cta_links = None  # type: ignore
    reload_cta_cache = None  # type: ignore

load_dotenv()

st.set_page_config(page_title="Thai Crypto Press Release Generator", layout="wide")


def load_cta_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


def find_cta_row(df: pd.DataFrame, project_name: str, site_key: str) -> Optional[dict]:
    if df is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    pn = cols.get("project_name")
    sk = cols.get("site_key")
    if not pn or not sk:
        st.error("CSV must include 'project_name' and 'site_key' headers.")
        return None
    subset = df[(df[pn].astype(str).str.strip().str.lower() == project_name.strip().lower()) &
                (df[sk].astype(str).str.strip().str.upper() == site_key.strip().upper())]
    if subset.empty:
        st.warning("No CTA row matched this project + site. The CTA block will be omitted.")
        return None
    if len(subset) > 1:
        st.warning("Multiple CTA rows matched; using the first one.")
    return subset.iloc[0].to_dict()


def _extract_doc_id(google_doc_url: str) -> Optional[str]:
    import re
    patterns = [
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"/document/u/\d+/d/([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, google_doc_url)
        if m:
            return m.group(1)
    return None


def fetch_google_doc_html(google_doc_url: str) -> Optional[str]:
    """Fetch Google Doc as HTML via export endpoint. The doc must be viewable by 'Anyone with the link'."""
    try:
        # If it's a published-to-web URL, try direct get first
        if "pub" in google_doc_url or "publish" in google_doc_url:
            r = requests.get(google_doc_url, timeout=20)
            if r.ok and "<html" in r.text.lower():
                return r.text

        doc_id = _extract_doc_id(google_doc_url)
        if not doc_id:
            return None
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"
        r = requests.get(export_url, timeout=20)
        if r.ok:
            return r.text
        return None
    except Exception:
        return None


st.title("Thai Crypto Press Release Generator (Google Discover–optimized)")

with st.sidebar:
    st.header("Settings")
    main_keyword = st.text_input("Main Keyword (Thai or English)", value="เหรียญคริปโตที่น่าลงทุนวันนี้")
    project_name = st.text_input("Project Name", value="Bitcoin Hyper")
    site_key = st.selectbox("Target Site Key", [
        "ICOBENCH", "CRYPTONEWS", "BITCOINIST", "INSIDEBITCOINS", "CRYPTODNES", "OTHERS"
    ], index=0)
    english_headline_seed = st.text_input("Optional English headline seed (used as a base)", "")

    # CTA SHEET: Accept edit URL and convert to CSV export automatically
    CTA_EDIT_URL = "https://docs.google.com/spreadsheets/d/18LgIQv6m2XFIkl2Plxt2r3bHHvBP8c1PhUSY7-s2lOI/edit?gid=1826014060#gid=1826014060"
    def _to_csv_export(url: str) -> str:
        try:
            # turn /edit?gid=... to /export?format=csv&gid=...
            if "/edit" in url:
                base, rest = url.split("/edit", 1)
                # try to find gid
                gid = ""
                if "gid=" in url:
                    gid = url.split("gid=")[-1].split("&")[0].split("#")[0]
                if gid:
                    return f"{base}/export?format=csv&gid={gid}"
                return f"{base}/export?format=csv"
            return url
        except Exception:
            return url
    CTA_SHEET_CSV_URL = _to_csv_export(CTA_EDIT_URL)
    FIXED_CTA_SHEET_ID = "18LgIQv6m2XFIkl2Plxt2r3bHHvBP8c1PhUSY7-s2lOI"
    FIXED_CTA_GID = "1826014060"
    export_url = f"https://docs.google.com/spreadsheets/d/{FIXED_CTA_SHEET_ID}/export?format=csv&gid={FIXED_CTA_GID}"
    if st.session_state.get("_cta_src") != export_url:
        try:
            r = requests.get(export_url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            st.session_state["cta_df"] = df
            st.session_state["_cta_src"] = export_url
            st.success("Loaded CTA data from fixed Google Sheet.")
        except Exception as e:
            st.error(f"Failed to load fixed Google Sheet CSV: {e}")

    # Optional: allow reloading centralized CTA CSV cache when updated
    if reload_cta_cache is not None and st.button("Reload CTA CSV cache"):
        try:
            reload_cta_cache()
            st.success("CTA CSV cache cleared. It will be reloaded on next use.")
        except Exception as e:
            st.warning(f"Could not reload CTA CSV cache: {e}")

    # Centralized CSV status (for visibility when integrating project_cta_links)
    try:
        import importlib
        cta_mod = importlib.import_module("search2.project_cta_links")
        csv_name = getattr(cta_mod, "CSV_FILENAME", None)
        mod_file = getattr(cta_mod, "__file__", None)
        if csv_name and mod_file:
            csv_path = os.path.join(os.path.dirname(mod_file), csv_name)
            if os.path.exists(csv_path):
                mtime = os.path.getmtime(csv_path)
                from datetime import datetime
                st.caption(f"Centralized CTA CSV: {csv_path} (updated {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')})")
            else:
                st.caption(f"Centralized CTA CSV not found at: {csv_path}")
    except Exception:
        # Silent: this is only for operator visibility
        pass

    # Sidebar CTA controls removed for a cleaner UI; CTA is handled within the main content pipeline

st.subheader("1) Source content")
gd_col1, gd_col2 = st.columns([3,1])
with gd_col1:
    gdoc_url = st.text_input("Google Doc URL (Anyone with link can view)")
with gd_col2:
    if st.button("Fetch from Google Doc"):
        if not gdoc_url.strip():
            st.warning("Please paste a Google Doc URL.")
        else:
            html = fetch_google_doc_html(gdoc_url.strip())
            if html:
                st.session_state["source_content"] = html
                st.success("Fetched Google Doc HTML.")
            else:
                st.error("Failed to fetch. Ensure the Doc is shared as 'Anyone with the link can view'.")

docx_file = st.file_uploader("or Upload .docx (we will convert to HTML)", type=["docx"], key="docx_upl")
if docx_file is not None:
    try:
        try:
            import mammoth  # type: ignore
        except Exception:
            mammoth = None  # type: ignore
        if mammoth is None:
            st.error("Please install 'mammoth' to convert .docx to HTML: pip install mammoth")
        else:
            result = mammoth.convert_to_html(docx_file)
            html_from_docx = result.value  # The generated HTML
            st.session_state["source_content"] = html_from_docx
            st.success("Converted .docx to HTML.")
    except Exception as e:
        st.error(f".docx conversion failed: {e}")

source_content = st.text_area("Google Doc content (auto-filled if fetched)", height=300, key="source_content")

col_a, col_b = st.columns(2)
with col_a:
    generate_btn = st.button("Generate Press Release")
with col_b:
    upload_btn = st.button("Upload to WordPress", type="primary")

if generate_btn:
    if not source_content.strip():
        st.error("Please paste the source content.")
        st.stop()

    # Build initial outputs (structure-preserving sanitize + SEO options)
    outputs = generate_press_release_outputs(
        pasted_html_or_text=source_content,
        main_keyword=main_keyword,
        project_name=project_name,
        site_key=site_key,
        english_headline_seed=english_headline_seed.strip() or None,
        rewrite_to_thai=True,
    )

    html_clean = ensure_allowed_tags(outputs["html"])  # safety pass

    # Split sections
    sections = split_sections_by_headings(html_clean)

    # CTA from Google Sheet or manual fallback
    cta_df = st.session_state.get("cta_df")
    cta_row = find_cta_row(cta_df, project_name, site_key) if cta_df is not None else None

    # Decide CTA block
    cta_block = None
    # 1) Prefer project_cta_links CSV-driven paragraphs if available
    if get_project_cta_links is not None:
        try:
            # Try multiple case variants for project_name and site_key to improve match robustness
            csv_cta = None
            proj_variants = list(dict.fromkeys([
                project_name,
                str(project_name).strip(),
                str(project_name).title(),
                str(project_name).upper(),
                str(project_name).lower(),
            ]))
            site_variants = list(dict.fromkeys([
                site_key,
                str(site_key).title(),
                str(site_key).upper(),
                str(site_key).lower(),
            ]))
            for pj in proj_variants:
                for sk in site_variants:
                    csv_cta = get_project_cta_links(pj, sk)
                    if csv_cta:
                        break
                if csv_cta:
                    break
            if csv_cta:
                cta_block = csv_cta
                st.info("CTA source: Centralized CSV (project_cta_links)")
        except Exception as e:
            st.warning(f"Failed to get CTA from centralized CSV: {e}")
    # 2) Fallback to Google Sheet row + LLM/rules if CSV not available or missing
    if not cta_block and cta_row:
        try:
            cta_block = llm_generate_thai_cta(
                project_name=project_name,
                site_key=site_key,
                main_keyword=main_keyword,
                cta_row=cta_row,
            ) or build_cta_paragraphs(project_name=project_name, site_key=site_key, cta_row=cta_row)
            if cta_block:
                st.info("CTA source: Google Sheet + LLM/rules")
        except Exception as e:
            st.warning(f"Failed to build CTA from Google Sheet data: {e}")
    # 3) Final fallback: manual CTA HTML if user provided
    if not cta_block:
        manual_cta_html = st.session_state.get("manual_cta_html", "").strip()
        if manual_cta_html:
            cta_block = manual_cta_html
            st.info("CTA source: Manual HTML from session")

    # Decide whether to REPLACE the last section (if it's already a how-to-buy CTA) or APPEND a new final CTA
    if cta_block:
        from bs4 import BeautifulSoup

        # Build CTA wrapper once
        cta_soup = BeautifulSoup("<div data-section='1'></div>", "html.parser")
        wrapper = cta_soup.div
        h2 = cta_soup.new_tag("h2")
        h2.string = f"วิธีซื้อโทเคน {project_name}"
        wrapper.append(h2)
        wrapper.append(BeautifulSoup(cta_block, "html.parser"))

        decision = None
        try:
            # Evaluate using ORIGINAL content and include trailing paragraphs window (helps catch half-CTA endings)
            from bs4 import BeautifulSoup as _BS
            orig_html = ensure_allowed_tags(st.session_state.get("source_content", ""))
            orig_sections = split_sections_by_headings(orig_html)
            last_text = ""
            if orig_sections:
                # Last section text (original only)
                last_text = orig_sections[-1].get_text(" ", strip=True)
            # Also collect last few paragraphs from the whole document
            soup_all = _BS(orig_html, "html.parser")
            paras = soup_all.find_all("p")[-5:] if soup_all else []
            tail_paras = "\n".join(p.get_text(" ", strip=True) for p in paras if p)
            # Extract href hostnames from trailing paragraphs to preserve social/affiliate cues
            try:
                from urllib.parse import urlparse as _u
                links = []
                for p in paras:
                    for a in p.find_all("a"):
                        href = a.get("href") or ""
                        if href:
                            host = _u(href).netloc or href
                            if host and host not in links:
                                links.append(host)
                links_summary = ("\nLINKS: " + ", ".join(links)) if links else ""
            except Exception:
                links_summary = ""
            trailing = (last_text + "\n" + tail_paras + links_summary).strip()
            decision = llm_should_replace_last_section(
                last_section_text=trailing,
                project_name=project_name,
                main_keyword=main_keyword,
                site_key=site_key,
            )
        except Exception:
            decision = None

        # Replace only when LLM explicitly says so; otherwise append as its own section
        if (decision and decision.get("decision") == "replace_last"):
            sections[-1].clear()
            # Important: iterate over a static copy, not the live generator
            for child in list(wrapper.contents):
                sections[-1].append(child)
        else:
            # Default: append as a dedicated final section
            sections.append(wrapper)

    final_html = ensure_allowed_tags(join_sections_to_html(sections))
    # If we replaced the last section with standardized CTA, scrub any lingering weak-CTA tail paragraphs
    try:
        if (decision and decision.get("decision") == "replace_last"):
            from bs4 import BeautifulSoup as _BS_clean
            _soup = _BS_clean(final_html, "html.parser")
            cues_text = [
                "ซื้อ", "วิธีซื้อ", "เยี่ยมชมเว็บไซต์", "เว็บไซต์ทางการ", "เว็บไซต์พรีเซล", "ลงทะเบียน", "พรีเซล", "สมัคร",
                "ติดตาม", "คอมมูนิตี้", "X (Twitter)", "Telegram", "Instagram",
                "Buy", "How to buy", "Visit official", "Presale website", "Register"
            ]
            href_cues = ["/visit/", "transfer=", "?ref=", "utm_", "/en/staking", "/staking", "x.com", "instagram.com"]
            # Inspect only the last 8 paragraphs for safety
            tail_ps = _soup.find_all("p")[-8:]
            removed = False
            for p in list(tail_ps):
                txt = p.get_text(" ", strip=True) or ""
                hrefs = [a.get("href") or "" for a in p.find_all("a")]
                if any(c in txt for c in cues_text) or any(any(h in hrf for h in href_cues) for hrf in hrefs):
                    p.decompose()
                    removed = True
            if removed:
                final_html = str(_soup)
    except Exception:
        pass
    # Attempt to repair truncated/imbalanced HTML and warn user if a fix was applied
    try:
        from bs4 import BeautifulSoup as _BS_fix
        repaired = str(_BS_fix(final_html, "html.parser"))
        if repaired and repaired != final_html:
            st.warning("HTML appeared truncated or imbalanced. It has been auto-repaired.")
            final_html = repaired
    except Exception:
        pass

    # Normalize inline links across the whole document:
    # - Replace <br> inside <p> with spaces
    # - Merge paras that contain only links (optionally wrapped in formatting tags or separator-only text) into the previous paragraph
    try:
        from bs4 import BeautifulSoup as _BS_norm
        soup_norm = _BS_norm(final_html, "html.parser")
        # Replace <br> inside <p> with spaces to keep inline
        for p in soup_norm.find_all("p"):
            for br in p.find_all("br"):
                br.replace_with(" ")
        # Merge paragraphs that contain only anchors (possibly wrapped) or anchors with separator-only text
        def _is_anchor_wrapped(node):
            # Accept formatting wrappers that only contain anchors/whitespace
            if getattr(node, "name", None) in {"em", "strong", "span", "b", "i", "u", "small", "code"}:
                # if any non-anchor element exists inside, reject
                for ch in node.contents:
                    if isinstance(ch, str):
                        if ch.strip() and not re.match(r"^[\|\-•·,、/\\\\\s]+$", ch):
                            return False
                    else:
                        if getattr(ch, "name", None) != "a" and not _is_anchor_wrapped(ch):
                            return False
                return True
            return False

        import re as _norm_re
        re = _norm_re  # reuse name for inline checks above

        def _is_anchor_only_para(p_tag):
            has_anchor = False
            for node in p_tag.contents:
                if isinstance(node, str):
                    # allow only separators like | / • · , - and whitespace
                    if node.strip() and not re.match(r"^[\|\-•·,、/\\\\\s]+$", node):
                        return False
                else:
                    nm = getattr(node, "name", None)
                    if nm == "a":
                        has_anchor = True
                        continue
                    if _is_anchor_wrapped(node):
                        has_anchor = True
                        continue
                    # any other element -> not anchor-only
                    return False
            return has_anchor

        paragraphs = soup_norm.find_all("p")
        i = 1
        while i < len(paragraphs):
            cur = paragraphs[i]
            prev = paragraphs[i-1]
            if _is_anchor_only_para(cur):
                # Insert a space to ensure inline flow and move all anchor nodes
                prev.append(" ")
                for node in list(cur.contents):
                    prev.append(node)
                cur.decompose()
                paragraphs.pop(i)
                continue
            i += 1
        # Drop truly empty paragraphs (no text, no <img>, no <a>)
        for p in list(soup_norm.find_all("p")):
            if not (p.get_text(strip=True) or p.find("img") or p.find("a")):
                p.decompose()
        final_html = str(soup_norm)
    except Exception:
        pass

    # Remove bolding for main keyword: strip **...** markers and unwrap <strong> around the keyword
    try:
        import re as _re
        mk_txt = (main_keyword or "").strip()
        if mk_txt:
            # Remove markdown bold around the keyword specifically
            final_html = final_html.replace(f"**{mk_txt}**", mk_txt)
        # Remove any remaining **bold** markers globally (safe fallback)
        final_html = _re.sub(r"\*\*([^*]+)\*\*", r"\1", final_html)
        # Unwrap <strong> tags that contain the main keyword
        from bs4 import BeautifulSoup as _BS_unwrap
        soup_unwrap = _BS_unwrap(final_html, "html.parser")
        if mk_txt:
            for strong in soup_unwrap.find_all("strong"):
                if mk_txt in (strong.get_text() or ""):
                    strong.unwrap()
        final_html = str(soup_unwrap)
    except Exception:
        pass

    st.session_state.pr_outputs = {
        "titles": outputs["titles"],
        "meta_descriptions": outputs["meta_descriptions"],
        "slug": outputs["slug"],
        "html": final_html,
        "rewrite_attempted": outputs.get("rewrite_attempted", False),
        "rewrite_applied": outputs.get("rewrite_applied", False),
    }

    # Keyword coverage diagnostics (non-blocking)
    try:
        from bs4 import BeautifulSoup as _BS
        body_text = _BS(final_html, "html.parser").get_text(" ", strip=True)
        st.session_state["kw_body_count"] = body_text.count(main_keyword)
    except Exception:
        st.session_state["kw_body_count"] = None

    st.success("Generated! Review SEO options and HTML below.")

if "pr_outputs" in st.session_state:
    st.markdown("---")
    st.subheader("3) Choose SEO title and meta description")
    # Surface Thai rewrite status
    if st.session_state.pr_outputs.get("rewrite_attempted") and not st.session_state.pr_outputs.get("rewrite_applied"):
        st.warning("Thai rewrite was requested but not applied. Check OPENROUTER_API_KEY and model access, then try again.")
    t1, t2 = st.columns(2)
    with t1:
        # Minimal: choose first suggested title; allow edit
        title_opts = st.session_state.pr_outputs["titles"]
        chosen_title = title_opts[0] if title_opts else ""
        chosen_title = st.text_input("Final title", value=chosen_title, key="final_title_input")

    with t2:
        # Minimal: choose first suggested meta; allow edit
        meta_opts = st.session_state.pr_outputs["meta_descriptions"]
        chosen_meta = meta_opts[0] if meta_opts else ""
        chosen_meta = st.text_area("Final meta", value=chosen_meta, height=80, key="final_meta_input")

    # Enforce slug 7–10 words
    slug_val = st.session_state.pr_outputs["slug"]
    def _limit_words_slug(s: str, min_w: int = 7, max_w: int = 10) -> str:
        parts = [p for p in str(s).strip().lower().replace(" ", "-").split('-') if p]
        if len(parts) <= max_w:
            return '-'.join(parts)
        return '-'.join(parts[:max_w])
    def _word_count(s: str) -> int:
        return len([p for p in str(s).split('-') if p])

    slug_val = _limit_words_slug(slug_val)

    col_a, col_b = st.columns([3,1])
    with col_a:
        slug_val = st.text_input("Slug (English, descriptive, 7–10 words)", value=slug_val, key="slug_input")
    with col_b:
        def _expand_slug_from_title(slug: str, title_txt: str, kw: str) -> str:
            base = slug
            # Try to append keyword tokens if missing/short
            extra = []
            for token in str(kw).lower().split():
                if token and token not in base:
                    extra.append(token)
            base = (base + '-' + '-'.join(extra)).strip('-') if extra else base
            return _limit_words_slug(base)
        if st.button("Expand slug"):
            st.session_state["slug_input"] = _expand_slug_from_title(slug_val, st.session_state.get("final_title_input", ""), main_keyword)

    # Simplified: hide diagnostics/warnings for a cleaner UI

    st.subheader("4) Preview & HTML code")
    from streamlit import components
    html_render = st.session_state.pr_outputs["html"]
    # For copying/pasting into WordPress, use compact copy-safe HTML (no prettify line breaks)
    html_pretty = make_copyable_html(html_render)
    tab1, tab2 = st.tabs(["Rendered preview", "HTML (copyable)"])
    with tab1:
        components.v1.html(html_render, height=2000, scrolling=True)
    with tab2:
        st.code(html_pretty, language="html")
        st.download_button(
            label="Download HTML",
            data=html_pretty.encode("utf-8"),
            file_name=f"{st.session_state.get('slug_input', 'press-release')}.html",
            mime="text/html",
        )

    # Removed section 5 (CTA Generator UI) to keep the app focused and professional.

    if upload_btn:
        if submit_article_to_wp is None:
            st.error("WordPress uploader not found. Please ensure 'search2/search2.py' or 'wordpress_uploader.py' is available.")
            st.stop()

        wp_url = os.getenv("wp_url")
        wp_username = os.getenv("wp_username")
        wp_app_password = os.getenv("wp_app_password")
        if not all([wp_url, wp_username, wp_app_password]):
            st.error("Missing WordPress credentials in .env (wp_url, wp_username, wp_app_password)")
            st.stop()

        payload = {
            "title": st.session_state.get("title_final", chosen_title),
            "content": st.session_state.pr_outputs["html"],
            "slug": st.session_state.get("slug_input", st.session_state.pr_outputs["slug"]),
            "excerpt": chosen_meta,
            "meta_description": chosen_meta,
        }

        with st.spinner("Uploading draft to WordPress..."):
            try:
                result = submit_article_to_wp(payload, wp_url, wp_username, wp_app_password)
            except TypeError:
                # Some uploaders expect different signature; try search2 version
                try:
                    result = submit_article_to_wp(payload, wp_url, wp_username, wp_app_password, primary_keyword=main_keyword, site_name=site_key, content_type="post")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        if result:
            st.success("Uploaded as draft successfully!")
        else:
            st.error("Upload returned no result or failed.")
