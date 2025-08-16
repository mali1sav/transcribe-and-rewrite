import os
import sys
import io
import csv
from typing import Optional

# Ensure project root is on sys.path so 'press_release' package is importable when run via Streamlit
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import requests

from press_release.press_release_llm import generate_press_release_outputs
from press_release.html_utils import (
    ensure_allowed_tags,
    split_sections_by_headings,
    find_promotional_section_index,
    insert_su_note_into_section,
    join_sections_to_html,
    build_cta_paragraphs,
)

# Reuse uploader utilities if available
try:
    # Prefer the more complete uploader from search2 if present
    from search2.search2 import submit_article_to_wordpress as submit_article_to_wp
except Exception:
    try:
        from wordpress_uploader import submit_article_to_wordpress as submit_article_to_wp
    except Exception:
        submit_article_to_wp = None  # type: ignore

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
    main_keyword = st.text_input("Main Keyword (Thai or English)", "Best Wallet")
    project_name = st.text_input("Project Name", "Best Wallet")
    site_key = st.selectbox("Target Site Key", [
        "ICOBENCH", "CRYPTONEWS", "BITCOINIST", "INSIDEBITCOINS", "CRYPTODNES", "OTHERS"
    ], index=2)
    date_string = st.text_input("Date string to enforce when mentioned", "ณ วันที่ 15 สิงหาคม 2025")
    english_headline_seed = st.text_input("Optional English headline seed (used as a base)", "")
    rewrite_to_thai = st.checkbox("Rewrite content to Thai news style (recommended)", value=True)

    st.markdown("---")
    st.subheader("CTA Google Sheet")
    gsheet_url = st.text_input(
        "Google Sheet URL (Anyone with link can view)",
        key="gsheet_url",
        placeholder="https://docs.google.com/spreadsheets/d/<id>/edit?gid=0",
    )
    # Auto-fetch when URL changes
    def _extract_sheet_id_gid(url: str):
        import re
        m_id = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        m_gid = re.search(r"[?&]gid=(\d+)", url)
        return (m_id.group(1) if m_id else None, m_gid.group(1) if m_gid else "0")

    last_url = st.session_state.get("_last_gsheet_url", "")
    if gsheet_url and gsheet_url != last_url:
        sid, gid = _extract_sheet_id_gid(gsheet_url.strip()) if gsheet_url.strip() else (None, None)
        if not sid:
            st.error("Invalid Google Sheet URL. Use .../spreadsheets/d/<id>/edit?gid=0")
        else:
            export_url = f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid or '0'}"
            try:
                r = requests.get(export_url, timeout=20)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                st.session_state["cta_df"] = df
                st.session_state["_last_gsheet_url"] = gsheet_url
                st.success("Loaded CTA data from Google Sheet.")
            except Exception as e:
                st.error(f"Failed to load Google Sheet CSV: {e}")

    st.markdown("---")
    st.subheader("Manual CTA Fallback")
    st.caption("If the Google Sheet doesn't match, we will insert this block instead.")
    st.text_area(
        "Manual CTA HTML or shortcode (e.g., [su_button ...])",
        key="manual_cta_html",
        height=120,
        placeholder="<p>สนใจรายละเอียดเพิ่มเติม คลิกปุ่มด้านล่าง</p>\n[su_button url=\"https://example.com\" background=\"#ffd600\" color=\"#000\"]สมัครตอนนี้[/su_button]",
    )

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

st.subheader("2) Optional su_note content (placed mid promotional section)")
su_note_input = st.text_area("su_note content (omit brackets) — keep concise and time-sensitive", placeholder="เช็คเวลาโบนัสรอบปัจจุบันก่อนพลาดโอกาส!")

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
        date_string=date_string,
        english_headline_seed=english_headline_seed.strip() or None,
        rewrite_to_thai=rewrite_to_thai,
    )

    html_clean = ensure_allowed_tags(outputs["html"])  # safety pass

    # Split sections and place su_note inside promotional section
    sections = split_sections_by_headings(html_clean)

    # Build su_note shortcode if provided
    su_note_html = ""
    if su_note_input.strip():
        body = su_note_input.strip()
        su_note_html = f"[su_note note_color=\"#ffffff\" text_color=\"#000000\" radius=\"0\"]{body}[/su_note]"

    promo_idx = find_promotional_section_index(sections, project_name=project_name, main_keyword=main_keyword)
    if su_note_html:
        insert_su_note_into_section(sections[promo_idx], su_note_html)

    # CTA from Google Sheet or manual fallback
    cta_df = st.session_state.get("cta_df")
    cta_row = find_cta_row(cta_df, project_name, site_key) if cta_df is not None else None

    # Decide CTA block
    cta_block = None
    if cta_row:
        cta_block = build_cta_paragraphs(project_name=project_name, site_key=site_key, cta_row=cta_row)
    else:
        manual_cta_html = st.session_state.get("manual_cta_html", "").strip()
        if manual_cta_html:
            cta_block = manual_cta_html

    # Append CTA paragraphs/button near end of promotional section if available
    if cta_block:
        from bs4 import BeautifulSoup
        promo_soup = BeautifulSoup(str(sections[promo_idx]), "html.parser")
        promo_soup.append(BeautifulSoup(cta_block, "html.parser"))
        sections[promo_idx].clear()
        for child in promo_soup.div.children if promo_soup.div else promo_soup.children:
            sections[promo_idx].append(child)

    final_html = ensure_allowed_tags(join_sections_to_html(sections))

    st.session_state.pr_outputs = {
        "titles": outputs["titles"],
        "meta_descriptions": outputs["meta_descriptions"],
        "slug": outputs["slug"],
        "html": final_html,
    }

    st.success("Generated! Review SEO options and HTML below.")

if "pr_outputs" in st.session_state:
    st.markdown("---")
    st.subheader("3) Choose SEO title and meta description")
    t1, t2 = st.columns(2)
    with t1:
        chosen_title = st.selectbox("Pick one title", st.session_state.pr_outputs["titles"], index=0)
    with t2:
        chosen_meta = st.selectbox("Pick one meta description", st.session_state.pr_outputs["meta_descriptions"], index=0)

    st.text_input("Slug (English, includes main keyword)", value=st.session_state.pr_outputs["slug"], key="slug_input")

    st.subheader("4) HTML Preview (clean, WordPress Text mode)")
    st.code(st.session_state.pr_outputs["html"], language="html")

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
