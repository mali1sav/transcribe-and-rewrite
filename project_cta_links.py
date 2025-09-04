"""
Project-specific CTA links for promotional content (paragraph format)
"""

import csv
import os
from functools import lru_cache
from typing import Optional, Dict, Tuple


def _build_paragraph_cta(project: str, links: Dict[str, str]) -> str:
    """Constructs two CTA paragraphs using available links.
    Keys used if present:
      - how_to_buy, price_prediction, review
      - website, x, telegram
      - su_button (HTML shortcode/button markup to append at the end)
    Missing items are simply omitted from the paragraph text.
    """
    # Paragraph 1: prefer price prediction/review first, then how-to-buy
    price_url = links.get("price_prediction")
    review_url = links.get("review")
    how_url = links.get("how_to_buy")

    # Build the first segment: price prediction or review
    if price_url:
        first_seg = f'<a href="{price_url}">บทวิเคราะห์ราคา {project} </a>'
    elif review_url:
        first_seg = f'<a href="{review_url}">บทวิเคราะห์/รีวิว {project} </a>'
    else:
        first_seg = f'บทวิเคราะห์ราคา {project} '

    # Build the second segment: how to buy
    if how_url:
        second_seg = f'หรือศึกษา<a href="{how_url}">วิธีซื้อ {project}</a> ด้วยขั้นตอนง่ายๆ'
    else:
        second_seg = f'หรือศึกษา วิธีซื้อ {project} ด้วยขั้นตอนง่ายๆ'

    # Compose full sentence with leading recommendation phrase and double space before "เพื่อ"
    p1 = f"<p>หากคุณสนใจ แนะนำให้อ่าน{first_seg}{second_seg}  เพื่อประกอบการตัดสินใจอย่างรอบคอบ</p>"

    # Paragraph 2: official site / X / Telegram
    p2_bits = []
    site_txt = f'เว็บไซต์ทางการของ {project}'
    x_txt = 'X'
    tg_txt = 'ช่อง Telegram'
    if links.get("website"):
        site_txt = f'<a href="{links["website"]}">{site_txt}</a>'
    if links.get("x"):
        x_txt = f'<a href="{links["x"]}">{x_txt}</a>'
    if links.get("telegram"):
        tg_txt = f'<a href="{links["telegram"]}">{tg_txt}</a>'
    p2 = f"<p>ติดตามข้อมูลเพิ่มเติมได้ที่{site_txt} หรือติดตามใน {x_txt} และ {tg_txt}</p>"

    # Optionally append a shortcode/button block at the end (e.g., [su_button] or custom HTML)
    button_html = links.get("su_button")
    if button_html:
        return p1 + "\n" + p2 + "\n" + button_html
    return p1 + "\n" + p2


CSV_FILENAME = "CTAs - Sheet1 - CTAs - Sheet1.csv (3).csv"


@lru_cache(maxsize=1)
def _load_cta_map() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Load CTA mappings from the CSV into a dict keyed by (project_name, site_key).
    Treat 'N/A' (any case) and empty strings as missing.
    Columns (minimum expected): project_name,site_key,how_to_buy_url,price_prediction_url,official_url,x_url,telegram_url,su_button_url,review_url
    """
    path = os.path.join(os.path.dirname(__file__), CSV_FILENAME)
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not os.path.exists(path):
        return mapping
    # Use utf-8-sig to tolerate BOM from Excel/Sheets exports
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            project = (row.get("project_name") or "").strip()
            site_key = (row.get("site_key") or "").strip()
            if not project or not site_key:
                continue
            def norm(v: Optional[str]) -> Optional[str]:
                if v is None:
                    return None
                v = v.strip()
                if not v or v.lower() == 'n/a':
                    return None
                return v
            def _fix_mojibake(text: Optional[str]) -> Optional[str]:
                """Attempt to repair common UTF-8 -> Latin-1 mojibake for Thai.
                Only applied when we detect mojibake patterns like 'à¹' and no Thai range characters.
                """
                if not text:
                    return text
                try:
                    # If there are Thai code points already, return as-is
                    if any('\u0e00' <= ch <= '\u0e7f' for ch in text):
                        return text
                    # Heuristic: mojibake often contains these sequences
                    if 'à¹' in text or 'à¸' in text:
                        repaired = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                        # If repair produced Thai, use it
                        if any('\u0e00' <= ch <= '\u0e7f' for ch in repaired):
                            return repaired
                except Exception:
                    pass
                return text
            links = {
                "how_to_buy": norm(row.get("how_to_buy_url")),
                "price_prediction": norm(row.get("price_prediction_url")),
                # optional review URL if present in CSV
                "review": norm(row.get("review_url")),
                "website": norm(row.get("official_url")),
                "x": norm(row.get("x_url")),
                "telegram": norm(row.get("telegram_url")),
                # optional HTML shortcode/button block to append at the end
                "su_button": _fix_mojibake(norm(row.get("su_button_url"))),
            }
            mapping[(project, site_key)] = links
    return mapping


def get_project_cta_links(project_name: str, site_key: str) -> Optional[str]:
    """
    Returns CTA paragraphs for a given project and site_key using CSV-driven mapping.

    Args:
        project_name: Name of the project/token (case-sensitive to CSV)
        site_key: Source site selected in sidebar (e.g., 'Bitcoinist', 'Cryptonews')

    Returns:
        HTML paragraph string (two <p> elements) or None if not found
    """
    mapping = _load_cta_map()
    links = mapping.get((project_name, site_key))
    if not links:
        return None
    return _build_paragraph_cta(project_name, links)


def reload_cta_cache() -> None:
    """Clear the CSV cache so latest CTA data is reloaded on next access."""
    try:
        _load_cta_map.cache_clear()
    except Exception:
        # Safe no-op on environments where cache state cannot be cleared
        pass