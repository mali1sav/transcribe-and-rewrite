from typing import Dict, Any, List

try:
    from jinja2 import Environment, BaseLoader, select_autoescape
except Exception:  # pragma: no cover
    Environment = None  # type: ignore


def _get_env():
    if Environment is None:
        raise RuntimeError("Jinja2 is required to render CTA templates. Please install jinja2.")
    return Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(disabled_extensions=(".html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _common_context(row: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize None/NaN to empty string for Jinja2 truthy checks
    safe = {k: ("" if v is None else v) for k, v in row.items()}
    return safe


# -----------------------
# Template registry
# -----------------------
TEMPLATES: Dict[str, Dict[str, str]] = {
    "ICOBench": {
        "icobench_v1": """
<section>
  <p>
    ต้องการเริ่มต้นกับ <strong>{{ project_name }}</strong>? อ่าน
    {% if how_to_buy_url %}<a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">วิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} และ <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">คาดการณ์ราคา</a>{% endif %}.
  </p>

  <p>เข้าเว็บทางการได้ที่ปุ่มด้านล่าง:</p>
  {{ su_button_url | safe }}

  <p>
    ติดตามประกาศล่าสุดทาง
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X (Twitter)</a>{% endif %}
    {% if x_url and telegram_url %} และ {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
</section>
""",
        "icobench_v2": """
<section>
  <p>
    <strong>{{ project_name }}</strong> กำลังได้รับความสนใจ —
    {% if how_to_buy_url %}ดู <a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">วิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} และ <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">แนวโน้มราคา</a>{% endif %}
    เพื่อประกอบการตัดสินใจอย่างรอบคอบ
  </p>

  {{ su_button_url | safe }}

  <p>
    ข่าวสารเรียลไทม์:
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
    {% if x_url and telegram_url %} · {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
</section>
""",
        "icobench_v3": """
<section>
  <p>
    ศึกษา <strong>{{ project_name }}</strong> ให้ครบ:
    {% if how_to_buy_url %}<a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">คู่มือวิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} + <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">บทวิเคราะห์ราคา</a>{% endif %}
  </p>

  {% if add_fomo_note %}
  [su_note note_color="#EEF7C0"]
  โอกาสมีจำกัด—อย่าพลาดรอบพรีเซลหากคุณเชื่อมั่นในศักยภาพของโปรเจกต์นี้
  [/su_note]
  {% endif %}

  {{ su_button_url | safe }}

  <p>
    อัปเดตล่าสุด:
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
    {% if x_url and telegram_url %} | {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
</section>
""",
        "icobench_v4": """
<section>
  <p>
    ยังตัดสินใจไม่ได้?
    {% if review_url %}อ่าน <a href="{{ review_url }}" target="_blank" rel="noopener">รีวิวเชิงลึกของ {{ project_name }}</a>{% endif %}
    {% if price_prediction_url %} และ <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">มุมมองราคา</a>{% endif %}
    ก่อนดำเนินการ
  </p>

  <p>เข้าหน้าโครงการอย่างเป็นทางการ:</p>
  {{ su_button_url | safe }}

  <p>
    คอมมูนิตี้:
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
    {% if telegram_url and x_url %} · {% endif %}
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
  </p>
</section>
""",
        "icobench_v5": """
<section>
  <p>
    <strong>{{ project_name }}</strong> — ลองดูข้อมูลสำคัญ:
    {% if how_to_buy_url %}<a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">วิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} / <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">คาดการณ์ราคา</a>{% endif %}
    {% if review_url %} / <a href="{{ review_url }}" target="_blank" rel="noopener">รีวิว</a>{% endif %}
  </p>

  {{ su_button_url | safe }}

  <p>
    ติดตามข่าวสาร:
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
    {% if x_url and telegram_url %} &middot; {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
</section>
""",
    },
    # Default pack for Bitcoinist, Cryptonews, CryptoDnes
    "DEFAULT": {
        "default_v1": """
<section>
  <p>
    ติดตาม {{ project_name }}:
    {% if how_to_buy_url %}<a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">วิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} · <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">คาดการณ์ราคา</a>{% endif %}
    {% if review_url %} · <a href="{{ review_url }}" target="_blank" rel="noopener">รีวิว</a>{% endif %}
  </p>
  {{ su_button_url | safe }}
  <p>
    {% if official_url %}<a href="{{ official_url }}" target="_blank" rel="noopener">เว็บไซต์ทางการ</a>{% endif %}
  </p>
  <p>
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
    {% if x_url and telegram_url %} | {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
</section>
""",
        "default_v2": """
<section>
  <p>
    ต้องการเริ่มกับ <strong>{{ project_name }}</strong>? {% if review_url %}อ่าน <a href="{{ review_url }}" target="_blank" rel="noopener">รีวิว</a>{% endif %}
  </p>
  {{ su_button_url | safe }}
  <p>
    {% if official_url %}หรือไปยัง <a href="{{ official_url }}" target="_blank" rel="noopener">เว็บไซต์ทางการ</a>{% endif %}
  </p>
</section>
""",
        "default_v3": """
<section>
  <p>
    {{ project_name }} — ลิงก์สำคัญ:
    {% if how_to_buy_url %}<a href="{{ how_to_buy_url }}" target="_blank" rel="noopener">วิธีซื้อ</a>{% endif %}
    {% if price_prediction_url %} / <a href="{{ price_prediction_url }}" target="_blank" rel="noopener">คาดการณ์ราคา</a>{% endif %}
    {% if review_url %} / <a href="{{ review_url }}" target="_blank" rel="noopener">รีวิว</a>{% endif %}
  </p>
  {{ su_button_url | safe }}
</section>
""",
        "default_v4": """
<section>
  <p>
    เข้าร่วมคอมมูนิตี้ของ {{ project_name }}:
    {% if x_url %}<a href="{{ x_url }}" target="_blank" rel="noopener">X</a>{% endif %}
    {% if x_url and telegram_url %} &middot; {% endif %}
    {% if telegram_url %}<a href="{{ telegram_url }}" target="_blank" rel="noopener">Telegram</a>{% endif %}
  </p>
  {{ su_button_url | safe }}
</section>
""",
        "default_v5": """
<section>
  <p>
    ข้อมูลอย่างเป็นทางการของ {{ project_name }}:
    {% if official_url %}<a href="{{ official_url }}" target="_blank" rel="noopener">เว็บไซต์ทางการ</a>{% endif %}
  </p>
  {{ su_button_url | safe }}
</section>
""",
    },
}


def render_templates(site_key: str, row: Dict[str, Any], add_fomo_note: bool = False) -> List[Dict[str, str]]:
    env = _get_env()
    pack_key = "ICOBench" if site_key.lower() == "icobench" else "DEFAULT"
    templates = TEMPLATES.get(pack_key, TEMPLATES["DEFAULT"])  # fallback safe
    results: List[Dict[str, str]] = []
    context = _common_context(row)
    context["add_fomo_note"] = add_fomo_note

    for key, tpl in templates.items():
        template = env.from_string(tpl)
        html = template.render(**context)
        # Strip empty anchors/paragraphs potentially introduced by missing fields
        results.append({"id": key, "html": html})
    return results
