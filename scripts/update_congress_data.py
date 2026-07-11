"""
Refreshes congress_module.json by checking each congress's official website/source_link
and re-extracting dates, deadlines, and status — grounded strictly in the fetched page
text (never from the model's prior knowledge) to avoid hallucinated dates.

Run manually:
    ANTHROPIC_API_KEY=sk-... python3 scripts/update_congress_data.py

Run automatically: see .github/workflows/update-congress-data.yml
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timezone

import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "congress_module.json")
MODEL = "claude-sonnet-5"
REQUEST_TIMEOUT = 15
FETCH_DELAY_SECONDS = 2  # be polite to congress sites and PubMed-style hosts

EXTRACT_FIELDS = [
    "start_date", "end_date", "date_note", "city", "venue",
    "registration_link", "abstract_deadline", "early_registration_deadline",
    "program_link", "fee_info", "cme_info",
]

EXTRACTION_SYSTEM_PROMPT = """You extract structured congress/conference metadata from a single \
webpage's text content. You MUST use only information explicitly present in the provided text. \
Never rely on prior/background knowledge about this congress. If a field is not stated in the \
text, return null for it — do not guess or infer from typical patterns.

Return ONLY a JSON object with exactly these keys (all optional, null if unknown):
start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), date_note (string, e.g. "Fall 2026" if no \
exact date given), city, venue, registration_link (URL), abstract_deadline (YYYY-MM-DD), \
early_registration_deadline (YYYY-MM-DD), program_link (URL), fee_info (short string), \
cme_info (short string).

No prose, no markdown fences — just the raw JSON object."""


def fetch_page_text(url: str) -> str | None:
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ENTireCongressBot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! fetch failed for {url}: {e}", file=sys.stderr)
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:12000]


def extract_fields(client: Anthropic, page_text: str) -> dict:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Webpage text:\n\n{page_text}"}],
    )
    text_block = next((b for b in msg.content if b.type == "text"), None)
    if text_block is None:
        print("  ! no text block in model response", file=sys.stderr)
        return {}
    raw = text_block.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ! could not parse model output as JSON: {raw[:200]}", file=sys.stderr)
        return {}
    return {k: v for k, v in data.items() if k in EXTRACT_FIELDS}


def merge_updates(congress: dict, updates: dict, today: str) -> bool:
    changed = False
    for key, new_value in updates.items():
        if new_value is None:
            continue
        old_value = congress.get(key)
        if new_value != old_value:
            congress[key] = new_value
            changed = True
    if changed:
        congress["last_checked"] = today
        if congress.get("verification_status") == "awaiting_dates" and congress.get("start_date"):
            congress["verification_status"] = "partially_verified"
    return changed


def mark_past_if_over(congress: dict, today: date) -> bool:
    end_str = congress.get("end_date") or congress.get("start_date")
    if not end_str:
        return False
    try:
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        return False
    if end < today and congress.get("verification_status") != "past":
        congress["verification_status"] = "past"
        return True
    return False


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    today_str = date.today().isoformat()
    today = date.today()
    any_changed = False

    for congress in data["congresses"]:
        url = congress.get("source_link") or congress.get("website")
        changed_this = mark_past_if_over(congress, today)

        if congress.get("verification_status") != "past" and url:
            print(f"Checking {congress['id']} -> {url}")
            page_text = fetch_page_text(url)
            if page_text:
                updates = extract_fields(client, page_text)
                if merge_updates(congress, updates, today_str):
                    changed_this = True
                    print(f"  updated: {updates}")
            time.sleep(FETCH_DELAY_SECONDS)

        any_changed = any_changed or changed_this

    if any_changed:
        data["module_info"]["last_updated"] = today_str
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("congress_module.json updated.")
    else:
        print("No changes detected.")


if __name__ == "__main__":
    main()
