#!/usr/bin/env python3
"""
get_tournament_ids_post.py

Scrape FIR tournament IDs via:
  POST https://fir.tournamentsoftware.com/find/tournament/DoSearch

Key: tournament IDs are often in href/onclick/data-* attributes, not as plain "/tournament/<GUID>" text.
So we parse the HTML and extract GUIDs from any attribute value that contains "tournament".
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup

BASE = "https://fir.tournamentsoftware.com"
FIND_URL = f"{BASE}/find"
DOSEARCH_URL = f"{BASE}/find/tournament/DoSearch"

FILTERS = {
    "StartDate": "2000-01-01",
    "EndDate": "2026-01-01",
    "DateFilterType": "0",
    "PostalCode": "12211",
}

HEADERS_GET = {
    "User-Agent": "RacketlonScraper/1.0 (+mailto:zainmagdon@gmail.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HEADERS_POST = {
    "User-Agent": "RacketlonScraper/1.0 (+mailto:zainmagdon@gmail.com)",
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": BASE,
    "Referer": FIND_URL,
}

# Generic GUID pattern
GUID_ONLY_RE = re.compile(
    r"\b([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\b"
)


def extract_form_fields(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    data = {}
    for inp in soup.select("input[name]"):
        name = inp.get("name")
        if name:
            data[name] = inp.get("value", "")
    return data


def extract_tournament_guids_from_fragment(html_fragment: str) -> set[str]:
    """
    Extract tournament GUIDs from HTML by looking specifically in:
      - href
      - onclick
      - data-* attributes
    but ONLY when the attribute value mentions "tournament" to avoid collecting unrelated GUIDs
    (e.g., club image GUIDs).
    """
    soup = BeautifulSoup(html_fragment, "lxml")
    found = set()

    # Check <a href=...>
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "tournament" in href.lower():
            for m in GUID_ONLY_RE.finditer(href):
                found.add(m.group(1).upper())

    # Check onclick handlers
    for tag in soup.select("[onclick]"):
        oc = tag.get("onclick", "")
        if "tournament" in oc.lower():
            for m in GUID_ONLY_RE.finditer(oc):
                found.add(m.group(1).upper())

    # Check any data-* attributes on any element
    for tag in soup.find_all(True):
        for k, v in (tag.attrs or {}).items():
            if not isinstance(k, str) or not k.lower().startswith("data-"):
                continue
            # v can be list or string
            if isinstance(v, list):
                vv = " ".join(str(x) for x in v)
            else:
                vv = str(v)
            if "tournament" in vv.lower():
                for m in GUID_ONLY_RE.finditer(vv):
                    found.add(m.group(1).upper())

    # Extra fallback: sometimes the whole fragment contains a URL like "/tournament/<GUID>" but
    # not in an attribute we caught; search only around the word "tournament" to avoid club GUIDs.
    for m in re.finditer(
        r"tournament.{0,120}", html_fragment, flags=re.IGNORECASE | re.DOTALL
    ):
        chunk = m.group(0)
        for g in GUID_ONLY_RE.finditer(chunk):
            found.add(g.group(1).upper())

    return found


def main():
    s = requests.Session()
    s.headers.update(HEADERS_GET)

    print("GET /find ...")
    r0 = s.get(FIND_URL, params={**FILTERS, "page": 1}, timeout=30)
    r0.raise_for_status()

    base_form = extract_form_fields(r0.text)
    print("Extracted", len(base_form), "form fields from /find")

    # Switch to POST headers for XHR
    s.headers.clear()
    s.headers.update(HEADERS_POST)

    seen = set()
    max_pages = 300  # safety cap

    for page in range(1, max_pages + 1):
        payload = dict(base_form)
        payload["Page"] = str(page)

        # Apply filters using likely field names
        payload["TournamentFilter.DateFilterType"] = FILTERS["DateFilterType"]
        payload["TournamentFilter.StartDate"] = FILTERS["StartDate"]
        payload["TournamentFilter.EndDate"] = FILTERS["EndDate"]
        payload["TournamentFilter.PostalCode"] = FILTERS["PostalCode"]

        # sometimes model binding uses these too
        payload["StartDate"] = FILTERS["StartDate"]
        payload["EndDate"] = FILTERS["EndDate"]
        payload["PostalCode"] = FILTERS["PostalCode"]

        payload.setdefault("TournamentFilter_Q", "")
        payload.setdefault("TournamentFilter.Q", "")

        rr = s.post(DOSEARCH_URL, data=payload, timeout=30)
        rr.raise_for_status()

        fragment = rr.text
        gids = extract_tournament_guids_from_fragment(fragment)

        new = [g for g in gids if g not in seen]
        for g in new:
            seen.add(g)

        print(
            f"page {page:03d}: +{len(new)} (total {len(seen)}) len={len(fragment)}"
        )

        # stop when no new IDs added
        if len(new) == 0:
            # helpful quick signal that we did get results markup
            print("Stopping (no new IDs). Fragment starts with:")
            print(fragment[:250].replace("\n", "\\n"))
            break

        time.sleep(0.4)

    out = sorted(seen)
    with open("tournament_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tournament_id"])
        for tid in out:
            w.writerow([tid])

    print(f"Saved {len(out)} tournament IDs to tournament_ids.csv")


if __name__ == "__main__":
    main()
