#!/usr/bin/env python3
"""
match_scraper.py (HYBRID)

Outputs a single matches.csv for both:
- New mode: /tournament/<GUID>/matches/YYYYMMDD (div.match.match--list)
- Legacy mode: /sport/legacymatches.aspx?id=<GUID>&d=YYYYMMDD (tables)

Usage:
  conda activate racketlon
  pip install requests beautifulsoup4 lxml
  python match_scraper.py
  python match_scraper.py --limit 5
  python match_scraper.py --only 061FEB73-CE57-45C1-9E3C-C56D40BA9112
  python match_scraper.py --fresh

Notes:
- Resume: by default it writes scraper_state.json and continues.
- Use --fresh to start from scratch.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

BASE = "https://fir.tournamentsoftware.com"
TOURNAMENT_IDS_CSV = "tournament_ids.csv"
OUT_CSV = "matches.csv"
STATE_JSON = "scraper_state.json"

HEADERS = {
    "User-Agent": "RacketlonScraper/1.0 (+mailto:zain@officespacesoftware.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

DISCIPLINES = ["TT", "BD", "SQ", "TN"]

DAY_URL_RE_NEW = re.compile(r"/tournament/([0-9a-fA-F-]{36})/matches/(\d{8})")
LEGACY_DAY_RE = re.compile(r"legacymatches\.aspx", re.IGNORECASE)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(8.0, 0.6 * (2**attempt)))


def fetch(session: requests.Session, url: str, *, timeout: int = 45) -> str:
    for attempt in range(5):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except (requests.RequestException, requests.Timeout) as e:
            if attempt == 4:
                raise
            print(
                f"⚠️ fetch failed ({type(e).__name__}) {url} — retrying...",
                file=sys.stderr,
            )
            _backoff_sleep(attempt)
    raise RuntimeError("unreachable")


def normalize_guid(g: str) -> str:
    return g.strip().upper()


def load_tournament_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "tournament_id" not in (reader.fieldnames or []):
            raise ValueError(
                "tournament_ids.csv must have a header column: tournament_id"
            )
        for row in reader:
            tid = (row.get("tournament_id") or "").strip()
            if tid:
                ids.append(normalize_guid(tid))
    return ids


def load_state(fresh: bool) -> Dict:
    if fresh:
        return {"done": {}}
    p = Path(STATE_JSON)
    if not p.exists():
        return {"done": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"done": {}}


def save_state(state: Dict) -> None:
    Path(STATE_JSON).write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_rows(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)


def clean_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def join_nonempty(parts: List[Optional[str]], sep=" / ") -> str:
    return sep.join([p for p in parts if p])


def safe_int(s: str) -> Optional[int]:
    s = clean_text(s)
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------
# NEW MODE SCRAPING
# ---------------------------


def discover_new_day_urls(base_html: str) -> List[Tuple[str, str]]:
    """Return list of (url, YYYYMMDD) for /tournament/<guid>/matches/YYYYMMDD links."""
    soup = BeautifulSoup(base_html, "lxml")
    out = set()
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        m = DAY_URL_RE_NEW.search(href)
        if m:
            full = href if href.startswith("http") else (BASE + href)
            out.add((full, m.group(2)))
    return sorted(out, key=lambda t: t[1])


def parse_new_match_div(
    match_div: BeautifulSoup, tournament_id: str, match_date: Optional[str]
) -> dict:
    # time from nearest previous group header
    time_h5 = match_div.find_previous("h5", class_="match-group__header")
    match_time = (
        clean_text(time_h5.get_text(" ", strip=True)) if time_h5 else ""
    )

    # draw + draw_id + round
    draw = ""
    draw_id = ""
    round_name = ""

    draw_a = match_div.select_one(".match__header-title a[href*='draw.aspx']")
    if draw_a:
        draw = clean_text(draw_a.get_text(" ", strip=True))
        href = draw_a.get("href") or ""
        qs = parse_qs(urlparse(href).query)
        if "draw" in qs and qs["draw"]:
            draw_id = qs["draw"][0]

    # round (often second header-title item)
    round_el = match_div.select_one(
        ".match__header-title .match__header-title-item:nth-of-type(2)"
    )
    if round_el:
        round_name = clean_text(round_el.get_text(" ", strip=True))

    # duration + location (often tooltip title and/or footer)
    duration = ""
    location = ""

    aside = match_div.select_one(".match__header-aside-block")
    if aside and aside.has_attr("title"):
        title = aside["title"]
        # "Duration: 1h 00m | Main Location - Court 1"
        if "Duration:" in title:
            try:
                duration = clean_text(
                    title.split("Duration:", 1)[1].split("|", 1)[0]
                )
            except Exception:
                pass
        if "|" in title:
            location = clean_text(title.split("|", 1)[1])

    foot_loc = match_div.select_one(".match__footer .nav-link__value")
    if foot_loc and foot_loc.get_text(strip=True):
        location = clean_text(foot_loc.get_text(" ", strip=True))

    # teams + winner
    rows = match_div.select(".match__row")
    team1_names, team2_names = [], []
    team1_ids, team2_ids = [], []
    team1_nats, team2_nats = [], []
    team1_clubs, team2_clubs = [], []
    winner_side: Optional[int] = None
    status_message = ""

    for idx, row in enumerate(rows[:2], start=1):
        if "has-won" in (row.get("class") or []):
            winner_side = idx

        # status message sometimes exists
        msg = row.select_one(".match__message")
        if msg and msg.get_text(strip=True):
            status_message = clean_text(msg.get_text(" ", strip=True))

        players = []
        for a in row.select("a.nav-link[data-player-id]"):
            players.append(
                (
                    clean_text(a.get_text(" ", strip=True)),
                    a.get("data-player-id") or "",
                    a.get("data-nationality-id") or "",
                    a.get("data-club-id") or "",
                )
            )

        names = [p[0] for p in players if p[0]]
        pids = [p[1] for p in players if p[1]]
        nats = [p[2] for p in players if p[2]]
        clubs = [p[3] for p in players if p[3]]

        if idx == 1:
            team1_names, team1_ids, team1_nats, team1_clubs = (
                names,
                pids,
                nats,
                clubs,
            )
        else:
            team2_names, team2_ids, team2_nats, team2_clubs = (
                names,
                pids,
                nats,
                clubs,
            )

    # points: div.match__result -> ul.points (in TT/BD/SQ/TN order) -> two li.points__cell
    scores = {d: (None, None) for d in DISCIPLINES}
    raw_points_parts = []
    res = match_div.select_one("div.match__result")
    if res:
        uls = res.select("ul.points")
        for i, ul in enumerate(uls[:4]):
            cells = ul.select("li.points__cell")
            if len(cells) >= 2:
                p1 = safe_int(cells[0].get_text(" ", strip=True))
                p2 = safe_int(cells[1].get_text(" ", strip=True))
                d = DISCIPLINES[i]
                scores[d] = (p1, p2)
                if p1 is not None or p2 is not None:
                    raw_points_parts.append(
                        f"{'' if p1 is None else p1}-{'' if p2 is None else p2}"
                    )

    # H2H URL if present
    h2h_url = ""
    h2h = match_div.select_one("a.match__btn-h2h[href]")
    if h2h:
        href = h2h.get("href") or ""
        h2h_url = href if href.startswith("http") else (BASE + href)

    return {
        "mode": "new",
        "tournament_id": tournament_id,
        "match_date": match_date or "",
        "match_time": match_time,
        "draw": draw,
        "draw_id": draw_id,
        "round": round_name,
        "duration": duration,
        "location": location,
        "team1_players": " / ".join(team1_names),
        "team2_players": " / ".join(team2_names),
        "team1_player_ids": " / ".join(team1_ids),
        "team2_player_ids": " / ".join(team2_ids),
        "team1_nationalities": " / ".join(team1_nats),
        "team2_nationalities": " / ".join(team2_nats),
        "team1_club_ids": " / ".join(team1_clubs),
        "team2_club_ids": " / ".join(team2_clubs),
        "winner_side": winner_side if winner_side is not None else "",
        "status_message": status_message,
        "TT_p1": scores["TT"][0],
        "TT_p2": scores["TT"][1],
        "BD_p1": scores["BD"][0],
        "BD_p2": scores["BD"][1],
        "SQ_p1": scores["SQ"][0],
        "SQ_p2": scores["SQ"][1],
        "TN_p1": scores["TN"][0],
        "TN_p2": scores["TN"][1],
        "raw_points": "|".join(raw_points_parts),
        "h2h_url": h2h_url,
    }


def scrape_new_mode(
    session: requests.Session, tournament_id: str
) -> List[dict]:
    base_url = f"{BASE}/tournament/{tournament_id}/Matches"
    base_html = fetch(session, base_url)

    day_urls = discover_new_day_urls(base_html)
    targets: List[Tuple[str, Optional[str]]] = []
    if day_urls:
        targets = [(u, d) for (u, d) in day_urls]
    else:
        # sometimes matches are on base page
        targets = [(base_url, None)]

    all_rows: List[dict] = []
    for url, day in targets:
        html = fetch(session, url)
        soup = BeautifulSoup(html, "lxml")
        match_divs = soup.select("div.match.match--list")
        if not match_divs:
            continue
        for md in match_divs:
            all_rows.append(parse_new_match_div(md, tournament_id, day))
    return all_rows


# ---------------------------
# LEGACY MODE SCRAPING
# ---------------------------


def discover_legacy_day_urls(
    session: requests.Session, tournament_id: str
) -> List[Tuple[str, Optional[str]]]:
    """
    Legacy tournaments usually have a calendar on:
      /sport/tournament.aspx?id=<GUID>
    It links to legacymatches.aspx?id=<GUID>&d=YYYYMMDD and often an unscheduled link without d.
    """
    url = f"{BASE}/sport/tournament.aspx?id={tournament_id}"
    html = fetch(session, url)
    soup = BeautifulSoup(html, "lxml")

    found: List[Tuple[str, Optional[str]]] = []
    seen = set()

    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if not LEGACY_DAY_RE.search(href):
            continue

        full = href if href.startswith("http") else urljoin(BASE, href)
        qs = parse_qs(urlparse(full).query)
        day = qs.get("d", [""])[0]
        day = day if (day.isdigit() and len(day) == 8) else None

        key = (full, day)
        if key not in seen:
            seen.add(key)
            found.append(key)

    # Sort with dated pages first
    found.sort(key=lambda t: (t[1] is None, t[1] or ""))
    return found


def guess_winner_from_total(
    total_p1: Optional[int], total_p2: Optional[int]
) -> Optional[int]:
    if total_p1 is None or total_p2 is None:
        return None
    if total_p1 > total_p2:
        return 1
    if total_p2 > total_p1:
        return 2
    return None


def parse_legacy_table(
    day_html: str, tournament_id: str, match_date: Optional[str]
) -> List[dict]:
    """
    Legacy pages typically have tables where each match is a row with columns like:
      Time | Draw/Event | Round | Location | Player1 | Player2 | TT | BD | SQ | TN | Total
    Markup varies, so we do robust header mapping + fallbacks.
    """
    soup = BeautifulSoup(day_html, "lxml")

    tables = soup.select("table")
    if not tables:
        return []

    # pick the table with the most rows
    table = max(tables, key=lambda t: len(t.select("tr")))
    trs = table.select("tr")
    if len(trs) < 2:
        return []

    # Build header map if possible
    header_cells = trs[0].find_all(["th", "td"])
    headers = [
        clean_text(h.get_text(" ", strip=True)).lower() for h in header_cells
    ]

    def col_index(*names: str) -> Optional[int]:
        for name in names:
            n = name.lower()
            for i, h in enumerate(headers):
                if n == h or n in h:
                    return i
        return None

    i_time = col_index("time")
    i_draw = col_index("draw", "event", "category")
    i_round = col_index("round")
    i_loc = col_index("court", "location", "venue")
    i_p1 = col_index("player 1", "player1", "home")
    i_p2 = col_index("player 2", "player2", "away")
    i_tt = col_index("tt", "table tennis")
    i_bd = col_index("bd", "badminton")
    i_sq = col_index("sq", "squash")
    i_tn = col_index("tn", "tennis")
    i_tot1 = col_index("total 1", "total1", "home total", "total")
    i_tot2 = col_index("total 2", "total2", "away total")

    # Fallback if no headers: assume common ordering near end
    def cell_text(c):
        return clean_text(c.get_text(" ", strip=True))

    out: List[dict] = []

    for tr in trs[1:]:
        tds = tr.find_all("td")
        if not tds:
            continue

        # crude skip for separator rows
        row_text = clean_text(tr.get_text(" ", strip=True))
        if not row_text or len(row_text) < 5:
            continue

        def get(i: Optional[int]) -> str:
            if i is None or i >= len(tds):
                return ""
            return cell_text(tds[i])

        match_time = get(i_time)

        draw = get(i_draw)
        round_name = get(i_round)
        location = get(i_loc)

        p1 = get(i_p1)
        p2 = get(i_p2)

        # Some legacy pages mark winner with "W" near player name or a CSS class
        winner_side: Optional[int] = None
        if " w " in f" {p1.lower()} ":
            winner_side = 1
            p1 = p1.replace(" W", "").replace("w", "").strip()
        if " w " in f" {p2.lower()} ":
            winner_side = 2
            p2 = p2.replace(" W", "").replace("w", "").strip()

        # Parse TT/BD/SQ/TN; legacy often shows like "21-14" or "21 14"
        def parse_pair(s: str) -> Tuple[Optional[int], Optional[int]]:
            s = clean_text(s)
            if not s:
                return (None, None)
            if "-" in s:
                a, b = s.split("-", 1)
                return (safe_int(a), safe_int(b))
            # fallback: two ints in cell
            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                return (safe_int(nums[0]), safe_int(nums[1]))
            return (None, None)

        tt = parse_pair(get(i_tt))
        bd = parse_pair(get(i_bd))
        sq = parse_pair(get(i_sq))
        tn = parse_pair(get(i_tn))

        # totals sometimes exist as two cols; sometimes one (combined)
        total1 = safe_int(get(i_tot1)) if i_tot1 is not None else None
        total2 = safe_int(get(i_tot2)) if i_tot2 is not None else None

        if winner_side is None:
            winner_side = guess_winner_from_total(total1, total2)

        # legacy has no player-id/nationality attributes typically
        raw_points = "|".join(
            [
                f"{'' if tt[0] is None else tt[0]}-{'' if tt[1] is None else tt[1]}",
                f"{'' if bd[0] is None else bd[0]}-{'' if bd[1] is None else bd[1]}",
                f"{'' if sq[0] is None else sq[0]}-{'' if sq[1] is None else sq[1]}",
                f"{'' if tn[0] is None else tn[0]}-{'' if tn[1] is None else tn[1]}",
            ]
        )

        out.append(
            {
                "mode": "legacy",
                "tournament_id": tournament_id,
                "match_date": match_date or "",
                "match_time": match_time,
                "draw": draw,
                "draw_id": "",
                "round": round_name,
                "duration": "",
                "location": location,
                "team1_players": p1,
                "team2_players": p2,
                "team1_player_ids": "",
                "team2_player_ids": "",
                "team1_nationalities": "",
                "team2_nationalities": "",
                "team1_club_ids": "",
                "team2_club_ids": "",
                "winner_side": winner_side if winner_side is not None else "",
                "status_message": "",
                "TT_p1": tt[0],
                "TT_p2": tt[1],
                "BD_p1": bd[0],
                "BD_p2": bd[1],
                "SQ_p1": sq[0],
                "SQ_p2": sq[1],
                "TN_p1": tn[0],
                "TN_p2": tn[1],
                "raw_points": raw_points,
                "h2h_url": "",
            }
        )

    return out


def scrape_legacy_mode(
    session: requests.Session, tournament_id: str
) -> List[dict]:
    day_urls = discover_legacy_day_urls(session, tournament_id)
    if not day_urls:
        return []

    all_rows: List[dict] = []
    for url, day in day_urls:
        html = fetch(session, url)
        all_rows.extend(parse_legacy_table(html, tournament_id, day))
    return all_rows


# ---------------------------
# MAIN (HYBRID)
# ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore scraper_state.json and start over.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N tournaments.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only process a specific tournament GUID.",
    )
    args = ap.parse_args()

    tournament_ids = load_tournament_ids(TOURNAMENT_IDS_CSV)
    if args.only:
        tournament_ids = [normalize_guid(args.only)]
    if args.limit:
        tournament_ids = tournament_ids[: args.limit]

    state = load_state(args.fresh)

    # Unified schema (same for both modes)
    fieldnames = [
        "mode",
        "tournament_id",
        "match_date",
        "match_time",
        "draw",
        "draw_id",
        "round",
        "duration",
        "location",
        "team1_players",
        "team2_players",
        "team1_player_ids",
        "team2_player_ids",
        "team1_nationalities",
        "team2_nationalities",
        "team1_club_ids",
        "team2_club_ids",
        "winner_side",
        "status_message",
        "TT_p1",
        "TT_p2",
        "BD_p1",
        "BD_p2",
        "SQ_p1",
        "SQ_p2",
        "TN_p1",
        "TN_p2",
        "raw_points",
        "h2h_url",
    ]

    ensure_csv_header(OUT_CSV, fieldnames)

    with requests.Session() as session:
        for idx, tid in enumerate(tournament_ids, 1):
            if state.get("done", {}).get(tid):
                continue

            print(f"\n[{idx:03d}/{len(tournament_ids):03d}] {tid}")

            rows: List[dict] = []
            mode_used = ""

            # 1) Try NEW mode first
            try:
                new_rows = scrape_new_mode(session, tid)
            except Exception as e:
                new_rows = []
                print(f"  ⚠️ new-mode error: {e}")

            if new_rows:
                rows = new_rows
                mode_used = "new"
                print(f"  ✅ new-mode: {len(rows)} matches")
            else:
                # 2) Fall back to LEGACY mode
                try:
                    legacy_rows = scrape_legacy_mode(session, tid)
                except Exception as e:
                    legacy_rows = []
                    print(f"  ⚠️ legacy-mode error: {e}")

                if legacy_rows:
                    rows = legacy_rows
                    mode_used = "legacy"
                    print(f"  ✅ legacy-mode: {len(rows)} matches")
                else:
                    print("  ❌ no matches found in either mode (skipping)")
                    state.setdefault("done", {})[tid] = True
                    save_state(state)
                    continue

            append_rows(OUT_CSV, fieldnames, rows)

            state.setdefault("done", {})[tid] = True
            save_state(state)
            print(f"  💾 wrote {len(rows)} rows ({mode_used})")

    print("\nDone.")


if __name__ == "__main__":
    main()
