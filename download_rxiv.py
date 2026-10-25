r"""Download bioRxiv or medRxiv preprints that match the keyword list and convert
them to Markdown for data_prep.py, in the same way as download_arxiv.py does
for arXiv (one script for both servers: --server biorxiv|medrxiv).

1. harvest  the server's metadata (title, abstract, category, licence, link to
            the JATS XML) from the official API, api.biorxiv.org/details,
            in 30-day windows from the server's first day (bioRxiv 2013-11,
            medRxiv 2019-06) to today, one page (30-100 records) per request,
            cached in <meta>\metadata\<server>.jsonl.gz with a progress file,
            so a rerun continues where it stopped. --update adds the papers
            posted or revised since the last harvest (from 3 days before it).
            Each paper (DOI) is used once: its latest version.
2. select   papers whose title and abstract match the keywords (default: at
            least 3 matches from 3 different terms; a title match counts 3
            times, as for arXiv), ranked by keyword density. Saved in
            <meta>\list.tsv.
3. fetch    each paper's full text, one request at a time, at least --min-gap
            seconds apart (default 1 s), after a random --wait pause (default
            5-15 s): first the JATS XML the API links to (the server's own
            structured full text); if it is missing, has no body, or is
            blocked (www.biorxiv.org / www.medrxiv.org sit behind Cloudflare,
            which may answer automated requests with a bot-check page, 403 or
            429), the same preprint's JATS from Europe PMC (preprints are
            indexed there under PPR ids; full text for some of them); then the
            PDF (<server>.org/content/<doi>v<version>.full.pdf).
4. convert  JATS to Markdown with download_pmc.py's converter: title and
            abstract at the top, headings, paragraphs, lists, equations as
            LaTeX, tables, numbered citations kept as in the text, figures
            reduced to their captions; references, acknowledgements, funding
            and similar sections left out; e-mail addresses masked. PDFs are
            converted with convert_to_markdown.py's PDF converter in a child
            process with a time limit (download_arxiv.pdf_markdown).

Outputs: <out>\<DOI with / as _>__<title>.md (written to .part, then renamed)
and <meta>\report.tsv (one row per paper: id = DOI, status, characters,
method = jats / europepmc / pdf or why there is no text, category, year,
licence, file). Statuses ok, too_short and no_text are final; errors (network
trouble, blocked pages) are tried again by the next run. Existing outputs are
skipped, so the script can be stopped (Ctrl+C) and rerun. If several papers in
a row are blocked the run stops, so a blocking site is not hammered.

Licences vary from paper to paper and are recorded (list.tsv, report.tsv):
cc_by, cc_by_nc, cc_by_nd, cc_by_nc_nd, cc0 (Creative Commons) and cc_no = no
reuse licence (all rights reserved by the authors; posted for reading).

Safety: XML is parsed as data (Python's parser fetches no external DTD or
entity), nothing downloaded is executed, every download has a size limit and a
timeout, and full-text links are followed only to the server's own site.

Defaults: --out %USERPROFILE%\ai_training_data\<server>, --meta _<out name>_meta
next to it (data_prep --text-root skips folders starting with _).

Examples:
    python download_rxiv.py --server biorxiv
    python download_rxiv.py --server medrxiv --out %USERPROFILE%\ai_training_data\medrxiv
    python download_rxiv.py --server biorxiv --update        (later: new preprints)
"""

import argparse
import datetime
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_prep as dp  # noqa: E402
import download_arxiv as ax  # noqa: E402  (pacing, cache reading, PDF route, report reading)
from download_pmc import EMAIL_RE, Jats, _tag, safe_name  # noqa: E402  (JATS -> Markdown)

API_URL = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}"
SITES = {"biorxiv": "https://www.biorxiv.org", "medrxiv": "https://www.medrxiv.org"}
FIRST_DAY = {"biorxiv": "2013-11-01", "medrxiv": "2019-06-01"}
PDF_URL = "{site}/content/{doi}v{version}.full.pdf"
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML"
USER_AGENT = "tinyGPT-rxiv-downloader/1.0 (personal research; python urllib)"
WINDOW_DAYS = 30
MAX_API = 20 << 20  # one metadata page
MAX_XML = 50 << 20  # one JATS file
MAX_PDF = ax.MAX_DOWNLOAD
MAX_BLOCKED_IN_A_ROW = 5
FINAL = ax.FINAL  # ok, too_short, no_text: not retried on a rerun
REPORT_COLUMNS = ["id", "status", "chars", "method", "category", "year", "licence", "file"]
LIST_COLUMNS = ["id", "version", "score", "year", "category", "licence", "jatsxml", "title", "abstract"]
_flat = ax._flat


# ---------------------------------------------------------------------------
# HTTP: download_arxiv's http_get (one connection, minimum gap, 429/503 pause)
# ---------------------------------------------------------------------------
class Blocked(Exception):
    """The site answered with a bot check (Cloudflare) or refused (403)."""


_CHALLENGE_RE = re.compile(rb"just a moment|cf[-_]chl|challenge-platform|attention required|cloudflare ray id|"
                           rb"enable javascript and cookies|checking your browser|cf-browser-verification", re.I)


def looks_like_html(body):
    head = (body or b"")[:2000].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html")) or b"<html" in head[:300]


def is_challenge(body, headers=None):
    """True for a Cloudflare bot-check / block page (by header or by content)."""
    if headers is not None and (headers.get("cf-mitigated") or "").lower() == "challenge":
        return True
    return bool(body) and looks_like_html(body) and bool(_CHALLENGE_RE.search(body[:20000]))


def _give_up(exc):
    """For http_get: a 429/503 that is a bot check is not a slow-down request; waiting would not help."""
    try:
        body = exc.read(20000)
    except Exception:  # noqa: BLE001
        body = b""
    exc.body = body
    return is_challenge(body, exc.headers)


GAP = [1.0]  # --min-gap


def get(url, limit, tries=4, timeout=60):
    """Bytes, None (404) or 'too_large'; raises Blocked for a bot check or a 403."""
    try:
        body = ax.http_get(url, limit=limit, tries=tries, timeout=timeout, min_gap=GAP[0], user_agent=USER_AGENT,
                           site=urllib.parse.urlsplit(url).hostname or "server", give_up=_give_up)
    except urllib.error.HTTPError as exc:
        if exc.code == 403 or is_challenge(getattr(exc, "body", b""), exc.headers):
            raise Blocked(f"{exc.code}" + (" bot check" if is_challenge(getattr(exc, "body", b""), exc.headers)
                                            else "")) from None
        raise
    if isinstance(body, bytes) and is_challenge(body):
        raise Blocked("bot-check page")
    return body


# ---------------------------------------------------------------------------
# 1. harvest
# ---------------------------------------------------------------------------
def parse_details_page(body):
    """([records], total in the window, records on the page) from one API
    answer. Records keep what the later steps need; an empty window ('no posts
    found') gives ([], 0, 0)."""
    if looks_like_html(body):
        raise Blocked("the API answered with a web page")
    data = json.loads(body)
    msg = (data.get("messages") or [{}])[0]
    records, page = [], data.get("collection") or []
    for r in page:
        doi = _flat(r.get("doi"))
        if not doi:
            continue
        try:
            version = int(r.get("version") or 1)
        except (TypeError, ValueError):
            version = 1
        records.append({"doi": doi, "version": version, "date": _flat(r.get("date")),
                        "title": _flat(r.get("title")), "abstract": _flat(r.get("abstract")),
                        "category": _flat(r.get("category")).lower(), "license": _flat(r.get("license")),
                        "jatsxml": _flat(r.get("jatsxml")), "published": _flat(r.get("published")),
                        "type": _flat(r.get("type"))})
    if msg.get("status") != "ok" and not records:
        return [], 0, 0
    try:
        total = int(msg.get("total") or msg.get("count") or 0)
    except (TypeError, ValueError):
        total = 0
    return records, total, len(page)


def harvest(server, cache_dir, update=False, start=None):
    """Append the server's records to <cache>/<server>.jsonl.gz page by page,
    window by window; the progress file lets a rerun continue. With update=True
    a finished harvest is extended from 3 days before it ended (versions seen
    twice are removed later). Returns the number of records written."""
    base = os.path.join(cache_dir, server)
    state_path, cache_path = base + ".state.json", base + ".jsonl.gz"
    state = {"from": start or FIRST_DAY[server], "cursor": 0, "pages": 0, "records": 0, "complete": False}
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            state.update(json.load(handle))
    # anything after "bytes" is a page whose progress was never saved: cut it off, it is fetched again
    if state.get("bytes") is not None and os.path.isfile(cache_path) and os.path.getsize(cache_path) > state["bytes"]:
        with open(cache_path, "r+b") as handle:
            handle.truncate(state["bytes"])
    if not state["pages"] and os.path.exists(cache_path):
        os.remove(cache_path)  # an interrupted first page: start clean
    if state["complete"] and update:
        since = datetime.date.fromisoformat(state.get("harvested_until") or datetime.date.today().isoformat())
        state.update(complete=False, cursor=0, **{"from": (since - datetime.timedelta(days=3)).isoformat()})
        print(f"  [{server}] cached: {state['records']:,} records; adding papers posted since {state['from']}",
              flush=True)
    elif state["complete"]:
        print(f"  [{server}] cached: {state['records']:,} records")
        return state["records"]
    t0 = time.time()
    while True:
        end = ax._window_end(state["from"], WINDOW_DAYS)
        url = API_URL.format(server=server, start=state["from"], end=end, cursor=state["cursor"])
        body = get(url, MAX_API, tries=4, timeout=120)
        if not isinstance(body, bytes):
            raise RuntimeError(f"{server}: no answer from the API for {state['from']}..{end}")
        records, total, size = parse_details_page(body)
        if records:
            with gzip.open(cache_path, "at", encoding="utf-8") as out:  # gzip members append cleanly
                for r in records:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
        state["bytes"] = os.path.getsize(cache_path) if os.path.isfile(cache_path) else 0
        state["pages"] += 1
        state["records"] += len(records)
        state["cursor"] += size
        if not size or state["cursor"] >= total:  # this window is done: the next, until today
            if end < datetime.date.today().isoformat():
                state.update(cursor=0, **{"from": (datetime.date.fromisoformat(end)
                                                   + datetime.timedelta(days=1)).isoformat()})
            else:
                state.update(complete=True, harvested_until=datetime.date.today().isoformat())
        with open(state_path + ".part", "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(state_path + ".part", state_path)
        if state["pages"] % 50 == 0 or state["complete"]:
            print(f"  [{server}] {state['records']:,} records, {state['pages']:,} pages, up to {state['from']} "
                  f"({time.time() - t0:,.0f} s)", flush=True)
        if state["complete"]:
            return state["records"]


def read_cache(cache_path):
    """Records of the cache, one per DOI: its latest version (the last copy of
    it if the same version was harvested twice). Two passes keep memory small."""
    best = {}  # doi -> (version, line number)
    for number, line in ax._cache_lines(cache_path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        doi, version = rec.get("doi"), rec.get("version", 1)
        if doi and (doi not in best or version >= best[doi][0]):
            best[doi] = (version, number)
    keep = {number for _, number in best.values()}
    for number, line in ax._cache_lines(cache_path, warn=False):
        if number in keep:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# 2. select
# ---------------------------------------------------------------------------
def build_list(cache_path, keyword_path, min_hits, min_distinct, path):
    kw = dp.KeywordFilter(dp.read_terms(keyword_path), [], min_hits, min_distinct, title_weight=3)
    rows, n = [], 0
    for rec in read_cache(cache_path):
        n += 1
        keep, _, hits, _ = kw.evaluate(rec["title"], rec["abstract"])
        if not keep:
            continue
        words = len(rec["title"].split()) + len(rec["abstract"].split())
        rows.append((-hits / math.sqrt(max(words, 50)), rec))
    rows.sort(key=lambda r: (r[0], r[1]["doi"]))
    out = []
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(LIST_COLUMNS) + "\n")
        for neg, rec in rows:
            row = {"id": rec["doi"], "version": str(rec.get("version", 1)), "score": f"{-neg:.3f}",
                   "year": rec.get("date", "")[:4], "category": rec.get("category", ""),
                   "licence": rec.get("license", ""), "jatsxml": rec.get("jatsxml", ""),
                   "title": _flat(rec["title"]), "abstract": _flat(rec["abstract"])}
            handle.write("\t".join(row[c] for c in LIST_COLUMNS) + "\n")
            out.append(row)
    os.replace(path + ".part", path)
    print(f"Paper list: {len(out):,} of {n:,} preprints match the keywords; {path}")
    return out


# ---------------------------------------------------------------------------
# 3. fetch + convert
# ---------------------------------------------------------------------------
def jats_markdown(xml_bytes, title, abstract):
    """(markdown, body characters) of one JATS article, with the title and
    abstract at the top (from the metadata when the XML lacks them)."""
    root = ET.fromstring(xml_bytes)  # expat: no external DTD or entity is fetched
    article = root if _tag(root) == "article" else next((e for e in root.iter() if _tag(e) == "article"), None)
    if article is None:
        raise ValueError(f"<{_tag(root)}> is not a JATS article")
    info, md = Jats(article).markdown()
    if article.find(".//article-meta/abstract") is None and abstract:
        rest = md.split("\n", 1)[1] if info["title"] else md
        md = f"## Abstract\n\n{EMAIL_RE.sub('[email]', _flat(abstract))}\n\n{rest.lstrip()}"
        md = f"# {_flat(info['title'] or title)}\n\n" + md
    elif not info["title"]:
        md = f"# {_flat(title)}\n\n" + md
    return md, info["body_chars"]


def _own_site(url, server):
    parts = urllib.parse.urlsplit(url or "")
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (host == server + ".org" or host.endswith("." + server + ".org"))


def europepmc_xml(doi):
    """(JATS bytes or None, reason) for the preprint's full text in Europe PMC."""
    query = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite",
                                    "pageSize": 10})
    body = get(EPMC_SEARCH_URL + "?" + query, MAX_API, tries=3)
    if not isinstance(body, bytes):
        return None, "europepmc: no answer"
    hits = [r for r in json.loads(body).get("resultList", {}).get("result", [])
            if r.get("source") == "PPR" and (r.get("doi") or "").lower() == doi.lower()]
    if not hits:
        return None, "europepmc: not indexed"
    hit = next((r for r in hits if r.get("inEPMC") == "Y"), None)
    if hit is None or not re.fullmatch(r"PPR\d+", hit.get("id") or ""):
        return None, "europepmc: no full text"
    try:
        xml = get(EPMC_XML_URL.format(hit["id"]), MAX_XML, tries=2)
    except urllib.error.HTTPError as exc:  # Europe PMC answers 500 for preprints it holds only as a PDF
        return None, f"europepmc: no xml ({exc.code})"
    if not isinstance(xml, bytes) or looks_like_html(xml) or not xml.lstrip()[:1] == b"<":
        return None, "europepmc: no xml"
    return xml, "europepmc"


def _trouble(exc):
    return f"{type(exc).__name__}: {str(exc)[:60]}"


def fetch_paper(row, server, out_dir, min_body, pdf_seconds=ax.PDF_SECONDS, pdf_problem=None, europepmc=True,
                pdf=True):
    """Fetch, convert and write one preprint; returns its report row. Routes:
    the server's JATS, Europe PMC's JATS, the PDF. A route that is blocked or
    fails on the network makes the paper an error (tried again next run)
    unless another route gives the text."""
    base = {"id": row["id"], "category": row.get("category", ""), "year": row.get("year", ""),
            "licence": row.get("licence", ""), "chars": 0, "method": "", "file": ""}
    title, abstract = row.get("title", ""), row.get("abstract", "")
    notes, trouble, blocked, short = [], [], False, None  # short: (markdown, method) of a text below min_body

    def accept(md, method, size=None):
        """(md, method) if long enough; else kept as the best short text and None."""
        nonlocal short
        if (len(md) if size is None else size) >= min_body:
            return md, method
        if short is None or len(md) > len(short[0]):
            short = (md, method)
        notes.append(f"{method} too short")
        return None

    def from_jats(xml, method):
        try:
            md, body_chars = jats_markdown(xml, title, abstract)
        except (ET.ParseError, ValueError) as exc:
            notes.append(f"{method} unreadable ({type(exc).__name__})")
            return None
        # judged by its body: XML with the front matter only (common in older bioRxiv XML) goes to the next route
        return accept(md, method, body_chars)

    found = None
    # 1. the server's JATS
    url = row.get("jatsxml", "")
    if url and _own_site(url, server):
        try:
            xml = get(url, MAX_XML)
            if xml is None:
                notes.append("jats missing")
            elif not isinstance(xml, bytes):
                notes.append("jats too large")
            elif looks_like_html(xml):
                notes.append("jats is a web page")
            else:
                found = from_jats(xml, "jats")
        except Blocked as exc:
            blocked = True
            trouble.append(f"jats blocked ({exc})")
        except Exception as exc:  # noqa: BLE001 - network trouble: the next routes may still work
            if isinstance(exc, ax.Stopped):
                raise
            trouble.append(f"jats {_trouble(exc)}")
    else:
        notes.append("no jats link")
    # 2. Europe PMC
    if found is None and europepmc:
        try:
            xml, why = europepmc_xml(row["id"])
            if xml is None:
                notes.append(why)
            else:
                found = from_jats(xml, "europepmc")
        except Blocked as exc:
            trouble.append(f"europepmc blocked ({exc})")
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ax.Stopped):
                raise
            trouble.append(f"europepmc {_trouble(exc)}")
    # 3. the PDF
    if found is None and pdf:
        if pdf_problem:
            trouble.append("no PDF converter")
        else:
            url = PDF_URL.format(site=SITES[server], doi=row["id"], version=row.get("version") or 1)
            try:
                blob = get(url, MAX_PDF)
                if isinstance(blob, bytes) and blob[:5] == b"%PDF-":
                    try:
                        found = accept(ax.pdf_markdown(blob, pdf_seconds), "pdf")
                    except subprocess.TimeoutExpired:  # retried by a run with a larger --pdf-seconds
                        return {**base, "status": "pdf_timeout", "method": f"pdf took over {pdf_seconds:g} s"}
                    except Exception as exc:  # noqa: BLE001 - a damaged PDF: recorded, not fatal
                        reason = f"{type(exc).__name__}: {str(exc)[:80]}"
                        if isinstance(exc, OSError) or ax._NOT_THE_PAPER_RE.search(reason):
                            trouble.append(f"pdf failed: {reason}")
                        else:
                            notes.append(f"pdf failed: {reason}")
                elif blob is None:
                    notes.append("pdf missing")
                elif isinstance(blob, bytes) and looks_like_html(blob):
                    trouble.append("pdf is a web page")
                else:
                    notes.append("pdf too large" if blob == "too_large" else "pdf not a PDF")
            except Blocked as exc:
                # the site's bot protection refuses PDFs (and article pages) to scripts while the JATS files stay
                # open; a paper reaching this point has no JATS file, so the PDF refusal is final for it and is
                # not a sign that the run is being blocked (that is what a refused JATS file shows)
                notes.append(f"pdf refused ({exc})")
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, ax.Stopped):
                    raise
                trouble.append(f"pdf {_trouble(exc)}")
    why = "; ".join(notes + trouble)
    if found is None:
        if trouble:  # not the paper's fault: tried again on the next run
            return {**base, "status": "error: " + "; ".join(trouble), "method": "; ".join(notes),
                    "blocked": blocked}
        if short is not None:
            return {**base, "status": "too_short", "chars": len(short[0]), "method": short[1] + "; " + why}
        return {**base, "status": "no_text", "method": why or "no source"}
    md, method = found
    name = f"{row['id'].replace('/', '_')}__{safe_name(title)}.md"
    path = os.path.join(out_dir, name)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)
    os.replace(path + ".part", path)
    return {**base, "status": "ok", "chars": len(md), "method": method, "file": name}


def todo_rows(rows, done, pdf_limits, have, pdf_seconds):
    """The list rows still to fetch: no Markdown file yet, no final status, and
    a PDF that ran out of time only when the limit is now larger."""
    def wanted(row):
        if row["id"].replace("/", "_") in have:
            return False
        if row["id"] in pdf_limits:
            return pdf_seconds > pdf_limits[row["id"]]
        return done.get(row["id"]) not in FINAL
    return [r for r in rows if wanted(r)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", choices=sorted(SITES), default="biorxiv")
    p.add_argument("--out", help="folder for the Markdown files (default %%USERPROFILE%%\\ai_training_data\\<server>)")
    p.add_argument("--meta", help="folder for the metadata cache, list.tsv and report.tsv (default: _<out name>_meta "
                                  "next to --out, which data_prep --text-root skips)")
    p.add_argument("--keywords", default=os.path.join(HERE, "keywords.txt"))
    p.add_argument("--min-hits", type=int, default=3, help="keyword matches in title + abstract (title counts 3x)")
    p.add_argument("--min-distinct", type=int, default=3, help="different keyword terms needed")
    p.add_argument("--max-papers", type=int, default=0, help="download at most this many (0 = all in the list)")
    p.add_argument("--wait", default="5-15", help="random pause in seconds before each paper, MIN-MAX (default 5-15)")
    p.add_argument("--min-gap", type=float, default=1.0, help="seconds between the starts of any two requests "
                                                              "(default 1)")
    p.add_argument("--min-body-chars", type=int, default=3000, help="skip papers whose text is shorter")
    p.add_argument("--pdf-seconds", type=float, default=ax.PDF_SECONDS,
                   help=f"time limit of one PDF conversion (default {ax.PDF_SECONDS}); papers that ran out of time "
                        "are tried again by a run with a larger limit")
    p.add_argument("--no-europepmc", action="store_true", help="do not try Europe PMC for the full text")
    p.add_argument("--no-pdf", action="store_true", help="do not fall back to the PDF")
    p.add_argument("--rebuild-list", action="store_true", help="select again from the cached metadata (after "
                                                                "changing the keywords or thresholds)")
    p.add_argument("--update", action="store_true",
                   help="also fetch the metadata of preprints posted since the last harvest, then select again")
    p.add_argument("--list-only", action="store_true", help="harvest and select, then stop")
    args = p.parse_args(argv)
    if args.max_papers < 0:
        p.error("--max-papers must be >= 0")
    if args.pdf_seconds <= 0:
        p.error("--pdf-seconds must be > 0")
    if not 0.5 <= args.min_gap <= 600:
        p.error("--min-gap must be 0.5..600 seconds")
    try:
        low, _, high = args.wait.partition("-")
        args.wait = (float(low), float(high or low))
    except ValueError:
        p.error("--wait must be MIN-MAX seconds, e.g. 5-15")
    if not 0 <= args.wait[0] <= args.wait[1] <= 3600:
        p.error("--wait must be MIN-MAX with 0 <= MIN <= MAX <= 3600")
    # %USERPROFILE% / $HOME / ~ are expanded here too: PowerShell leaves %VAR% as it is
    out = args.out or os.path.join("~", "ai_training_data", args.server)
    args.out = os.path.abspath(os.path.expanduser(os.path.expandvars(out)))
    if args.meta:
        args.meta = os.path.expanduser(os.path.expandvars(args.meta))
    args.meta = os.path.abspath(args.meta or os.path.join(os.path.dirname(args.out),
                                                          f"_{os.path.basename(args.out)}_meta"))
    return args


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_args(argv)
    ax._STOP.clear()
    ax._lower_priority()
    GAP[0] = args.min_gap
    cache = os.path.join(args.meta, "metadata")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(cache, args.server + ".jsonl.gz")
    state_path = os.path.join(cache, args.server + ".state.json")
    list_path = os.path.join(args.meta, "list.tsv")
    try:
        was_complete = os.path.isfile(state_path) and json.load(open(state_path, encoding="utf-8")).get("complete")
        print(f"Harvesting {args.server} metadata into {cache} ...", flush=True)
        harvest(args.server, cache, update=args.update)
        if not was_complete or args.update or args.rebuild_list or not os.path.isfile(list_path):
            rows = build_list(cache_path, args.keywords, args.min_hits, args.min_distinct, list_path)
        else:
            rows = ax.read_list(list_path)
            print(f"Paper list: {len(rows):,} preprints from {list_path} (--rebuild-list selects again)")
    except (KeyboardInterrupt, ax.Stopped):
        ax._STOP.set()
        print("\nStopped; rerun the same command to continue.")
        raise SystemExit(130)
    except Blocked as exc:
        raise SystemExit(f"The API refused the harvest ({exc}); rerun later to continue.")
    if args.list_only:
        return
    for name in os.listdir(args.out):  # a file cut off by an interruption
        if name.endswith(".md.part"):
            os.remove(os.path.join(args.out, name))
    report_path = os.path.join(args.meta, "report.tsv")
    pdf_limits = {}
    done = ax.read_report(report_path, pdf_limits)
    have = {name.split("__", 1)[0] for name in os.listdir(args.out) if name.endswith(".md")}
    todo = todo_rows(rows, done, pdf_limits, have, args.pdf_seconds)
    pdf_problem = None if args.no_pdf else ax.check_pdf_route()
    if pdf_problem:
        print(f"Warning: the PDF converter cannot run: {pdf_problem}. Papers that need it are left for a later run.",
              flush=True)
    if args.max_papers:
        todo = todo[:max(0, args.max_papers - sum(1 for r in rows if done.get(r["id"]) == "ok"))]
    per_paper = max(args.min_gap, sum(args.wait) / 2) + 2 * args.min_gap
    print(f"{len(rows) - len(todo):,} preprints done or not wanted; fetching {len(todo):,} into {args.out} "
          f"(about {3600 / per_paper:,.0f} an hour)", flush=True)
    pause = random.Random()
    new_report = not os.path.isfile(report_path)
    counts, t0, blocked_run = {}, time.time(), 0
    with open(report_path, "a", encoding="utf-8", newline="\n") as report:
        if new_report:
            report.write("\t".join(REPORT_COLUMNS) + "\n")
        try:
            for n, row in enumerate(todo, 1):
                if args.wait[1] > 0:
                    time.sleep(pause.uniform(*args.wait))
                try:
                    result = fetch_paper(row, args.server, args.out, args.min_body_chars, args.pdf_seconds,
                                         pdf_problem, not args.no_europepmc, not args.no_pdf)
                except ax.Stopped:
                    raise KeyboardInterrupt from None
                except Exception as exc:  # noqa: BLE001 - recorded, retried on the next run
                    result = {"id": row["id"], "status": f"error: {_trouble(exc)}", "chars": 0, "method": "",
                              "category": row.get("category", ""), "year": row.get("year", ""),
                              "licence": row.get("licence", ""), "file": ""}
                report.write("\t".join(_flat(result[c]) for c in REPORT_COLUMNS) + "\n")
                report.flush()
                key = result["status"].split(":")[0]
                counts[key] = counts.get(key, 0) + 1
                blocked_run = blocked_run + 1 if result.get("blocked") else 0
                if blocked_run >= MAX_BLOCKED_IN_A_ROW:
                    print(f"Stopping: {blocked_run} preprints in a row were blocked by {SITES[args.server]} (bot "
                          "check or 403). Try again later, or with a larger --wait.", flush=True)
                    break
                if n % 25 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  {n:,}/{len(todo):,} preprints ({rate * 3600:,.0f}/hour, about "
                          f"{(len(todo) - n) / max(rate, 1e-9) / 3600:,.1f} h left) {counts}", flush=True)
        except KeyboardInterrupt:
            ax._STOP.set()
            print("\nStopped; rerun the same command to continue.")
            raise SystemExit(130)
    print(f"Done: {counts.get('ok', 0):,} preprints written to {args.out}; statuses {counts}; report {report_path}")


if __name__ == "__main__":
    main()
