"""How often each keyword appears in the literature: the Europe PMC hit count of
every term in one or more keyword batch files (the keywords.txt format). A term
with no hits is probably misspelled or invented and needs a look.

    python tools/term_evidence.py OUT.tsv BATCH.txt [BATCH.txt ...]

Writes OUT.tsv (term, category, batch, query, hits; hits = -1 when a term cannot
be searched, e.g. a suffix wildcard) and keeps a cache next to it (OUT.tsv.cache.json),
so an interrupted run continues where it stopped. About 5 requests a second.
"""
import concurrent.futures
import json
import os
import sys
import threading
import time

import requests

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UA = "tinyGPT-keyword-check/1.0 (personal research)"
_lock = threading.Lock()
_last = [0.0]


def read_batch(path):
    out, cat = [], None
    for line in open(path, encoding="utf-8").read().splitlines():
        s = line.strip()
        if s.startswith("# "):
            cat = s[2:].strip()
        elif s and not s.startswith("#"):
            out.append((s, cat))
    return out


def query_for(term):
    """Europe PMC query for a keyword term, or None when it cannot be searched."""
    core = term.lstrip("=").strip()
    if core.startswith("*"):
        return None
    prefix = core.endswith("*")
    core = core.rstrip("*").replace('"', " ").strip()
    if not core:
        return None
    if prefix and " " not in core:
        return core + "*"  # single-word prefix: Europe PMC wildcard
    if prefix:  # a phrase cannot end in a wildcard: the leading words as a phrase, plus the last stem
        head, last = core.rsplit(" ", 1)
        return f'"{head}" AND {last}*'
    return f'"{core}"'


def hits(session, query):
    for attempt in range(5):
        with _lock:  # at most about 5 requests a second in total
            wait = _last[0] + 0.2 - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = session.get(API, params={"query": query, "format": "json", "pageSize": 1, "resultType": "idlist"},
                            timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return int(r.json().get("hitCount", 0))
        except (requests.RequestException, ValueError):
            time.sleep(10 * (attempt + 1))
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out, batches = sys.argv[1], sys.argv[2:]
    cache_path = out + ".cache.json"
    try:
        cache = json.load(open(cache_path, encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    rows = [(t, c, os.path.basename(b), query_for(t)) for b in batches for t, c in read_batch(b)]
    todo = sorted({q for *_, q in rows if q and q not in cache})
    print(f"{len(rows):,} terms, {len(todo):,} queries to run ({len(cache):,} cached)", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for q, n in zip(todo, pool.map(lambda q: hits(session, q), todo)):
                if n is not None:
                    cache[q] = n
                done += 1
                if done % 200 == 0:
                    json.dump(cache, open(cache_path + ".part", "w", encoding="utf-8"))
                    os.replace(cache_path + ".part", cache_path)
                    print(f"  {done:,}/{len(todo):,}", flush=True)
    finally:
        json.dump(cache, open(cache_path + ".part", "w", encoding="utf-8"))
        os.replace(cache_path + ".part", cache_path)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("term\tcategory\tbatch\tquery\thits\n")
        for t, c, b, q in rows:
            n = -1 if q is None else cache.get(q, "")
            handle.write(f"{t}\t{c}\t{b}\t{q or ''}\t{n}\n")
    zero = sum(1 for *_, q in rows if q and cache.get(q) == 0)
    print(f"done: {zero:,} terms with no Europe PMC hits; table in {out}")


if __name__ == "__main__":
    main()
