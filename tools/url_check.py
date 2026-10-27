"""Check that the URLs in a documentation-sources TSV answer (status, final address,
type, size). Prints a JSON report. One request at a time, 1 s apart.

    python tools/url_check.py SOURCES.tsv

The TSV needs a header with at least the columns 'tool' and 'url'.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request

UA = "tinyGPT-docs-check/1.0 (personal research; python urllib)"


def check(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(200_000)
            return {"status": r.status, "final": r.geturl(), "type": r.headers.get("content-type", ""),
                    "bytes_read": len(body), "ok": True}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "ok": False, "error": str(exc)[:120]}
    except Exception as exc:  # noqa: BLE001
        return {"status": None, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # JSON may hold non-ASCII names, also when redirected to a file
    lines = [ln for ln in open(sys.argv[1], encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))  # '#' lines are comments, as in download_docs.py
    out = []
    for r in rows:
        res = check(r["url"].strip())
        out.append({"tool": r["tool"], "url": r["url"], **res})
        time.sleep(1)
    bad = [o for o in out if not o["ok"]]
    print(json.dumps({"sources": len(out), "failed": len(bad), "failures": bad,
                      "redirected": [o for o in out if o.get("ok") and o.get("final") != o["url"]]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
