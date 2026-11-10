"""Find keyword terms whose every match is also a match of another term, so one
mention in a text counts as two distinct terms (keywords.txt format).

    python tools/kw_covered.py KEYWORDS.txt [--nested] [--fix]

A term A is covered by a term B with the same number of words when each word of
B matches each word of A: 'polymerase' by 'polymer*', 'spatial transcriptomics'
by 'spatial transcriptomic*', '=CITE-seq' by 'CITE-seq'. Such an A adds nothing
and only counts twice.
--nested also reports phrases that contain a shorter phrase of 2+ words ('China
Kadoorie Biobank' contains 'China Kadoorie'). These are the same kind of double
count as 'linear regression' beside 'regression', which the list has always
allowed, so they are only reported, never removed by --fix.
Named words beside a suffix term ('adalimumab' beside '*mab') are left alone.
Case: a case-insensitive B covers any A; a case-sensitive B covers only
case-sensitive A with the same letters. Exact duplicates keep the first one.
--fix removes the same-length covered terms in place (the narrower term goes).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prep as dp  # noqa: E402


def shape(term):
    words, kinds, cs, lead, trail = dp._term_words(term)
    if not words:
        return None
    return [(w if cs else low, kind) for w, (low, kind) in zip(words, kinds)], cs


def word_covers(b, a, cs_b):
    """Does word pattern b match every word that pattern a matches?"""
    (wb, kb), (wa, ka) = b, a
    if not cs_b:
        wa = wa.lower()
    if kb == "exact":
        return ka == "exact" and wa == wb
    if kb == "prefix":
        return ka in ("exact", "prefix") and wa.startswith(wb)
    return ka in ("exact", "suffix") and wa.endswith(wb)


def covers(sb, sa, nested):
    (pb, cs_b), (pa, cs_a) = sb, sa
    n, m = len(pb), len(pa)
    if (cs_b and not cs_a) or n > m or (n < m and (not nested or n == 1)):
        return False
    return any(all(word_covers(b, a, cs_b) for b, a in zip(pb, pa[s:s + n])) for s in range(m - n + 1))


def find(terms, nested):
    """{covered term: covering term} for a list of terms."""
    shapes = {t: shape(t) for t in terms}
    # possible covering terms, by (the first 3 letters of a word, the number of words); those with a shorter word by
    # the number of words alone. A term can only be covered by one with as many words or, nested, with 2 or more
    # but fewer (covers), so only those are looked at: comparing all of them was close to quadratic
    by_word, short = {}, {}
    for t in terms:
        s = shapes[t]
        if not s or any(kind == "suffix" for _, kind in s[0]):
            continue
        n = len(s[0])
        for w, _ in s[0]:
            (short.setdefault(n, set()) if len(w) < 3 else by_word.setdefault((w.lower()[:3], n), set())).add(t)
    covered, seen = {}, {}
    for t in terms:
        sa = shapes[t]
        if not sa:
            continue
        key = (tuple(sa[0]), sa[1])
        if seen.get(key, t) != t:
            covered[t] = seen[key]
            continue
        seen[key] = t
        m, cands = len(sa[0]), set()
        for n in [m] + (list(range(2, m)) if nested else []):
            cands |= short.get(n, set())
            for w, _ in sa[0]:
                cands |= by_word.get((w.lower()[:3], n), set())
        for u in sorted(cands):
            # a term of the same shape (a case variant) is left to the `seen` rule above: the first one stays
            if u != t and u not in covered and shapes[u] and shapes[u] != sa and covers(shapes[u], sa, nested):
                covered[t] = u
                break
    return covered


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keywords", metavar="KEYWORDS.txt", help="keyword file to check")
    p.add_argument("--nested", action="store_true", help="also report phrases that contain a shorter phrase")
    p.add_argument("--fix", action="store_true", help="remove the same-length covered terms from the file")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    path, nested, fix = args.keywords, args.nested, args.fix
    lines = open(path, encoding="utf-8").read().splitlines()
    terms = list(dict.fromkeys(ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")))
    covered = find(terms, nested)
    same = find(terms, False) if nested else covered
    for t, u in sorted(covered.items(), key=lambda kv: kv[0].lower()):
        print(f"{t}\t{'same length' if t in same else 'nested'}\t{u}")
    print(f"{len(same)} same-length covered terms" + (f", {len(covered) - len(same)} nested" if nested else ""),
          file=sys.stderr)
    if fix and same:
        out = [ln for ln in lines if ln.strip().startswith("#") or ln.strip() not in same]
        with open(path + ".part", "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(out) + "\n")
        os.replace(path + ".part", path)


if __name__ == "__main__":
    main()
