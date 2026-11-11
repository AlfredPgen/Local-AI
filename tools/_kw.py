"""The keyword batch format (the keywords.txt format), read in one place for
tools/kw_check.py and tools/term_evidence.py."""


def read_batch(path):
    """[(term, category)] of a batch file: '# category' lines, each followed by one
    term per line; any other line starting with '#' is a comment."""
    out, cat = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            s = line.strip()
            if s.startswith("# "):
                cat = s[2:].strip()
            elif s and not s.startswith("#"):
                out.append((s, cat))
    return out
