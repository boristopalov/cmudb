#!/usr/bin/env python3
"""Enforce the comment policy described in CONTRIBUTING.md.

Rust sources may only carry doc comments (`///`, `//!`) and `// SAFETY:`
justifications directly above an `unsafe` block. Doc comments may not hold
hidden knowledge such as TODOs, workarounds or invariants; that belongs in
code (types, asserts, errors, tests).

Usage: scripts/check_comments.py [paths...]   (defaults to src/ and tests/)
"""

import pathlib
import re
import sys

BANNED_IN_DOCS = re.compile(
    r"\b(todo|fixme|hack|xxx|workaround|for now|temporar(y|ily)|invariant|"
    r"assum(e|es|ed|ing|ption)|must|never|always|careful|beware|note that)\b",
    re.IGNORECASE,
)


def comments(src: str):
    """Yield (line, kind, text) for every comment, skipping string/char literals."""
    i, line, n = 0, 1, len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif src.startswith("//", i):
            end = src.find("\n", i)
            end = n if end == -1 else end
            yield line, "line", src[i:end]
            i = end
        elif src.startswith("/*", i):
            start_line, depth, j = line, 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    line += src[j] == "\n"
                    j += 1
            yield start_line, "block", src[i:j]
            i = j
        elif c == "r" and re.match(r'r#*"', src[i:]) and not re.match(r"\w", src[i - 1 : i] or " "):
            hashes = re.match(r"r(#*)\"", src[i:]).group(1)
            close = '"' + hashes
            j = src.find(close, i + 2 + len(hashes))
            j = n if j == -1 else j + len(close)
            line += src.count("\n", i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            line += src.count("\n", i, j)
            i = j + 1
        elif c == "'" and re.match(r"'(\\.|[^\\'])'|'\\u\{[0-9a-fA-F]+\}'|'\\x[0-9a-fA-F]{2}'", src[i:]):
            i += re.match(r"'(\\.|[^\\'])'|'\\u\{[0-9a-fA-F]+\}'|'\\x[0-9a-fA-F]{2}'", src[i:]).end()
        else:
            i += 1


def check(path: pathlib.Path):
    src = path.read_text()
    lines = src.splitlines()
    for line, kind, text in comments(src):
        is_doc = kind == "line" and (text.startswith("///") or text.startswith("//!")) and not text.startswith("////")
        if is_doc:
            hit = BANNED_IN_DOCS.search(text)
            if hit:
                yield line, f"doc comment mentions {hit.group(0)!r}; express it in code instead"
            continue
        if kind == "line" and text.startswith("// SAFETY:"):
            following = "\n".join(lines[line : line + 3])
            if "unsafe" in following:
                continue
            yield line, "SAFETY comment is not directly above an unsafe block"
            continue
        yield line, "non-doc comment; express it in code (names, types, asserts, errors, tests)"


def main(argv):
    roots = [pathlib.Path(p) for p in (argv or ["src", "tests"])]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.rs")))
    problems = [(f, line, msg) for f in files for line, msg in check(f)]
    for f, line, msg in problems:
        print(f"{f}:{line}: {msg}")
    if problems:
        print(f"\n{len(problems)} comment(s) violate the policy in CONTRIBUTING.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
