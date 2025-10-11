import argparse
import csv
import os
import re
import sys
from typing import Iterable, Tuple


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def iter_image_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith('.'):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXTS:
                yield os.path.join(dirpath, fn)


def to_posix_path(path: str) -> str:
    return path.replace(os.sep, '/')


def clean_base_name(name: str) -> str:
    # Trim stray whitespace and remove accidental embedded suffixes like " .jpg"
    name = name.strip()
    for bad in (" .jpg", " .jpeg", " .png", " .JPG", " .JPEG", " .PNG"):
        if bad in name:
            name = name.replace(bad, "")
    # Also strip any trailing image extension accidentally embedded in the stem
    while True:
        lower = name.lower()
        if lower.endswith('.jpg'):
            name = name[: -4]
        elif lower.endswith('.jpeg'):
            name = name[: -5]
        elif lower.endswith('.png'):
            name = name[: -4]
        else:
            break
    name = name.rstrip(' ._-')
    return name


def parse_title_and_year(stem: str) -> Tuple[str, str]:
    """Return (title, year) from a filename stem (without extension).

    Supports patterns:
    - YYYY-title
    - id_title-with-hyphens-YYYY
    - Other variants with a 4-digit year at end
    Falls back gracefully if year can't be found.
    """
    stem = clean_base_name(stem)

    # Case 1: starts with 4-digit year and a delimiter
    m = re.match(r"^(\d{4})[-_](.+)$", stem)
    if m:
        year = m.group(1)
        title_part = m.group(2)
    else:
        # Drop leading numeric id + delimiter if present
        m2 = re.match(r"^(\d+)[-_](.+)$", stem)
        name2 = m2.group(2) if m2 else stem

        # Find last 4-digit sequence (likely the year)
        years = re.findall(r"(\d{4})(?!\d)", name2)
        if years:
            year = years[-1]
            # Remove trailing delimiter + year if present
            removed = False
            for sep in ('-', '_', ' '):
                suf = sep + year
                if name2.endswith(suf):
                    title_part = name2[: -len(suf)]
                    removed = True
                    break
            if not removed:
                idx = name2.rfind(year)
                if idx != -1:
                    title_part = (name2[:idx] + name2[idx + len(year):]).rstrip('-_ ')
                else:
                    title_part = name2
        else:
            # No year found
            year = ""
            title_part = name2

    # Normalize title: hyphens/underscores to spaces, collapse whitespace, lowercase
    title = title_part.replace('-', ' ').replace('_', ' ')
    title = re.sub(r"\s+", " ", title).strip().lower()
    return title, year


def generate_rows(root: str) -> Iterable[Tuple[str, str, str, str]]:
    files = sorted(iter_image_files(root))
    for abs_path in files:
        rel = os.path.relpath(abs_path, root)
        rel_posix = to_posix_path(rel)
        parts = rel.split(os.sep)
        style = parts[0] if parts else ""
        stem, _ext = os.path.splitext(os.path.basename(abs_path))
        title, year = parse_title_and_year(stem)
        yield rel_posix, title, year, style


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate CSV mapping for paintings-by-style")
    ap.add_argument("--root", default="data/paintings-by-style", help="Root directory of style/artist/images")
    ap.add_argument("--output", default="data/paintings-by-style.csv", help="Output CSV path")
    ap.add_argument("--stdout", action="store_true", help="Print CSV to stdout instead of writing file")
    args = ap.parse_args(list(argv))

    rows = list(generate_rows(args.root))
    header = ["directory+filename", "painting title", "year of painting", "style"]

    if args.stdout:
        w = csv.writer(sys.stdout)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
