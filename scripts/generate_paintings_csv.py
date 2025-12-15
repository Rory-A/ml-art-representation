import csv
import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote
from typing import Optional, Tuple


PAINTINGS_DIR = Path("data/paintings")
UNUSED_DIR = Path("data/unused-paintings")
OUTPUT_CSV = Path("data/paintings.csv")


def strip_multiple_extensions(name: str) -> str:
    """Remove one or more trailing image extensions and stray whitespace."""
    s = name
    s = s.rstrip()
    pattern = re.compile(r"(\.(?:jpe?g|png|gif|webp|tif|tiff))\s*$", re.IGNORECASE)
    while True:
        new_s = pattern.sub("", s)
        if new_s == s:
            break
        s = new_s.rstrip()
    return s


def find_last_year(text: str) -> Tuple[Optional[int], Optional[re.Match]]:
    """Return the final four-digit year in the text, if any."""
    year_re = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")
    last = None
    last_m = None
    for m in year_re.finditer(text):
        y = int(m.group(1))
        if 1000 <= y <= 2099:
            last = y
            last_m = m
    return last, last_m


def filename_to_title_and_year(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (title, year) from a filename; return (None, None) if missing."""
    base = os.path.basename(filename)
    base_wo_ext = strip_multiple_extensions(base)
    base_wo_ext = re.sub(r"^\d+[_-]", "", base_wo_ext)

    decoded = unquote(base_wo_ext)

    year, match = find_last_year(decoded)
    if year is None or match is None:
        return None, None

    title_part = decoded[: match.start()].strip(" _-.")
    title = re.sub(r"[_-]+", " ", title_part)
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        title = re.sub(r"[_-]+", " ", decoded).strip()
    return title, str(year)


def main() -> None:
    if not PAINTINGS_DIR.exists():
        raise SystemExit(f"Directory not found: {PAINTINGS_DIR}")

    UNUSED_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str]] = []

    for entry in sorted(PAINTINGS_DIR.iterdir()):
        if not entry.is_file():
            continue
        # Skip hidden/system files
        if entry.name.startswith('.'):
            continue

        title, year = filename_to_title_and_year(entry.name)
        if year is None:
            # Move to unused-paintings
            dest = UNUSED_DIR / entry.name
            # If a file with the same name exists, append a numeric suffix
            if dest.exists():
                stem = strip_multiple_extensions(entry.name)
                ext = entry.suffix or ''
                # ext may be wrong if multiple extensions; try to recover last suffix from original name
                # Prefer to keep original suffix from entry.name
                orig_ext_match = re.search(r"(\.[^.\s]+)\s*$", entry.name)
                if orig_ext_match:
                    ext = orig_ext_match.group(1)
                base_stem = Path(stem).name
                i = 1
                while True:
                    candidate = UNUSED_DIR / f"{base_stem}-{i}{ext}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    i += 1
            shutil.move(str(entry), str(dest))
        else:
            # Include in CSV: basename, title, year
            rows.append((entry.name, title, year))

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "title", "year"])
        for filename, title, year in rows:
            writer.writerow([filename, title, year])

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
