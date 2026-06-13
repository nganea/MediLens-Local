import argparse
import csv
import html
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT = "medicines_nhs_bnf_reviewed.csv"
DEFAULT_OUTPUT = "medicines_nhs_bnf_reviewed_with_source_extracts.csv"
SOURCE_CACHE_DIR = Path("source_pages")

COMMON_USE_PATTERNS = [
    r"\bindications\b",
    r"\buses\b",
    r"\bused to\b",
    r"\bused for\b",
    r"\bfor treating\b",
    r"\btreat\b",
    r"\btreats\b",
    r"\btreating\b",
    r"\bprevent\b",
    r"\bprevents\b",
    r"\bpreventing\b",
    r"\brelieve\b",
    r"\brelieves\b",
    r"\brelieving\b",
    r"\bhelps\b",
    r"\bbelongs to\b",
]

SAFETY_PATTERNS = [
    r"\bcautions\b",
    r"\bcontra-indications\b",
    r"\bcontraindications\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bnot suitable\b",
    r"\bwho can and cannot\b",
    r"\bcheck with\b",
    r"\btell your doctor\b",
    r"\btalk to your doctor\b",
    r"\bask your doctor\b",
    r"\bask a pharmacist\b",
    r"\bbefore taking\b",
    r"\bserious\b",
    r"\burgent\b",
    r"\bemergency\b",
    r"\bside effects\b",
    r"\bpregnant\b",
    r"\bbreastfeeding\b",
]

BOILERPLATE_PHRASES = [
    "view ",
    "including dose, uses",
    "skip to main content",
    "what it's used for, side effects, dosage",
    "what it’s used for, side effects, dosage",
    "what it is used for, side effects, dosage",
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def html_to_text(markup: str) -> str:
    markup = re.sub(r"(?is)<script.*?</script>", " ", markup)
    markup = re.sub(r"(?is)<style.*?</style>", " ", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    text = html.unescape(markup)
    return re.sub(r"\s+", " ", text).strip()


def meta_description_text(markup: str) -> str:
    descriptions = []
    for tag in re.findall(r"(?is)<meta[^>]+>", markup):
        if "description" not in tag.lower():
            continue
        match = re.search(r"""content=["']([^"']+)["']""", tag, re.I)
        if match:
            descriptions.append(html.unescape(match.group(1)))
    return " ".join(descriptions)


def read_cached_source(row: dict) -> str:
    candidates = [
        SOURCE_CACHE_DIR / f"{slugify(row['generic_name'])}.txt",
        SOURCE_CACHE_DIR / f"{slugify(row['source_url'])}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def fetch_source_text(url: str, timeout: int) -> tuple[str, str]:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 medicine-label-helper source extraction"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as error:
        return "", f"http_{error.code}"
    except URLError as error:
        return "", f"url_error:{error.reason}"
    except TimeoutError:
        return "", "timeout"

    return f"{meta_description_text(body)} {html_to_text(body)}".strip(), "fetched"


def get_source_text(row: dict, timeout: int) -> tuple[str, str]:
    cached = read_cached_source(row)
    if cached:
        return cached, "cached_text"
    return fetch_source_text(row["source_url"], timeout)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip()
        if 20 <= len(sentence) <= 360:
            cleaned.append(sentence)
    return cleaned


def has_any_pattern(sentence: str, patterns: list[str]) -> bool:
    lower = sentence.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def is_boilerplate(sentence: str) -> bool:
    lower = sentence.lower()
    return any(phrase in lower for phrase in BOILERPLATE_PHRASES)


def relevant_to_medicine(sentence: str, row: dict) -> bool:
    lower = sentence.lower()
    names = [row["generic_name"]]
    names.extend(name.strip() for name in row["brand_names"].split(";") if name.strip())
    return any(name.lower() in lower for name in names)


def choose_sentences(text: str, row: dict, patterns: list[str], limit: int = 2) -> str:
    sentences = split_sentences(text)
    ranked = []
    for sentence in sentences:
        if is_boilerplate(sentence):
            continue
        if has_any_pattern(sentence, patterns):
            score = 2 if relevant_to_medicine(sentence, row) else 1
            ranked.append((score, sentence))

    ranked.sort(reverse=True)
    selected = []
    seen = set()
    for _, sentence in ranked:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            selected.append(sentence)
        if len(selected) >= limit:
            break

    if selected:
        return " ".join(selected)

    fallback = choose_windows(text, row, patterns, limit)
    if fallback:
        return fallback

    return row["common_uses"] if patterns == COMMON_USE_PATTERNS else row["safety_warning"]


def choose_windows(text: str, row: dict, patterns: list[str], limit: int = 2) -> str:
    lower = text.lower()
    windows = []
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            start = max(0, match.start() - 120)
            end = min(len(text), match.end() + 220)
            window = re.sub(r"\s+", " ", text[start:end]).strip(" ,.;:-")
            if is_boilerplate(window):
                continue
            if len(window) >= 35:
                score = 2 if relevant_to_medicine(window, row) else 1
                windows.append((score, window))

    windows.sort(reverse=True)
    selected = []
    seen = set()
    for _, window in windows:
        key = window.lower()
        if key not in seen:
            seen.add(key)
            selected.append(window)
        if len(selected) >= limit:
            break

    return " ".join(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract short source-derived use, safety, and medication-type fields."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows, start=1):
        text, status = get_source_text(row, args.timeout)
        row["source_extract_status"] = status
        row["source_common_uses"] = choose_sentences(text, row, COMMON_USE_PATTERNS)
        row["source_safety_warning"] = choose_sentences(text, row, SAFETY_PATTERNS)
        print(f"{index:03d}/{len(rows)} {row['generic_name']}: {status}")

    fieldnames = list(rows[0].keys())
    for field in [
        "source_extract_status",
        "source_common_uses",
        "source_safety_warning",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print("Review source-derived fields before using them as final user-facing text.")


if __name__ == "__main__":
    main()
