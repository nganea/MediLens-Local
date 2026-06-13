import argparse
import csv
import html
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT = "medicines_nhs_bnf_reviewed.csv"
DEFAULT_OUTPUT = "medicine_source_consistency_report.csv"
SOURCE_CACHE_DIR = Path("source_pages")

STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "ask",
    "because",
    "before",
    "by",
    "can",
    "doctor",
    "for",
    "from",
    "have",
    "if",
    "in",
    "is",
    "it",
    "medicine",
    "medicines",
    "not",
    "of",
    "or",
    "only",
    "pharmacist",
    "prescribed",
    "some",
    "take",
    "that",
    "the",
    "this",
    "to",
    "use",
    "used",
    "with",
    "you",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def words(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def overlap_score(source_text: str, field_text: str) -> float:
    field_words = words(field_text)
    if not field_words:
        return 1.0

    source_words = words(source_text)
    matched = field_words & source_words
    return round(len(matched) / len(field_words), 3)


def html_to_text(markup: str) -> str:
    markup = re.sub(r"(?is)<script.*?</script>", " ", markup)
    markup = re.sub(r"(?is)<style.*?</style>", " ", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    text = html.unescape(markup)
    return re.sub(r"\s+", " ", text).strip()


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
        headers={
            "User-Agent": "Mozilla/5.0 medicine-label-helper source review",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as error:
        return "", f"http_{error.code}"
    except URLError as error:
        return "", f"url_error:{error.reason}"
    except TimeoutError:
        return "", "timeout"

    if "html" in content_type:
        return html_to_text(body), "fetched"
    return body, "fetched"


def get_source_text(row: dict, timeout: int) -> tuple[str, str]:
    cached = read_cached_source(row)
    if cached:
        return cached, "cached_text"
    return fetch_source_text(row["source_url"], timeout)


def make_flags(common_score: float, safety_score: float, notes_score: float, source_status: str) -> str:
    flags = []

    if source_status != "fetched" and source_status != "cached_text":
        flags.append("source_unavailable")
    if common_score < 0.35:
        flags.append("review_common_uses")
    if safety_score < 0.25:
        flags.append("review_safety_warning")
    if notes_score < 0.25:
        flags.append("review_notes")

    return ";".join(flags) if flags else "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare medicines.csv summary fields with source-page text."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    report_rows = []
    for index, row in enumerate(rows, start=1):
        source_text, source_status = get_source_text(row, args.timeout)
        common_score = overlap_score(source_text, row["common_uses"])
        safety_score = overlap_score(source_text, row["safety_warning"])
        notes_score = overlap_score(source_text, row["notes"])
        flags = make_flags(common_score, safety_score, notes_score, source_status)

        report_rows.append(
            {
                "generic_name": row["generic_name"],
                "source_name": row["source_name"],
                "source_url": row["source_url"],
                "review_status": row["review_status"],
                "source_status": source_status,
                "common_uses_score": common_score,
                "safety_warning_score": safety_score,
                "notes_score": notes_score,
                "flags": flags,
            }
        )
        print(f"{index:03d}/{len(rows)} {row['generic_name']}: {flags}")

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote {args.output}")
    print("Low scores do not prove the row is wrong; they mean it needs human review.")
    print("For BNF pages that block fetching, paste page text into source_pages/<generic-name>.txt and rerun.")


if __name__ == "__main__":
    main()
