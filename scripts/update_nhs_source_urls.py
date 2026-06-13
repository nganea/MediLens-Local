import csv
import re
from pathlib import Path

import requests


INPUT_FILE = Path("medicines.csv")
OUTPUT_FILE = Path("medicines_nhs_reviewed.csv")
NHS_MEDICINES_BASE = "https://www.nhs.uk/medicines"


KNOWN_SLUGS = {
    "aspirin": "aspirin-for-pain-relief",
    "co-codamol": "co-codamol-for-adults",
    "fluoxetine": "fluoxetine-prozac",
    "hydrocortisone cream": "hydrocortisone-skin-cream",
    "ibuprofen": "ibuprofen-for-adults",
    "paracetamol": "paracetamol-for-adults",
    "salbutamol": "salbutamol-inhaler",
    "sildenafil": "sildenafil-viagra",
}


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def candidate_slugs(generic_name: str) -> list[str]:
    base = slugify(generic_name)
    candidates = []

    if generic_name in KNOWN_SLUGS:
        candidates.append(KNOWN_SLUGS[generic_name])

    candidates.extend([base, f"{base}-for-adults"])

    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def page_matches(text: str, generic_name: str, brand_names: str) -> bool:
    haystack = text.lower()
    names = [generic_name]
    names.extend(name.strip() for name in brand_names.split(";") if name.strip())
    return any(name.lower() in haystack for name in names)


def find_nhs_url(row: dict) -> tuple[str, str]:
    for slug in candidate_slugs(row["generic_name"]):
        url = f"{NHS_MEDICINES_BASE}/{slug}/"
        try:
            response = requests.get(url, timeout=6)
        except requests.RequestException:
            continue

        if response.status_code == 200 and page_matches(
            response.text,
            row["generic_name"],
            row["brand_names"],
        ):
            return url, "nhs_url_matched_needs_text_review"

    return row["source_url"], "needs_nhs_manual_review"


def main() -> None:
    with INPUT_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    matched = 0
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            url, status = find_nhs_url(row)
            row["source_name"] = "NHS Medicines A to Z"
            row["source_url"] = url
            row["review_status"] = status
            if status == "nhs_url_matched_needs_text_review":
                matched += 1

            writer.writerow(row)
            handle.flush()
            print(f"{index:03d}/{len(rows)} {row['generic_name']}: {status}")

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Matched exact NHS medicine pages for {matched} of {len(rows)} rows")
    print("Review common_uses and safety_warning before replacing medicines.csv.")


if __name__ == "__main__":
    main()
