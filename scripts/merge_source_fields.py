import csv
import re


INPUT_FILE = "medicines_nhs_bnf_reviewed_with_source_extracts.csv"
OUTPUT_FILE = "medicines_model_ready.csv"


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def strip_common_use_sentence(value: str) -> str:
    value = clean_spaces(value).rstrip(".")
    value = re.sub(r"^[A-Za-z0-9 -]+(?:\([^)]*\))?\s+is commonly used for\s+", "", value, flags=re.I)
    return value


def strip_safety_sentence(value: str) -> str:
    value = clean_spaces(value).rstrip(".")
    value = re.sub(r"\s+Ask a pharmacist or doctor if you are unsure\.?$", "", value, flags=re.I)
    return value


def main() -> None:
    with open(INPUT_FILE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_fields = [
        field
        for field in rows[0].keys()
        if field not in {"source_common_uses", "source_safety_warning", "source_extract_status"}
    ]

    for row in rows:
        if row.get("source_common_uses"):
            row["common_uses"] = strip_common_use_sentence(row["source_common_uses"])
        if row.get("source_safety_warning"):
            row["safety_warning"] = strip_safety_sentence(row["source_safety_warning"])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in output_fields} for row in rows])

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Rows written: {len(rows)}")
    print("The output keeps compact database-style common_uses and safety_warning fields.")


if __name__ == "__main__":
    main()
