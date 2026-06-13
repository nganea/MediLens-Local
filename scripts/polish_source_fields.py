import csv
import re


INPUT_FILE = "medicines_nhs_bnf_reviewed_with_source_extracts.csv"
OUTPUT_FILE = "medicines_nhs_bnf_reviewed_with_source_extracts.csv"


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def sentence(value: str) -> str:
    value = clean_spaces(value)
    if not value:
        return ""
    if value[-1] not in ".!?":
        value += "."
    return value


def medicine_display_name(row: dict) -> str:
    generic = clean_spaces(row["generic_name"])
    brands = [name.strip() for name in row["brand_names"].split(";") if name.strip()]
    if brands:
        return f"{generic} ({brands[0]})"
    return generic


def polish_common_uses(row: dict) -> str:
    name = medicine_display_name(row)
    uses = clean_spaces(row["common_uses"]).rstrip(".")
    return sentence(f"{name} is commonly used for {uses}")


def polish_safety_warning(row: dict) -> str:
    warning = sentence(row["safety_warning"])
    if "ask a pharmacist" not in warning.lower() and "ask a doctor" not in warning.lower():
        warning = f"{warning} Ask a pharmacist or doctor if you are unsure."
    return sentence(warning)


def trim_to_word_limit(value: str, limit: int = 150) -> str:
    words = clean_spaces(value).split()
    if len(words) <= limit:
        return clean_spaces(value)
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def main() -> None:
    with open(INPUT_FILE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["source_common_uses"] = trim_to_word_limit(polish_common_uses(row))
        row["source_safety_warning"] = trim_to_word_limit(polish_safety_warning(row))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {OUTPUT_FILE}")
    print(f"Rows updated: {len(rows)}")


if __name__ == "__main__":
    main()
