import csv
import re
from pathlib import Path


INPUT_FILE = Path("medicines_nhs_reviewed.csv")
OUTPUT_FILE = Path("medicines_nhs_bnf_reviewed.csv")
BNF_BASE = "https://bnf.nice.org.uk/drugs"


BNF_SLUGS = {
    "acyclovir": "aciclovir",
    "amitriptyline": "amitriptyline-hydrochloride",
    "beclometasone": "beclometasone-dipropionate",
    "calcium carbonate colecalciferol": "calcium-carbonate-with-colecalciferol",
    "chlorphenamine": "chlorphenamine-maleate",
    "dabigatran": "dabigatran-etexilate",
    "ethinylestradiol levonorgestrel": "ethinylestradiol-with-levonorgestrel",
    "formoterol": "formoterol-fumarate",
    "glyceryl trinitrate": "glyceryl-trinitrate",
    "ipratropium": "ipratropium-bromide",
    "isosorbide mononitrate": "isosorbide-mononitrate",
    "levodopa co-beneldopa": "co-beneldopa",
    "levodopa co-careldopa": "co-careldopa",
    "lisdexamfetamine": "lisdexamfetamine-mesilate",
    "medroxyprogesterone": "medroxyprogesterone-acetate",
    "methylphenidate": "methylphenidate-hydrochloride",
    "mometasone": "mometasone-furoate",
    "olmesartan": "olmesartan-medoxomil",
    "risedronate": "risedronate-sodium",
    "terbutaline": "terbutaline-sulfate",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def bnf_url_for(generic_name: str) -> str:
    slug = BNF_SLUGS.get(generic_name, slugify(generic_name))
    return f"{BNF_BASE}/{slug}/"


def main() -> None:
    with INPUT_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    changed = 0
    for row in rows:
        if row["review_status"] == "needs_nhs_manual_review":
            row["source_name"] = "BNF NICE candidate"
            row["source_url"] = bnf_url_for(row["generic_name"])
            row["review_status"] = "bnf_url_candidate_needs_manual_review"
            changed += 1

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Added BNF candidate URLs for {changed} rows")
    print("BNF blocked automated checks, so manually open candidate URLs before marking rows reviewed.")


if __name__ == "__main__":
    main()
