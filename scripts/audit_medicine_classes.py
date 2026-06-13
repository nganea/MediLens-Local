import csv


INPUT_FILE = "medicines_model_ready.csv"
OUTPUT_FILE = "medicines_model_ready_class_checked.csv"
REPORT_FILE = "medicine_class_audit_report.csv"


CLASS_FIXES = {
    "paracetamol": "Analgesic and antipyretic",
    "co-codamol": "Combination analgesic",
    "gabapentin": "Antiepileptic and neuropathic pain medicine",
    "pregabalin": "Antiepileptic and neuropathic pain medicine",
    "ezetimibe": "Cholesterol absorption inhibitor",
    "metformin": "Biguanide antidiabetic",
    "insulin glargine": "Long-acting insulin",
    "insulin aspart": "Rapid-acting insulin",
    "beclometasone": "Corticosteroid",
    "budesonide": "Corticosteroid",
    "fluticasone": "Corticosteroid",
    "chlorphenamine": "Sedating antihistamine",
    "gaviscon": "Alginate antacid",
    "hyoscine hydrobromide": "Antimuscarinic antiemetic",
    "co-amoxiclav": "Penicillin antibiotic combination",
    "nitrofurantoin": "Antibiotic",
    "metronidazole": "Nitroimidazole antibiotic",
    "atomoxetine": "Noradrenaline reuptake inhibitor",
    "sodium valproate": "Antiepileptic and mood stabiliser",
    "tamsulosin": "Alpha blocker",
    "finasteride": "5-alpha-reductase inhibitor",
    "mirabegron": "Beta-3 adrenergic agonist",
    "ethinylestradiol levonorgestrel": "Combined hormonal contraceptive",
    "levonorgestrel": "Progestogen emergency contraceptive",
    "ferrous sulfate": "Iron supplement",
    "folic acid": "Vitamin supplement",
    "methotrexate": "DMARD and antimetabolite",
    "azathioprine": "Immunosuppressant antimetabolite",
    "mesalazine": "Aminosalicylate",
    "benzoyl peroxide": "Topical acne treatment",
    "mometasone": "Corticosteroid",
    "latanoprost": "Prostaglandin analogue eye drop",
    "dorzolamide": "Carbonic anhydrase inhibitor eye drop",
    "chloramphenicol": "Antibiotic",
}


def main() -> None:
    with open(INPUT_FILE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    report = []
    for row in rows:
        generic_name = row["generic_name"]
        old_class = row["notes"]
        new_class = CLASS_FIXES.get(generic_name, old_class)
        status = "updated" if new_class != old_class else "kept"
        row["notes"] = new_class
        report.append(
            {
                "generic_name": generic_name,
                "old_notes": old_class,
                "new_notes": new_class,
                "status": status,
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    print(f"Updated {OUTPUT_FILE}")
    print(f"Wrote {REPORT_FILE}")
    print(f"Updated class labels: {sum(1 for row in report if row['status'] == 'updated')}")


if __name__ == "__main__":
    main()
