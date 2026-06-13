import argparse
import csv
import html
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_INPUT = "medicines_model_ready_class_checked.csv"
DEFAULT_OUTPUT = "medicine_semantic_consistency_report.csv"
SOURCE_CACHE_DIR = Path("source_pages")

STOPWORDS = {
    "a", "about", "and", "are", "as", "ask", "because", "before", "by", "can",
    "doctor", "for", "from", "have", "if", "in", "is", "it", "medicine",
    "medicines", "not", "of", "or", "only", "pharmacist", "prescribed",
    "some", "take", "that", "the", "this", "to", "use", "used", "with", "you",
}

CONCEPT_GROUPS = {
    "allergy": ["allergy", "allergic", "hay fever", "rhinitis", "urticaria", "hives", "itching", "itchy", "sneezing", "runny nose"],
    "angina": ["angina", "chest pain"],
    "antibiotic": ["antibiotic", "bacterial infection", "bacteria"],
    "asthma_breathing": ["asthma", "copd", "wheezing", "breathing difficulty", "bronchospasm", "shortness of breath"],
    "blood_clot": ["blood clot", "blood clots", "clot", "clots", "thrombosis", "embolism", "stroke", "anticoagulant", "dvt", "pulmonary embolism"],
    "blood_pressure": ["blood pressure", "hypertension"],
    "blood_sugar": ["blood sugar", "glucose", "diabetes", "diabetic", "glycaemic", "glycemic"],
    "bone_health": ["bone", "bones", "osteoporosis", "fracture", "calcium", "vitamin d"],
    "cholesterol": ["cholesterol", "lipid", "statin", "cardiovascular", "heart attack", "stroke"],
    "constipation": ["constipation", "laxative", "bowel"],
    "contraception": ["contraception", "contraceptive", "pregnancy prevention", "emergency contraception"],
    "depression_anxiety": ["depression", "anxiety", "panic", "mood", "ocd", "post-traumatic stress", "ptsd"],
    "diarrhoea": ["diarrhoea", "diarrhea", "loose stools"],
    "drowsiness": ["drowsiness", "drowsy", "sleepy", "sedation", "sedating"],
    "eczema_skin": ["eczema", "skin", "dermatitis", "psoriasis", "rash", "inflammation"],
    "epilepsy": ["epilepsy", "seizure", "seizures", "fits", "antiepileptic"],
    "fungal": ["fungal", "fungus", "thrush", "antifungal"],
    "glaucoma_eye": ["glaucoma", "ocular hypertension", "eye pressure", "eye drops"],
    "gout": ["gout", "uric acid"],
    "heart_failure": ["heart failure", "fluid build-up", "oedema", "edema"],
    "heart_rhythm": ["heart rhythm", "arrhythmia", "atrial fibrillation"],
    "heartburn_reflux": ["heartburn", "reflux", "indigestion", "stomach acid", "ulcer", "ulcers"],
    "infection": ["infection", "infections", "infected"],
    "inflammation": ["inflammation", "inflammatory", "swelling", "anti-inflammatory"],
    "kidney": ["kidney", "renal"],
    "migraine": ["migraine", "headache"],
    "nausea_vomiting": ["nausea", "vomiting", "sickness", "antiemetic"],
    "nerve_pain": ["nerve pain", "neuropathic pain", "neuralgia"],
    "pain_fever": ["pain", "ache", "aches", "fever", "temperature", "toothache", "period pain"],
    "parkinson": ["parkinson", "parkinson's"],
    "pregnancy_breastfeeding": ["pregnant", "pregnancy", "breastfeeding", "fertility"],
    "prostate_urinary": ["prostate", "urinary", "bladder", "urine", "overactive bladder"],
    "side_effects": ["side effects", "side-effect", "adverse effects"],
    "thyroid": ["thyroid", "hypothyroidism", "hyperthyroidism", "underactive thyroid", "overactive thyroid"],
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def html_to_text(markup: str) -> str:
    markup = re.sub(r"(?is)<script.*?</script>", " ", markup)
    markup = re.sub(r"(?is)<style.*?</style>", " ", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    return re.sub(r"\s+", " ", html.unescape(markup)).strip()


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
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 medicine-label-helper semantic review"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as error:
        return "", f"http_{error.code}"
    except URLError as error:
        return "", f"url_error:{error.reason}"
    except TimeoutError:
        return "", "timeout"
    return html_to_text(body), "fetched"


def get_source_text(row: dict, timeout: int) -> tuple[str, str]:
    cached = read_cached_source(row)
    if cached:
        return cached, "cached_text"
    return fetch_source_text(row["source_url"], timeout)


def tokens(value: str) -> set[str]:
    found = re.findall(r"[a-z0-9]+", value.lower())
    return {token for token in found if len(token) > 2 and token not in STOPWORDS}


def concept_hits(value: str) -> set[str]:
    lower = value.lower()
    hits = set()
    for concept, phrases in CONCEPT_GROUPS.items():
        if any(phrase in lower for phrase in phrases):
            hits.add(concept)
    return hits


def semantic_score(source_text: str, field_text: str) -> tuple[float, str, str]:
    field_tokens = tokens(field_text)
    source_tokens = tokens(source_text)
    token_score = len(field_tokens & source_tokens) / len(field_tokens) if field_tokens else 1.0

    field_concepts = concept_hits(field_text)
    source_concepts = concept_hits(source_text)
    concept_score = len(field_concepts & source_concepts) / len(field_concepts) if field_concepts else token_score

    score = round(max(token_score, concept_score), 3)
    missing_tokens = ";".join(sorted(field_tokens - source_tokens)[:12])
    missing_concepts = ";".join(sorted(field_concepts - source_concepts))
    return score, missing_tokens, missing_concepts


def flags(common_score: float, safety_score: float, source_status: str) -> str:
    result = []
    if source_status not in {"fetched", "cached_text"}:
        result.append("source_unavailable")
    if common_score < 0.45:
        result.append("review_common_uses")
    if safety_score < 0.35:
        result.append("review_safety_warning")
    return ";".join(result) if result else "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic consistency check for medicine CSV rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    report_rows = []
    for index, row in enumerate(rows, start=1):
        source_text, source_status = get_source_text(row, args.timeout)
        common_score, missing_common_tokens, missing_common_concepts = semantic_score(source_text, row["common_uses"])
        safety_score, missing_safety_tokens, missing_safety_concepts = semantic_score(source_text, row["safety_warning"])
        row_flags = flags(common_score, safety_score, source_status)

        report_rows.append(
            {
                "generic_name": row["generic_name"],
                "source_name": row["source_name"],
                "source_url": row["source_url"],
                "review_status": row["review_status"],
                "source_status": source_status,
                "common_uses_semantic_score": common_score,
                "safety_warning_semantic_score": safety_score,
                "missing_common_tokens": missing_common_tokens,
                "missing_common_concepts": missing_common_concepts,
                "missing_safety_tokens": missing_safety_tokens,
                "missing_safety_concepts": missing_safety_concepts,
                "flags": row_flags,
            }
        )
        print(f"{index:03d}/{len(rows)} {row['generic_name']}: {row_flags}")

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote {args.output}")
    print("Semantic scores are triage signals, not clinical certification.")


if __name__ == "__main__":
    main()
