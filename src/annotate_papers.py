import re
from pathlib import Path

import pandas as pd
import spacy

DATA_RAW = Path("data/raw/takayasu_europepmc_raw.csv")
DATA_PROCESSED_DIR = Path("data/processed")
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

IMAGING_KEYWORDS = {
    "ct": [" CT ", "CT angiography", "computed tomography"],
    "mri": [" MRI ", "magnetic resonance imaging"],
    "pet": [" PET ", "positron emission tomography"],
    "ultrasound": ["ultrasound", "Doppler"]
}

TREATMENT_KEYWORDS = {
    "steroids": ["steroid", "prednisone", "prednisolone", "glucocorticoid"],
    "biologics": ["infliximab", "tocilizumab", "adalimumab", "etanercept"],
    "surgery": ["bypass surgery", "revascularization", "surgical repair", "stent", "stenting"],
}

COMPLICATION_KEYWORDS = {
    "aortic_aneurysm": ["aortic aneurysm"],
    "coronary_involvement": ["coronary artery", "coronary involvement"],
}

AGE_PATTERN = re.compile(r"\b(\d{1,2})-year-old\b|\bage[d]?\s+(\d{1,2})\b", re.IGNORECASE)

def has_any(text, patterns):
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)

def extract_age_range(text):
    ages = []
    for m in AGE_PATTERN.finditer(text):
        g1, g2 = m.groups()
        if g1:
            ages.append(int(g1))
        elif g2:
            ages.append(int(g2))
    if not ages:
        return None, None
    return min(ages), max(ages)

def annotate_row(row):
    abstract = row.get("abstract") or ""
    doc = nlp(abstract)

    # imaging
    for key, patterns in IMAGING_KEYWORDS.items():
        row[f"mentions_{key}"] = has_any(abstract, patterns)

    # treatment
    for key, patterns in TREATMENT_KEYWORDS.items():
        row[f"mentions_{key}"] = has_any(abstract, patterns)

    # complications
    for key, patterns in COMPLICATION_KEYWORDS.items():
        row[f"mentions_{key}"] = has_any(abstract, patterns)

    # demographics
    age_min, age_max = extract_age_range(abstract)
    row["age_min"] = age_min
    row["age_max"] = age_max

    text_lower = abstract.lower()
    row["mentions_female"] = any(w in text_lower for w in ["female", "woman", "women", "girl"])
    row["mentions_male"] = any(w in text_lower for w in ["male", "man", "men", "boy"])

    return row

def main():
    df = pd.read_csv(DATA_RAW)
    annotated = df.apply(annotate_row, axis=1)
    out_path = DATA_PROCESSED_DIR / "takayasu_annotated.csv"
    annotated.to_csv(out_path, index=False)
    print(f"Saved annotated dataset to {out_path}")

if __name__ == "__main__":
    main()
