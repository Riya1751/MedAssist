"""Classical (non-LLM) text processing + rule-based fallback classifiers.

Why keep this when we have an LLM?
  1. It works offline / when the API key is missing or rate-limited.
  2. It gives the LLM verified "hints" (hybrid approach = fewer hallucinations).
  3. It is the "classical ML/NLP" part of your project report.

Techniques used: normalization, tokenization, fuzzy string matching (edit-distance
style via difflib), regex entity extraction, and a rule/knowledge-base classifier.
"""
import re
from collections import Counter
from difflib import get_close_matches

# ------------------------------------------------------------------ cleaning

def clean_ocr_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)   # drop garbled / non-printable chars
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


# ----------------------------------------------------------- safety keywords

EMERGENCY_WORDS = ["chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
                   "suicide", "kill myself", "overdose", "unconscious", "seizure",
                   "severe bleeding", "stroke", "face drooping"]

CRISIS_WORDS = ["suicide", "kill myself", "end my life", "want to die", "self harm", "self-harm", "hurt myself"]

def is_crisis(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in CRISIS_WORDS)


def is_emergency(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in EMERGENCY_WORDS)


# ------------------------------------------------------- medicine knowledge base
# Tiny demo KB. Extend it freely (name/brand -> generic, purpose, condition tag).
MEDICINE_KB = {
    "paracetamol": ("Paracetamol", "Reduces fever and mild pain", "fever"),
    "dolo": ("Paracetamol", "Reduces fever and mild pain", "fever"),
    "crocin": ("Paracetamol", "Reduces fever and mild pain", "fever"),
    "ibuprofen": ("Ibuprofen", "Pain, inflammation and fever relief", "pain"),
    "diclofenac": ("Diclofenac", "Pain and inflammation relief", "pain"),
    "amoxicillin": ("Amoxicillin", "Antibiotic for bacterial infections", "infection"),
    "azithromycin": ("Azithromycin", "Antibiotic for bacterial infections", "infection"),
    "augmentin": ("Amoxicillin + Clavulanic acid", "Antibiotic for bacterial infections", "infection"),
    "cetirizine": ("Cetirizine", "Antihistamine for allergy, sneezing, itching", "allergy"),
    "levocetirizine": ("Levocetirizine", "Antihistamine for allergy and cold symptoms", "allergy"),
    "montelukast": ("Montelukast", "Prevents allergic wheezing / asthma symptoms", "allergy"),
    "pantoprazole": ("Pantoprazole", "Reduces stomach acid (acidity, reflux)", "acidity"),
    "omeprazole": ("Omeprazole", "Reduces stomach acid (acidity, reflux)", "acidity"),
    "ranitidine": ("Ranitidine", "Reduces stomach acid", "acidity"),
    "metformin": ("Metformin", "Lowers blood sugar in type 2 diabetes", "diabetes"),
    "glimepiride": ("Glimepiride", "Lowers blood sugar in type 2 diabetes", "diabetes"),
    "amlodipine": ("Amlodipine", "Lowers blood pressure", "hypertension"),
    "telmisartan": ("Telmisartan", "Lowers blood pressure", "hypertension"),
    "losartan": ("Losartan", "Lowers blood pressure", "hypertension"),
    "atorvastatin": ("Atorvastatin", "Lowers cholesterol", "cholesterol"),
    "rosuvastatin": ("Rosuvastatin", "Lowers cholesterol", "cholesterol"),
    "ors": ("ORS", "Replaces fluids and salts lost in diarrhoea/vomiting", "dehydration"),
}

CONDITION_LABELS = {
    "fever": "Fever (possibly viral or infection-related)",
    "pain": "Pain / inflammation",
    "infection": "Bacterial infection",
    "allergy": "Allergy or cold-related symptoms",
    "acidity": "Acidity / acid reflux (gastric issue)",
    "diabetes": "Diabetes (blood sugar control)",
    "hypertension": "Hypertension (high blood pressure)",
    "cholesterol": "High cholesterol",
    "dehydration": "Dehydration / stomach upset",
}

# Short, safe, generic tips per condition tag (used only in rule-based fallback)
CONDITION_TIPS = {
    "fever": (["Rest well", "Drink plenty of fluids", "Monitor temperature"],
              ["Don't self-medicate extra doses", "Don't skip follow-up if fever lasts >3 days"]),
    "infection": (["Finish the full antibiotic course", "Take medicine at fixed times"],
                  ["Don't stop early when you feel better", "Don't share antibiotics"]),
    "allergy": (["Avoid known triggers (dust, pollen)", "Keep your room clean"],
                ["Don't drive if the medicine makes you drowsy"]),
    "acidity": (["Eat smaller, regular meals", "Take medicine before food if advised"],
                ["Avoid spicy, oily food", "Don't lie down right after eating"]),
    "diabetes": (["Walk 30 min daily", "Eat high-fibre, low-sugar meals", "Check sugar regularly"],
                 ["Avoid sugary drinks", "Don't skip meals while on medicine"]),
    "hypertension": (["Reduce salt", "Exercise regularly", "Check BP at home"],
                     ["Don't stop medicine suddenly", "Avoid smoking and too much caffeine"]),
    "cholesterol": (["Eat oats, fruits, vegetables", "Exercise regularly"],
                    ["Avoid fried and processed food"]),
    "pain": (["Rest the affected area"], ["Avoid taking painkillers on an empty stomach"]),
    "dehydration": (["Sip ORS often", "Eat light food"], ["Avoid street food and sugary drinks"]),
}

DOSAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|iu)\b", re.I)
FREQ_RE = re.compile(r"\b(od|bd|bid|tds|tid|qid|hs|sos|\d-\d-\d(?:-\d)?|once daily|twice daily)\b", re.I)


def find_medicines(text: str) -> list[dict]:
    """Fuzzy-match OCR words against the KB (handles OCR typos like 'paracetmol')."""
    found, seen = [], set()
    for line in text.splitlines():
        for tok in tokenize(line):
            match = get_close_matches(tok, MEDICINE_KB.keys(), n=1, cutoff=0.82)
            if not match or match[0] in seen:
                continue
            seen.add(match[0])
            generic, purpose, tag = MEDICINE_KB[match[0]]
            dose = DOSAGE_RE.search(line)
            freq = FREQ_RE.search(line)
            found.append({"name": generic, "dosage": dose.group(0) if dose else "",
                          "frequency": freq.group(0) if freq else "",
                          "purpose": purpose, "tag": tag})
    return found


def rule_based_prescription(text: str) -> dict:
    """Fallback 'classifier': majority vote of condition tags from matched medicines."""
    meds = find_medicines(text)
    votes = Counter(m["tag"] for m in meds)
    top = votes.most_common(1)[0][0] if votes else None
    dos, donts = [], []
    for tag in votes:
        d, n = CONDITION_TIPS.get(tag, ([], []))
        dos += d; donts += n
    return {
        "condition": CONDITION_LABELS.get(top, "Could not determine from the text"),
        "medicines": [{k: v for k, v in m.items() if k != "tag"} for m in meds],
        "dos": dos[:5], "donts": donts[:5], "diet_lifestyle": [], "prevention": [],
        "summary": "Basic rule-based reading (AI service unavailable). Please confirm with your doctor.",
    }


# ------------------------------------------------------------- lab reference ranges
# General adult ranges for demonstration; real ranges vary by lab, age and sex.
LAB_RANGES = {
    "hemoglobin": dict(label="Hemoglobin", unit="g/dL", low=12.0, high=17.0,
                       pat=r"(?<!glycosylated\s)(?:hemoglobin|haemoglobin)(?!\s*a1c)|\bhb\b"),
    "wbc": dict(label="WBC (white cells)", unit="/uL", low=4000, high=11000,
                pat=r"\bwbc\b|total\s*leucocyte\s*count|total\s*leukocyte\s*count|\btlc\b"),
    "platelets": dict(label="Platelets", unit="/uL", low=150000, high=450000,
                      pat=r"platelet(?:\s*count)?s?"),
    "rbc": dict(label="RBC (red cells)", unit="million/uL", low=4.2, high=5.9,
                pat=r"\brbc\b|red\s*blood\s*cell"),
    "total_cholesterol": dict(label="Total Cholesterol", unit="mg/dL", low=None, high=200,
                              pat=r"total\s*cholesterol|cholesterol,?\s*total"),
    "ldl": dict(label="LDL Cholesterol", unit="mg/dL", low=None, high=100, pat=r"\bldl\b"),
    "hdl": dict(label="HDL Cholesterol", unit="mg/dL", low=40, high=None, pat=r"\bhdl\b"),
    "triglycerides": dict(label="Triglycerides", unit="mg/dL", low=None, high=150, pat=r"triglycerides?"),
    "fasting_glucose": dict(label="Fasting Blood Sugar", unit="mg/dL", low=70, high=100,
                            pat=r"fasting\s*(?:blood\s*)?(?:sugar|glucose)|\bfbs\b"),
    "hba1c": dict(label="HbA1c", unit="%", low=None, high=5.7,
                  pat=r"hba1c|glycosylated\s*hemoglobin|glycated\s*hemoglobin"),
    "creatinine": dict(label="Creatinine", unit="mg/dL", low=0.6, high=1.3, pat=r"creatinine"),
    "tsh": dict(label="TSH", unit="uIU/mL", low=0.4, high=4.0, pat=r"\btsh\b"),
}


def _normalize_units(key: str, v: float) -> float:
    """Reports print WBC as '7.5' (x10^3) or platelets as '2.5' (lakh). Convert to /uL."""
    if key == "wbc" and v < 100:
        return v * 1000
    if key == "platelets":
        if v < 10:
            return v * 100000      # lakh style: 2.5 -> 250000
        if v < 1000:
            return v * 1000        # x10^3 style: 250 -> 250000
    return v


def _fmt_range(low, high) -> str:
    if low is not None and high is not None:
        return f"{low} - {high}"
    return f"< {high}" if high is not None else f"> {low}"


def rule_based_lab(text: str) -> dict:
    """Regex-extract known tests, compare to reference range, flag low/high."""
    params = []
    for key, r in LAB_RANGES.items():
        # find the test name, then the first number that follows it on the same line
        m = re.search(rf"(?:{r['pat']})[^0-9\n]{{0,40}}?(\d+(?:\.\d+)?)", text, re.I)
        if not m:
            continue
        value = _normalize_units(key, float(m.group(1)))
        status = "normal"
        if r["low"] is not None and value < r["low"]:
            status = "low"
        elif r["high"] is not None and value > r["high"]:
            status = "high"
        expl = ("Within the usual range." if status == "normal" else
                f"{r['label']} is {status} compared with the usual range. Discuss with your doctor.")
        params.append({"name": r["label"], "value": value, "unit": r["unit"],
                       "reference_range": _fmt_range(r["low"], r["high"]),
                       "status": status, "explanation": expl})
    abnormal = [p["name"] for p in params if p["status"] != "normal"]
    summary = (f"{len(params)} values read. Outside range: {', '.join(abnormal)}."
               if abnormal else f"{len(params)} values read; all within the usual range."
               ) if params else "No known test values could be read."
    return {"parameters": params, "summary": summary, "dos": [], "donts": [],
            "diet_lifestyle": [], "prevention": [], "follow_up": ["Review these results with your doctor."]}