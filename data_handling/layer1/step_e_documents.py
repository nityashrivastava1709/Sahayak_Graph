"""
STEP E  -  One row per required document, with standard document names.

Input : data/interim/schemes_step_a.csv
Output: data/processed/documents.csv
        data/interim/documents_unmapped_review.csv  (names the rules below did not recognise)

Why standard names matter
  In the knowledge graph there must be ONE shared "Aadhaar Card" node. If the data says
  "Aadhaar", "Aadhar card" and "Jan Aadhaar Card", those would become three nodes and
  document reasoning would break.

How it works
  1. The LLM reads each scheme's `documents` text and lists the documents (it drops notes
     such as "in the absence of Aadhaar, provide ..." and "any other document").
  2. CANONICAL_RULES below maps every document name to one standard name.
  3. Names that match no rule are kept as written and listed in the review file.
     Add a rule for them to CANONICAL_RULES and run this script again (it is free,
     because the LLM answers are cached).

Run:  python step_e_documents.py
"""
import re

import pandas as pd

import config
from llm_utils import cached_llm_json

# (regular expression, standard name). The FIRST matching rule wins, so order matters.
CANONICAL_RULES = [
        # Institution and education documents
    (r"head of (the )?(school|institution)|head of school/college|institute verification",
     "Institution Verification Certificate"),
    (r"marks sheet|statement of marks|high school certificate|sslc|s\.s\.l\.c\.|transfer certificate|leaving certificate",
     "Marksheet / Educational Certificate"),
    (r"educational details|educational documents|school details|teacher details|details of students",
     "Educational Details"),
    (r"ph\.?d\.? registration", "PhD Registration Certificate"),

    # Employment and service documents
    (r"90 days work certificate", "90 Days Work Certificate"),
    (r"ex-serviceman discharge book|service particulars",
     "Ex-serviceman Service Record"),
    (r"family planning certificate", "Family Planning Certificate"),
    (r"fitness certificate|medical board certificate",
     "Medical / Fitness Certificate"),

    # Financial documents
    (r"college-wise|head-wise expenditure|statement of accounts|uc/auc|unspent balance",
     "Financial Statement / Utilization Certificate"),
    (r"claim voucher|gst voucher|proof of deposit|neft details|payment details",
     "Payment / Transaction Proof"),
    (r"income tax return", "Income Tax Return"),
    (r"proof of loan utilization", "Loan Utilization Proof"),

    # Identity and family documents
    (r"legal guardian certificate", "Legal Guardian Certificate"),
    (r"parent identification documents", "Parent Identity Proof"),
    (r"national food security scheme card",
     "National Food Security Scheme Card"),
    (r"proof of birth", "Birth Certificate / Age Proof"),
    (r"statement of next kin", "Next of Kin Statement"),

    # Scheme-specific supporting documents
    (r"certificate of not availing benefit|non-availing of other scholarship",
     "Non-Availing of Benefits Certificate"),
    (r"certificate issued by competent authority",
     "Competent Authority Certificate"),
    (r"certificate of area and crop|proof of seed production",
     "Agriculture Production Proof"),
    (r"proof of registration of cooperative society",
     "Cooperative Society Registration Proof"),
    (r"proof of cpt clearance|graduation certificate",
     "Educational Qualification Proof"),
    (r"proof of daughter's name in official records",
     "Daughter's Name in Official Records"),
    (r"newspaper report of the missing person",
     "Missing Person Newspaper Report"),
    (r"statement of persons who have lastly seen",
     "Witness Statement"),
    (r"statement of panchayathars|panchayathar.s letter",
     "Panchayat Statement / Letter"),
    (r"resolution copy of general body meeting",
     "General Body Resolution"),
    (r"invitation/hosting letter", "Invitation / Hosting Letter"),
    (r"nomination form", "Nomination Form"),
    (r"non-judicial stamp", "Non-Judicial Stamp Paper"),
    (r"newspaper cutting", "Newspaper Cutting"),
    (r"powerpoint presentation", "PowerPoint Presentation"),
    (r"project viability and technical feasibility",
     "Project Viability / Feasibility Certificate"),
    (r"energization letter", "Energization Letter"),
    (r"frp notification", "FRP Notification"),
    (r"immobilizations certificate", "Immobilization Certificate"),
    (r"permit", "Permit"),
    (r"remarkable achievements|professional awards|recognition/fellowship",
     "Awards / Achievements Proof"),
    (r"residence in a recognized hostel", "Recognized Hostel Residence Proof"),
    (r"details of nearest sastraposhini school",
     "Nearest Sastraposhini School Details"),
    (r"list of prospective beneficiaries|detailed list of beneficiaries",
     "Beneficiary List"),
    (r"demand/proposal", "Demand / Proposal"),
    (r"public information certificate", "Public Information Certificate"),
    (r"proof of registration and fees paid",
     "Course Registration / Fee Proof"),

    # Additional mappings for unmatched documents
    (r"curriculum vitae|bio.?data", "Curriculum Vitae / Bio-data"),
    (r"\bnoc\b|no objection certificate", "No Objection Certificate"),
    (r"divorce certificate|proof of divorce|divorce deed", "Divorce Certificate"),
    (r"legal heir(ship)? certificate|statement of next of kin",
     "Legal Heir / Next of Kin Certificate"),
    (r"fir|first information report|police complaint|gd entry|csr",
     "Police Report / FIR"),
    (r"post.?mortem report|inquest report|viscera report",
     "Post-mortem / Inquest Documents"),
    (r"discharge summary|op ticket|medical records|medical reports|lab reports",
     "Medical Records / Reports"),
    (r"original medical bills|medical bills|bills/vouchers|original bills",
     "Medical Bills / Vouchers"),
    (r"experience certificate|employment certificate",
     "Employment / Experience Certificate"),
    (r"labour card", "Labour Card"),
    (r"family id|parivaar pahchan patra|parivar register",
     "Family ID / Family Register"),
    (r"economic weaker section|ews certificate",
     "EWS Certificate"),
    (r"non-reserved category certificate",
     "Non-Reserved Category Certificate"),
    (r"sc/st/ph certificate|sc/st category certificate",
     "SC/ST Category Certificate"),
    (r"pension payment order|\bppo\b", "Pension Payment Order"),
    (r"loan sanction letter", "Loan Sanction Letter"),
    (r"no.?objection|consent letter|letter of consent|consent form",
     "Consent / No Objection Document"),
    (r"recommendation letter|recommendation from gram panchayat",
     "Recommendation Letter"),
    (r"project proposal|demand/proposal|complete proposal|visiting proposal",
     "Project Proposal"),
    (r"project report", "Project Report"),
    (r"audit reports|audited balance sheet|profit.?and.?loss account",
     "Audit / Financial Statements"),
    (r"cultivation certificate|certificate of area and crop|proof of cultivation",
     "Cultivation Certificate"),
    (r"proof of marriage|proof of date of marriage",
     "Marriage Certificate"),
    (r"proof of annual family income",
     "Income Certificate"),

    # Existing rules continue below
    (r"samagra", "Samagra ID"),
    (r"samagra", "Samagra ID"),
    (r"address proof|proof of address", "Address Proof"),
    (r"bank|passbook|cancelled cheque|ifsc", "Bank Account Details"),
    (r"identity|id proof|proof of id", "Identity Proof"),
    (r"aadhaar|aadhar|\buid\b", "Aadhaar Card"),
    (r"passport.?size|photograph|\bphoto\b", "Passport-size Photograph"),
    (r"\bpassport\b", "Passport"),
    (r"mgnrega|mnrega|nrega|job card", "MGNREGA Job Card"),
    (r"proof of being a farmer|farmer (proof|cert|id|registration)|kisan (card|id)", "Proof of Farmer Status"),
    (r"no.?claim", "No Claim Certificate"),
    (r"income (cert|proof)|proof of income|salary (cert|slip)", "Income Certificate"),
    (r"caste|community cert|tribe cert|sc/st cert", "Caste Certificate"),
    (r"domicile|residen(ce|tial|t) (cert|proof)|proof of residen|permanent resident|nativity|local resident",
     "Domicile/Residence Certificate"),
    (r"birth cert|date of birth|proof of age|age proof|age cert|school leaving",
     "Birth Certificate / Age Proof"),
    (r"\bpan\b", "PAN Card"),
    (r"voter|driving licen|\bid card\b", "Identity Proof"),
    (r"mark.?sheet|marks? card|educational (qualification|cert)|education cert|academic|result|passing cert|degree|diploma",
     "Marksheet / Educational Certificate"),
    (r"admission|bonafide|bona fide|institution cert|study cert|enrol", "Proof of Admission / Bonafide Certificate"),
    (r"fee (receipt|structure|payment)|receipt", "Fee Receipt"),
    (r"disab|handicap|\bpwd\b|udid", "Disability Certificate"),
    (r"ration card|\bbpl\b|below poverty|antyodaya", "Ration Card / BPL Card"),
    (r"death cert", "Death Certificate"),
    (r"marriage cert|wedding|marriage invitation", "Marriage Certificate"),
    (r"widow", "Widow Certificate"),
    (r"land|khasra|khatauni|patta|jamabandi|7/12|lease", "Land Ownership Documents"),
    (r"medical cert|pregnan|mcp|health cert", "Medical Certificate"),
    (r"declaration|undertaking|affidavit", "Self Declaration / Affidavit"),
    (r"application form", "Application Form"),
    (r"registration cert|membership|society regist|licen[cs]e", "Registration / Membership Certificate"),
]

# Entries matching any of these are NOT documents (contact details, "any other document", ...).
# They are dropped, and counted in the report so nothing disappears silently.
DROP_PATTERNS = [
    r"mobile (number|no\b)", r"phone (number|no\b)", r"e-?mail",
    r"^(any )?other( necessary| required| relevant| supporting)? documents?( as required)?$",
    r"^any other", r"as (may be )?required", r"^none$", r"not required",
]


def is_not_a_document(raw_name: str) -> bool:
    lowered = raw_name.lower().strip()
    return any(re.search(p, lowered) for p in DROP_PATTERNS)


SYSTEM_PROMPT = """You read the 'documents required' text of an Indian government scheme and list the documents an applicant must submit.

Rules:
1. Return ONLY a JSON list. Each item: {"name": "<short document name>", "conditional": true/false}.
2. One item per distinct document. Keep names short (e.g. "Income Certificate", "Aadhaar Card", "Bank Passbook").
3. "conditional" is true only when the text says "if applicable", "if any", "for SC/ST applicants only" and similar.
4. Do NOT list instructions or notes (for example "in the absence of Aadhaar, provide ...", "self-attested copies", "any other document as required").
5. If a line offers alternatives for the same purpose (e.g. "Identity proof (Aadhaar, Voter ID)"), list ONE item for the purpose, e.g. "Identity Proof".
6. Do not invent documents that are not in the text."""


# The exact JSON shape Gemini must return (structured output).
DOCUMENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "short document name taken from the text"},
            "conditional": {"type": "boolean"},
        },
        "required": ["name", "conditional"],
    },
}


def canonical_name(raw_name: str):
    """Return (standard_name, matched_a_rule)."""
    lowered = raw_name.lower()
    for pattern, standard in CANONICAL_RULES:
        if re.search(pattern, lowered):
            return standard, True
    # capitalise each word WITHOUT str.title(), which turns  Panchayathar’s  into  Panchayathar’S
    return " ".join(w.capitalize() for w in raw_name.split()), False


def main():
    schemes = pd.read_csv(config.STEP_A_OUT, dtype=str).fillna("")
    rows, dropped = [], []
    for i, (_, s) in enumerate(schemes.iterrows(), start=1):
        print(f"[{i}/{len(schemes)}] {s['scheme_id']} {s['scheme_name'][:60]}")
        user = f"Scheme: {s['scheme_name']}\n\nDocuments text:\n{s['documents']}\n\nReturn the JSON list now."
        items = cached_llm_json("documents", s["scheme_id"], SYSTEM_PROMPT, user,
                                schema=DOCUMENTS_SCHEMA)
        if not isinstance(items, list):
            print("   no usable answer, skipped")
            continue
        for it in items:
            if isinstance(it, str):                      # tolerate a plain list of strings
                it = {"name": it, "conditional": False}
            if not isinstance(it, dict) or not str(it.get("name", "")).strip():
                continue
            raw = str(it["name"]).strip()
            if is_not_a_document(raw):
                dropped.append((s["scheme_id"], raw))
                continue
            standard, matched = canonical_name(raw)
            rows.append({"scheme_id": s["scheme_id"], "document_name": standard,
                         "raw_name": raw, "is_conditional": bool(it.get("conditional", False)),
                         "matched_rule": matched})

    docs = pd.DataFrame(rows)
    # one scheme should not list the same standard document twice
    docs = docs.drop_duplicates(["scheme_id", "document_name"]).reset_index(drop=True)
    docs[["scheme_id", "document_name", "raw_name", "is_conditional"]].to_csv(
        config.DOCUMENTS_CSV, index=False, encoding="utf-8-sig")

    unmatched = docs[~docs["matched_rule"]][["document_name", "raw_name"]]
    unmatched_path = config.INTERIM_DIR / "documents_unmapped_review.csv"
    unmatched.groupby("document_name").size().sort_values(ascending=False) \
        .rename("count").to_csv(unmatched_path, encoding="utf-8-sig")

    print(f"\nSaved {len(docs)} scheme-document links for {docs['scheme_id'].nunique()} schemes "
          f"-> {config.DOCUMENTS_CSV}")
    print(f"\nStandard documents (master list), most common first:")
    print(docs["document_name"].value_counts().head(30).to_string())
    if dropped:
        print(f"\nDropped {len(dropped)} entries that are not documents, for example: "
              f"{[d[1] for d in dropped[:5]]}")
        pd.DataFrame(dropped, columns=["scheme_id", "dropped_text"]).to_csv(
            config.INTERIM_DIR / "documents_dropped.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(unmatched)} links did not match any rule. Review {unmatched_path}, "
          "add rules to CANONICAL_RULES, then re-run this script.")


if __name__ == "__main__":
    main()