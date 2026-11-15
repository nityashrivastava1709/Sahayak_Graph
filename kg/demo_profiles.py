"""
demo_profiles.py  -  hand-made profiles for trying the graph.

Each profile uses the attribute names from the Layer 1 vocabulary
(age, gender, annual_income, state, occupation, social_category, disability, marital_status,
education_level, residence_type, owns_land, land_holding_acres, bpl_status, pregnancy_status).
Leave an attribute out when the person has not told you: "unknown" is handled properly
(the scheme is reported as 'possible' and the missing attribute is listed).
"""
PROFILES = {
    "farmer_mp": {            # woman farmer, Madhya Pradesh, small landholding
        "age": 34, "gender": "female", "annual_income": 180000, "state": "Madhya Pradesh",
        "occupation": "farmer", "social_category": "OBC", "owns_land": "yes",
        "land_holding_acres": 3, "residence_type": "rural", "bpl_status": "yes", "marital_status": "married",
    },
    "student_rajasthan": {    # school student, Rajasthan
        "age": 17, "gender": "male", "annual_income": 200000, "state": "Rajasthan",
        "occupation": "student", "education_level": "class_11_12", "social_category": "SC",
        "residence_type": "rural",
    },
    "widow_up": {             # widow, Uttar Pradesh, low income
        "age": 52, "gender": "female", "marital_status": "widowed", "annual_income": 90000,
        "state": "Uttar Pradesh", "bpl_status": "yes", "residence_type": "rural",
    },
    "pregnant_puducherry": {  # pregnant woman, Puducherry
        "age": 24, "gender": "female", "annual_income": 20000, "state": "Puducherry",
        "pregnancy_status": "yes", "residence_type": "urban",
    },
    "phd_woman": {            # woman researcher, postgraduate
        "age": 29, "gender": "female", "occupation": "unemployed", "education_level": "postgraduate",
        "annual_income": 150000, "state": "Delhi", "social_category": "General",
    },
    "fisherman_tamilnadu": {  # fisherman, Tamil Nadu
        "age": 45, "gender": "male", "occupation": "fisherman", "annual_income": 120000,
        "state": "Tamil Nadu", "residence_type": "rural", "bpl_status": "yes",
    },
    "sparse_age_state": {     # only two facts: shows 'possible' schemes + missing attributes
        "age": 70, "state": "Uttar Pradesh",
    },
}
