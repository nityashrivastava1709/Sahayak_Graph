import json
from pathlib import Path

contract_path = Path(__file__).parent / "scheme_response.json"

with contract_path.open(encoding="utf-8") as file:
    data = json.load(file)

# Check the top-level structure
assert set(data.keys()) == {
    "profile",
    "matched_schemes",
    "missing_attributes"
}

# Check the profile
profile = data["profile"]
assert isinstance(profile, dict)
assert isinstance(profile["age"], int)
assert isinstance(profile["income"], (int, float))
assert isinstance(profile["state"], str)
assert isinstance(profile["occupation"], str)
assert isinstance(profile["gender"], str)

# Check the scheme list
assert isinstance(data["matched_schemes"], list)

for scheme in data["matched_schemes"]:
    assert isinstance(scheme["scheme_id"], str)
    assert isinstance(scheme["name"], str)
    assert isinstance(scheme["match_score"], (int, float))
    assert isinstance(scheme["kg_path"], list)
    assert isinstance(scheme["supporting_text"], list)
    assert isinstance(scheme["source_chunk_ids"], list)
    assert isinstance(scheme["required_documents"], list)
    assert isinstance(scheme["department"], str)
    assert isinstance(scheme["conflicts_with"], list)

# Check missing attributes
assert isinstance(data["missing_attributes"], list)
assert all(isinstance(item, str) for item in data["missing_attributes"])

print("Contract validation successful!")