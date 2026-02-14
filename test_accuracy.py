"""Quick test to check analysis accuracy on known false claims."""
import requests
import json

URL = "http://127.0.0.1:8000/api/analyze/"
HEADERS = {"Content-Type": "application/json", "X-CSRFToken": "t"}
COOKIES = {"csrftoken": "t"}

tests = [
    {
        "title": "Putin is the next Prime Minister of India",
        "content": "Vladimir Putin is going to become the next Prime Minister of India",
        "expected": "HIGH (>70%)",
    },
    {
        "title": "Sun rises in the east",
        "content": "The sun rises in the east every morning as usual",
        "expected": "LOW (<30%)",
    },
]

for t in tests:
    print(f"\n{'='*60}")
    print(f"CLAIM: {t['title']}")
    print(f"EXPECTED: {t['expected']}")
    try:
        r = requests.post(URL, json={
            "title": t["title"],
            "text": t["content"],
            "url": "",
            "source_name": "Test",
        }, headers=HEADERS, cookies=COOKIES, timeout=60)
        d = r.json()
        res = d.get("results", d)
        ml = res.get("misinformation_likelihood", "?")
        cs = res.get("credibility_score", "?")
        rl = res.get("risk_level", "?")
        print(f"RESULT: Misinfo={ml}  Credibility={cs}  Risk={rl}")
        for ind in res.get("key_indicators", []):
            desc = ind if isinstance(ind, str) else ind.get('description','')
            print(f"  - {desc[:90]}")
    except Exception as e:
        print(f"ERROR: {e}")
