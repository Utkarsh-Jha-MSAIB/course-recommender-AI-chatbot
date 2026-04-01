import os
import re
import ast
from pathlib import Path
from typing import Any, List
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# INPUT_FILE = os.getenv("INPUT_FILE", "./data/GenAI_crawl5.xlsx")
INPUT_FILE= "./data/GenAI_crawl2.xlsx"
SHEET_NAME = "GenAI_crawl"
OUTPUT_DIR = Path("skill_artifacts_filtered")
OUTPUT_DIR.mkdir(exist_ok=True)

SKILL_COLUMNS = ["Skills Learned"]

def normalize_whitespace(text: Any) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def parse_skill_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        s = str(value).strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                raw_items = parsed if isinstance(parsed, list) else [s]
            except Exception:
                raw_items = re.split(r"[;,|]", s)
        else:
            raw_items = re.split(r"[;,|]", s)

    return [normalize_whitespace(x) for x in raw_items if normalize_whitespace(x)]

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

all_skills = []
for col in SKILL_COLUMNS:
    if col in df.columns:
        for value in df[col]:
            all_skills.extend(parse_skill_list(value))

skill_counts = Counter(all_skills)

skills_df = pd.DataFrame(
    [{"skill": skill, "count": count} for skill, count in skill_counts.items()]
).sort_values(by=["count", "skill"], ascending=[False, True])

skills_df.to_csv(OUTPUT_DIR / "unique_skills.csv", index=False)

print("INPUT_FILE from env/code:", INPUT_FILE)
print("Resolved path exists:", Path(INPUT_FILE).exists())
print("Current working directory:", Path.cwd())

print(f"Saved {len(skills_df)} unique skills to {OUTPUT_DIR / 'unique_skills.csv'}")