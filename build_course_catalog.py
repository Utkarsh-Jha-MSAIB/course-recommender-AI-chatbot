import ast
import os
import pickle
import re
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

INPUT_FILE = "./data/courses_with_images.xlsx"
SHEET_NAME = os.getenv("SHEET_NAME", "Catalog_Python")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "course_artifacts"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using input file: {INPUT_FILE}")
print(f"Using sheet name: {SHEET_NAME}")
print(f"Using output dir: {OUTPUT_DIR}")

# =========================================================
# EXPECTED COLUMNS
# =========================================================
EXPECTED_COLUMNS = [
    "Course Name",
    "University / Industry Partner Name",
    "Type of Content",
    "Difficulty Level",
    "Avg Total Learning Hours",
    "Learning Hours-by instructor",
    "Course Rating",
    "Enrollment Count",
    "Course URL",
    "Course Description",
    "Skills Learned",
    "Scorable Skills",
    "Specialization",
    "Specialization URL",
    "Subtitle Language",
    "Course Language",
    "ML Translation",
    "Domain",
    "Sub-Domain",
    "Course ID",
    "Specialization Ids",
    "New Course",
]

OPTIONAL_COLUMNS = [
    "Course Image URL",
]

# =========================================================
# HELPERS
# =========================================================
def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_text_title(value: Any) -> str:
    text = clean_text(value)
    return text.title() if text else ""


def clean_text_upper(value: Any) -> str:
    text = clean_text(value)
    return text.upper() if text else "UNKNOWN"


def clean_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", s)
    return float(match.group()) if match else np.nan


def clean_enrollment_count(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower().replace(",", "")
    s = re.sub(r"\bstudents?\b|\blearners?\b|\benrollments?\b|\benrolled\b", "", s).strip()

    match = re.search(r"([-+]?\d*\.?\d+)\s*([km]?)", s)
    if not match:
        return np.nan

    number = float(match.group(1))
    suffix = match.group(2)

    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000

    return float(number)


def clean_bool_flag(value: Any) -> int:
    if pd.isna(value):
        return 0
    return 1 if clean_text(value).lower() in {"true", "yes", "1"} else 0


def parse_list_like_field(value: Any) -> List[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        items = value
    else:
        s = str(value).strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                items = parsed if isinstance(parsed, list) else [s]
            except Exception:
                items = re.split(r"[;,|]", s)
        else:
            items = re.split(r"[;,|]", s)

    cleaned = []
    for item in items:
        item = clean_text(item)
        if item:
            cleaned.append(item)

    return cleaned


def merge_skills(skills_1: Any, skills_2: Any) -> List[str]:
    combined = parse_list_like_field(skills_1) + parse_list_like_field(skills_2)
    combined = sorted(set(combined))
    return combined


def min_max_scale(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.zeros(len(series)), index=series.index)

    if min_val == max_val:
        return pd.Series(np.ones(len(series)), index=series.index)

    return (series - min_val) / (max_val - min_val)


def hierarchical_group_median_impute(
    df: pd.DataFrame,
    target_col: str,
    grouping_levels: List[List[str]],
    fallback_value: float = 0.0,
) -> pd.Series:
    result = df[target_col].copy()

    for group_cols in grouping_levels:
        group_medians = df.groupby(group_cols, dropna=False)[target_col].transform("median")
        result = result.fillna(group_medians)

    global_median = df[target_col].median()
    if pd.isna(global_median):
        global_median = fallback_value

    return result.fillna(global_median)


def build_course_text(row: pd.Series) -> str:
    parts = [
        f"course name: {row['Course Name']}",
        f"partner: {row['University / Industry Partner Name']}",
        f"type: {row['Type of Content']}",
        f"difficulty: {row['Difficulty Level_Clean']}",
        f"domain: {row['Domain_Clean']}",
        f"sub-domain: {row['Sub-Domain_Clean']}",
        f"course language: {row['Course Language_Clean']}",
        f"subtitle language: {row['Subtitle Language_Clean']}",
        f"skills: {row['Unified Skills Text']}",
        f"description: {row['Course Description_Clean']}",
    ]
    return " | ".join([p for p in parts if clean_text(p)])


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

df = df.copy()

# Ensure optional columns exist so downstream code stays stable
for col in OPTIONAL_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# =========================================================
# BASIC CLEANING
# =========================================================
text_columns = [
    "Course Name",
    "University / Industry Partner Name",
    "Type of Content",
    "Difficulty Level",
    "Course URL",
    "Course Image URL",
    "Course Description",
    "Skills Learned",
    "Scorable Skills",
    "Specialization",
    "Specialization URL",
    "Subtitle Language",
    "Course Language",
    "ML Translation",
    "Domain",
    "Sub-Domain",
    "Course ID",
    "Specialization Ids",
    "New Course",
]

for col in text_columns:
    df[col] = df[col].apply(clean_text)

if "Course Name" in df.columns:
    df = df.drop_duplicates(subset=["Course Name"], keep="first")
else:
    df = df.drop_duplicates(subset=["Course Name", "Course URL"], keep="first")

# =========================================================
# CLEAN DERIVED FIELDS
# =========================================================
df["Partner_Clean"] = df["University / Industry Partner Name"].apply(clean_text)
df["Difficulty Level_Clean"] = df["Difficulty Level"].apply(clean_text_upper)
df["Domain_Clean"] = df["Domain"].apply(clean_text_title)
df["Sub-Domain_Clean"] = df["Sub-Domain"].apply(clean_text_title)
df["Course Language_Clean"] = df["Course Language"].apply(clean_text)
df["Subtitle Language_Clean"] = df["Subtitle Language"].apply(clean_text)
df["Course Description_Clean"] = df["Course Description"].apply(clean_text)
df["Course Image URL_Clean"] = df["Course Image URL"].apply(clean_text)

df["Avg Total Learning Hours_Clean"] = df["Avg Total Learning Hours"].apply(clean_numeric)
df["Learning Hours_By_Instructor_Clean"] = df["Learning Hours-by instructor"].apply(clean_numeric)
df["Course Rating_Raw_Clean"] = df["Course Rating"].apply(clean_numeric)
df["Enrollment_Count_Raw_Clean"] = df["Enrollment Count"].apply(clean_enrollment_count)

# =========================================================
# HIERARCHICAL IMPUTATION
# =========================================================
grouping_hierarchy = [
    ["Partner_Clean", "Domain_Clean", "Sub-Domain_Clean"],
    ["Partner_Clean", "Domain_Clean"],
    ["Domain_Clean", "Sub-Domain_Clean"],
    ["Partner_Clean"],
    ["Domain_Clean"],
]

# Use only instructor hours for consistency
df["Hours_Final"] = df["Learning Hours_By_Instructor_Clean"]

df["Hours_Final"] = hierarchical_group_median_impute(
    df=df,
    target_col="Hours_Final",
    grouping_levels=grouping_hierarchy,
    fallback_value=0.0,
)

df["Has_Rating"] = df["Course Rating_Raw_Clean"].notna().astype(int)
df["Course Rating_Clean"] = hierarchical_group_median_impute(
    df=df,
    target_col="Course Rating_Raw_Clean",
    grouping_levels=grouping_hierarchy,
    fallback_value=0.0,
)

df["Has_Enrollment"] = df["Enrollment_Count_Raw_Clean"].notna().astype(int)
df["Enrollment_Count_Clean"] = hierarchical_group_median_impute(
    df=df,
    target_col="Enrollment_Count_Raw_Clean",
    grouping_levels=grouping_hierarchy,
    fallback_value=0.0,
)

df["Popularity_Score_0_1"] = min_max_scale(df["Enrollment_Count_Clean"]).round(4)
df["Popularity_Percentile"] = df["Enrollment_Count_Clean"].rank(method="average", pct=True).fillna(0).round(4)
df["Enrollment_Count_Log1p"] = np.log1p(df["Enrollment_Count_Clean"]).round(4)

# =========================================================
# OTHER FIELDS
# =========================================================
df["New Course Flag"] = df["New Course"].apply(clean_bool_flag)
df["Specialization Ids List"] = df["Specialization Ids"].apply(parse_list_like_field)

df["Unified Skills List"] = df.apply(
    lambda row: merge_skills(row["Skills Learned"], row["Scorable Skills"]),
    axis=1
)
df["Unified Skills Text"] = df["Unified Skills List"].apply(lambda x: ", ".join(x))

# =========================================================
# EMBEDDING TEXT
# =========================================================
df["course_text_for_embedding"] = df.apply(build_course_text, axis=1)

# =========================================================
# OPTIONAL CHECKS
# =========================================================
print(f"Total rows after deduplication: {len(df)}")
print(f"Hours_Final missing: {df['Hours_Final'].isna().sum()}")
print(f"Course Rating_Clean missing: {df['Course Rating_Clean'].isna().sum()}")
print(f"Enrollment_Count_Clean missing: {df['Enrollment_Count_Clean'].isna().sum()}")
print(f"Course Image URL_Clean missing/blank: {(df['Course Image URL_Clean'] == '').sum()}")

# =========================================================
# GENERATE EMBEDDINGS
# =========================================================
model = SentenceTransformer(EMBEDDING_MODEL)

course_texts = df["course_text_for_embedding"].tolist()
embeddings = model.encode(
    course_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Embeddings shape:", embeddings.shape)

# =========================================================
# BUILD FAISS INDEX
# =========================================================
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

print("FAISS index size:", index.ntotal)

# =========================================================
# SAVE OUTPUTS
# =========================================================
cleaned_csv_path = OUTPUT_DIR / "cleaned_courses.csv"
embeddings_path = OUTPUT_DIR / "course_embeddings.npy"
faiss_index_path = OUTPUT_DIR / "course_faiss.index"
metadata_path = OUTPUT_DIR / "course_metadata.pkl"

df.to_csv(cleaned_csv_path, index=False)
np.save(embeddings_path, embeddings)
faiss.write_index(index, str(faiss_index_path))

metadata_cols = [
    "Course Name",
    "University / Industry Partner Name",
    "Type of Content",
    "Difficulty Level_Clean",
    "Hours_Final",
    "Course Rating_Clean",
    "Has_Rating",
    "Enrollment_Count_Clean",
    "Has_Enrollment",
    "Enrollment_Count_Log1p",
    "Domain_Clean",
    "Sub-Domain_Clean",
    "Course Language_Clean",
    "Unified Skills Text",
    "Course Description_Clean",
    "course_text_for_embedding",
    "Course URL",
    "Course Image URL_Clean",
    "Course ID",
    "Specialization",
    "New Course Flag",
]

metadata = df[metadata_cols].rename(
    columns={"Course Image URL_Clean": "Course Image URL"}
).to_dict(orient="records")

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print("\nSaved artifacts:")
print(f"- Cleaned CSV: {cleaned_csv_path}")
print(f"- Embeddings: {embeddings_path}")
print(f"- FAISS index: {faiss_index_path}")
print(f"- Metadata: {metadata_path}")
