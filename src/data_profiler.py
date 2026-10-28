import glob
import os
import csv
import json
import pandas as pd


def profile_dataset(sample_dir="use_cases/pnm_study_case/sample"):
    if not os.path.exists(sample_dir):
        parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", sample_dir)

        fallback_dir = os.path.join(os.path.dirname(__file__), "..", sample_dir)

        if os.path.exists(parent_dir):
            sample_dir = parent_dir
        elif os.path.exists(fallback_dir):
            sample_dir = fallback_dir


    csv_files = glob.glob(os.path.join(sample_dir, "*.csv"))

    print(f"Found {len(csv_files)} CSV files in {sample_dir}")

    dfs = []
    file_stats = []


    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines="skip",
                low_memory=False,
                dtype=str
            )


            df["source_file"] = filename

            dfs.append(df)


            file_stats.append({
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns)
            })
        except Exception as e:
            print(f"Error loading {filename}: {e}")


    if not dfs:
        print("No data found!")
        return {}


    combined_df = pd.concat(dfs, ignore_index=True)


    if "language" in combined_df.columns:
        clean_lang = combined_df["language"].astype(str).str.strip().str.lstrip("'")
    else:
        clean_lang = pd.Series(dtype=str)


    total_rows = len(combined_df)


    unique_ids = combined_df["original_id"].nunique() if "original_id" in combined_df.columns else 0


    duplicate_rows = total_rows - unique_ids


    missing_title = combined_df["title"].isna().sum() if "title" in combined_df.columns else 0

    empty_title = (combined_df["title"].str.strip() == "").sum() if "title" in combined_df.columns else 0


    missing_body = combined_df["body"].isna().sum() if "body" in combined_df.columns else 0

    empty_body = (combined_df["body"].str.strip() == "").sum() if "body" in combined_df.columns else 0


    lang_dist = clean_lang.value_counts().to_dict()


    stats = {
        "total_files": len(csv_files),
        "total_rows_loaded": int(total_rows),
        "unique_articles": int(unique_ids),
        "duplicate_rows": int(duplicate_rows),
        "missing_titles": int(missing_title + empty_title),
        "missing_bodies": int(missing_body + empty_body),
        "language_distribution": lang_dist,
        "files": file_stats
    }

    return stats


if __name__ == "__main__":
    results = profile_dataset()

    print(json.dumps(results, indent=2))
