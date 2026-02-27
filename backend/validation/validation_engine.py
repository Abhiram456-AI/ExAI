import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.models.soil_model import analyze_soil
from backend.models.microbial_features import extract_microbial_features
from backend.inference.Risk_engine import assess_risk


DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "validation_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "validation_summary.csv"


def _detect_crop_and_treatment(folder_name: str):
    """
    Example folder names:
        Tomato-LB
        Brinjal-NA
        Cabbage-LB
    """
    parts = folder_name.split("-")
    crop = parts[0].strip()
    medium = parts[1].strip() if len(parts) > 1 else "NA"

    # Map medium to treatment logic (you may refine later)
    # For now:
    # LB → Dry
    # NA → Treated
    treatment = "D" if medium.upper() == "LB" else "T"

    return crop, treatment


def _is_image_file(filename: str):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def run_full_validation():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found at: {DATASET_DIR}")

    print(f"\nScanning dataset at: {DATASET_DIR}\n")

    all_records = []

    for crop_folder in os.listdir(DATASET_DIR):

        crop_path = DATASET_DIR / crop_folder
        if not crop_path.is_dir():
            continue

        crop, treatment = _detect_crop_and_treatment(crop_folder)

        for day_folder in os.listdir(crop_path):

            day_path = crop_path / day_folder
            if not day_path.is_dir():
                continue

            for file in os.listdir(day_path):

                if not _is_image_file(file):
                    continue

                image_path = day_path / file

                try:
                    soil_data = analyze_soil(crop, treatment)
                    microbial_data = extract_microbial_features(str(image_path))
                    # Use positional arguments to match assess_risk signature
                    risk_data = assess_risk(
                        crop,
                        soil_data,
                        microbial_data,
                        mode="scientific"
                    )

                    record = {
                        "image_path": str(image_path),
                        "crop": crop,
                        "treatment": treatment,

                        "colony_density": microbial_data.get("colony_density"),
                        "coverage_ratio": microbial_data.get("coverage_ratio"),
                        "clustering_index": microbial_data.get("clustering_index"),
                        "mean_circularity": microbial_data.get("mean_circularity"),
                        "mean_solidity": microbial_data.get("mean_solidity"),
                        "area_variation": microbial_data.get("area_variation"),
                        "dominance": microbial_data.get("dominance"),

                        "risk_score": risk_data.get("Risk Score"),
                        "severity": risk_data.get("Severity Classification")
                    }

                    all_records.append(record)

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    df = pd.DataFrame(all_records)

    # Normalize severity values to avoid case/whitespace mismatch
    if "severity" in df.columns:
        df["severity"] = (
            df["severity"]
            .astype(str)
            .str.strip()
            .str.title()
        )

    df.to_csv(RESULT_CSV, index=False)

    print(f"\nSaved detailed results to: {RESULT_CSV}")

    generate_summary_statistics(df)


def generate_summary_statistics(df: pd.DataFrame):

    if df.empty:
        print("No data available for summary.")
        return

    summary_records = []

    crops = df["crop"].unique()

    for crop in crops:

        crop_df = df[df["crop"] == crop]

        summary = {
            "crop": crop,
            "num_samples": len(crop_df),

            "mean_risk_score": crop_df["risk_score"].mean(),
            "std_risk_score": crop_df["risk_score"].std(),

            "mean_density": crop_df["colony_density"].mean(),
            "mean_coverage": crop_df["coverage_ratio"].mean(),
            "mean_clustering": crop_df["clustering_index"].mean(),

            "percent_high_risk":
                (crop_df["severity"] == "High").mean() * 100,

            "percent_moderate_risk":
                (crop_df["severity"] == "Moderate").mean() * 100,

            "percent_low_risk":
                (crop_df["severity"] == "Low").mean() * 100
        }

        summary_records.append(summary)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved summary statistics to: {SUMMARY_CSV}")

    print("\n--- GLOBAL STATISTICS ---")
    print(f"Total Images Processed: {len(df)}")
    print(f"Overall Mean Risk Score: {df['risk_score'].mean():.3f}")
    print(f"Overall Risk Std Dev: {df['risk_score'].std():.3f}")

    print("\nDominance Distribution:")
    print(df["dominance"].value_counts())


if __name__ == "__main__":
    run_full_validation()
