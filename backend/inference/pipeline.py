# backend/inference/pipeline.py

import sys
import os

# Ensure project root is in Python path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from backend.models.soil_model import analyze_soil
from backend.models.microbial_features import extract_microbial_features
from backend.inference.Risk_engine import assess_risk


def format_scientific_output(result: dict) -> str:
    report = result["risk_assessment"]

    formatted = f"""
==========================================
        Soil Microbial Risk Report
==========================================

Crop: {report['Crop']}

Risk Score: {report['Risk Score']}
Severity: {report['Severity Classification']}

------------------------------------------
Soil Stress Level: {report['Soil Stress Level']}
Soil Acidity: {report['Soil Acidity']}

Microbial Colony Density: {report['Microbial Colony Density']}
Coverage Ratio: {report['Microbial Coverage Ratio']}
Clustering Index: {report['Microbial Clustering Index']}
Dominance Type: {report['Microbial Dominance Type']}

------------------------------------------
Technical Explanation:
{report['Full Technical Explanation']}

==========================================
"""
    return formatted


def format_farmer_output(result: dict) -> str:
    report = result["risk_assessment"]

    formatted = f"""
==========================================
        Soil Health Advisory Report
==========================================

Crop: {report['Crop']}

Overall Risk Level: {report['Overall Risk Level']}
Risk Score: {report['Risk Score']}

------------------------------------------
Soil Condition: {report['Soil Condition']}

Explanation:
{report['Detailed Explanation']}

Recommended Actions:
"""
    for action in report["Recommended Actions"]:
        formatted += f"- {action}\n"

    formatted += "\n==========================================\n"

    return formatted




def run_pipeline(crop: str, treatment: str, image_path: str, mode: str):

    # 1️⃣ Soil Analysis
    soil_result = analyze_soil(crop, treatment)

    # 2️⃣ Microbial Feature Extraction
    microbial_result = extract_microbial_features(image_path)

    if "error" in microbial_result:
        return {"error": "Image processing failed."}

    # 3️⃣ Ecological Risk Assessment
    risk_result = assess_risk(
        crop,
        soil_result,
        microbial_result,
        mode=mode
    )

    final_output = {
        "crop": crop,
        "soil_analysis": soil_result,
        "microbial_analysis": microbial_result,
        "risk_assessment": risk_result
    }

    return final_output


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run Soil-Borne Disease Prediction Pipeline"
    )

    parser.add_argument("--crop", type=str,
                        help="Crop name (e.g., Tomato, Brinjal, Cabbage)")
    parser.add_argument("--treatment", type=str,
                        help="Treatment type (e.g., D or T)")
    parser.add_argument("--image", type=str,
                        help="Path to image file")
    parser.add_argument("--mode", type=str, default="scientific",
                        help="Mode: scientific or farmer")

    args = parser.parse_args()

    # 🔹 CLI mode (all arguments provided)
    if args.crop and args.treatment and args.image:
        crop = args.crop
        treatment = args.treatment
        image_path = args.image
        mode = args.mode
    else:
        # 🔹 Interactive mode
        print("\nEntering Interactive Mode...\n")
        crop = input("Enter Crop (Tomato/Brinjal/Cabbage): ").strip()
        treatment = input("Enter Treatment (D/T): ").strip()
        image_path = input("Enter Image Path: ").strip()
        mode = input("Select Mode (scientific/farmer): ").strip()

    result = run_pipeline(crop, treatment, image_path, mode)

    # 🔹 RAW OUTPUT
    print("\n🔹 RAW OUTPUT:\n")
    print(result)

    # 🔹 FORMATTED REPORT
    if mode == "farmer":
        print("\n🔹 FARMER MODE REPORT:\n")
        print(format_farmer_output(result))
    else:
        print("\n🔹 SCIENTIFIC MODE REPORT:\n")
        print(format_scientific_output(result))