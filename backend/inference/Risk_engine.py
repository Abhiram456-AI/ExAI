"""
Risk Engine Module (Deterministic, Dual-Mode)

This module performs strict microbial risk assessment using:

- Soil analysis output
- Microbial feature extraction output
- Crop-specific weight calibration

It does NOT predict exact pathogens.
It performs risk categorization only.
No hallucination. No probabilistic guessing.
Fully rule-based and auditable.
"""

from typing import Dict


# ----------------------------------------
# Crop-Specific Weight Calibration
# ----------------------------------------

CROP_WEIGHTS = {
    "Tomato": {
        "soil": 0.3,
        "microbial_density": 0.25,
        "coverage": 0.2,
        "clustering": 0.15,
        "dominance": 0.1
    },
    "Brinjal": {
        "soil": 0.35,
        "microbial_density": 0.2,
        "coverage": 0.2,
        "clustering": 0.15,
        "dominance": 0.1
    },
    "Cabbage": {
        "soil": 0.25,
        "microbial_density": 0.3,
        "coverage": 0.2,
        "clustering": 0.15,
        "dominance": 0.1
    }
}


# ----------------------------------------
# Helper Normalization Functions
# ----------------------------------------

def _soil_index(soil: Dict) -> float:
    mapping = {
        "Low": 0.2,
        "Moderate": 0.5,
        "High": 1.0
    }
    return mapping.get(soil.get("stress_level"), 0.5)


def _dominance_index(microbial: Dict) -> float:
    mapping = {
        "Bacterial": 1.0,
        "Fungal": 0.75,
        "Mixed": 0.55,
        "Low": 0.2
    }
    return mapping.get(microbial.get("dominance"), 0.5)


def _microbial_load(microbial: Dict) -> float:
    density = microbial.get("colony_density", 0)
    return min(float(density), 1.0)


def _coverage_index(microbial: Dict) -> float:
    return min(float(microbial.get("coverage_ratio", 0)), 1.0)


def _clustering_index(microbial: Dict) -> float:
    return min(float(microbial.get("clustering_index", 0)), 1.0)


# ----------------------------------------
# Core Risk Calculation
# ----------------------------------------

def _calculate_risk_score(
    crop: str,
    soil: Dict,
    microbial: Dict
) -> float:

    if crop not in CROP_WEIGHTS:
        raise ValueError(f"No crop weights defined for: {crop}")

    weights = CROP_WEIGHTS[crop]

    soil_val = _soil_index(soil)
    density_val = _microbial_load(microbial)
    coverage_val = _coverage_index(microbial)
    clustering_val = _clustering_index(microbial)
    dominance_val = _dominance_index(microbial)

    risk_score = (
        weights["soil"] * soil_val +
        weights["microbial_density"] * density_val +
        weights["coverage"] * coverage_val +
        weights["clustering"] * clustering_val +
        weights["dominance"] * dominance_val
    )

    return round(risk_score, 3)


def _classify_severity(score: float) -> str:
    # Recalibrated to improve harmful-case recall while preserving precision
    # Low < 0.35
    # Moderate < 0.65
    # High ≥ 0.65
    if score < 0.35:
        return "Low"
    elif score < 0.65:
        return "Moderate"
    else:
        return "High"


# ----------------------------------------
# Risk Category Mapping (Non-Specific)
# ----------------------------------------

def _associated_risk_categories(
    soil: Dict,
    microbial: Dict
) -> list:

    categories = []

    dominance = microbial.get("dominance")
    coverage = microbial.get("coverage_ratio", 0)
    clustering = microbial.get("clustering_index", 0)

    if dominance == "Bacterial" and coverage > 0.25:
        categories.append("High bacterial proliferation risk")

    if dominance == "Fungal" and clustering > 0.18:
        categories.append("Root rot or fungal soil disease risk")

    if soil.get("acidity") == "Acidic":
        categories.append("Soil stress-related vulnerability")

    if not categories:
        categories.append("No significant disease-favoring conditions detected")

    return categories


# ----------------------------------------
# Farmer Mode Report
# ----------------------------------------

def _farmer_report(
    crop: str,
    risk_score: float,
    severity: str,
    categories: list
) -> Dict:

    if severity == "High":
        soil_status = "Weak and stressed"
        urgency = "Immediate attention is required."
        recommendations = [
            "Improve soil drainage immediately.",
            "Avoid overwatering and prevent water stagnation.",
            "Apply recommended bio-control treatment from an agricultural expert.",
            "Inspect plant roots for early signs of damage.",
            "Remove severely affected plants to prevent spread."
        ]
    elif severity == "Moderate":
        soil_status = "Slightly stressed"
        urgency = "Careful monitoring is recommended."
        recommendations = [
            "Monitor crop regularly for visible symptoms.",
            "Avoid excess moisture in the soil.",
            "Add organic compost to improve soil strength.",
            "Ensure proper spacing between plants."
        ]
    else:
        soil_status = "Healthy and stable"
        urgency = "No immediate risk detected."
        recommendations = [
            "Maintain current irrigation practices.",
            "Continue balanced fertilization.",
            "Regularly monitor crop health as a precaution."
        ]

    explanation = (
        f"The system analyzed soil condition and microbial growth patterns "
        f"for {crop}. Based on the observed microbial density and soil stress, "
        f"the overall risk level is classified as '{severity}'. "
        f"This means that the current soil environment shows conditions that "
        f"may {'favor harmful microbial growth' if severity != 'Low' else 'support balanced microbial activity'}. "
        f"The identified risk categories include: {', '.join(categories)}. "
        f"{urgency}"
    )

    return {
        "Crop": crop,
        "Soil Condition": soil_status,
        "Overall Risk Level": severity,
        "Risk Score": risk_score,
        "Detailed Explanation": explanation,
        "Identified Risk Categories": categories,
        "Recommended Actions": recommendations
    }


# ----------------------------------------
# Scientific Mode Report
# ----------------------------------------

def _scientific_report(
    crop: str,
    soil: Dict,
    microbial: Dict,
    risk_score: float,
    severity: str,
    categories: list
) -> Dict:

    soil_stress = soil.get("stress_level")
    acidity = soil.get("acidity")
    colony_density = microbial.get("colony_density")
    coverage_ratio = microbial.get("coverage_ratio")
    clustering_index = microbial.get("clustering_index")
    dominance = microbial.get("dominance")

    technical_explanation = (
        f"A deterministic ecological risk assessment was performed for {crop} "
        f"using crop-specific calibrated weights. The soil stress index was derived "
        f"from the qualitative classification '{soil_stress}'. "
        f"Microbial ecological indices included normalized colony density "
        f"({colony_density}), coverage ratio ({coverage_ratio}), and clustering "
        f"index ({clustering_index}). Dominance classification ('{dominance}') "
        f"was mapped to a predefined dominance weight. "
        f"The final risk score of {risk_score} was calculated through weighted "
        f"summation of soil stress, microbial density, spatial coverage, "
        f"clustering behavior, and dominance indices. "
        f"Severity classification was determined using recalibrated threshold boundaries "
        f"(Low < 0.35, Moderate < 0.65, High ≥ 0.65) following ecological density normalization. "
        f"No probabilistic modeling or pathogen-level inference was performed. "
        f"Associated risk categories were derived strictly from deterministic "
        f"conditional logic."
    )

    return {
        "Crop": crop,
        "Risk Score": risk_score,
        "Severity Classification": severity,
        "Soil Stress Level": soil_stress,
        "Soil Acidity": acidity,
        "Microbial Colony Density": colony_density,
        "Microbial Coverage Ratio": coverage_ratio,
        "Microbial Clustering Index": clustering_index,
        "Microbial Dominance Type": dominance,
        "Associated Risk Categories": categories,
        "Full Technical Explanation": technical_explanation
    }


# ----------------------------------------
# Main Public Function
# ----------------------------------------

def assess_risk(
    crop: str,
    soil_result: Dict,
    microbial_result: Dict,
    mode: str = "farmer"
) -> Dict:

    risk_score = _calculate_risk_score(
        crop,
        soil_result,
        microbial_result
    )

    severity = _classify_severity(risk_score)

    categories = _associated_risk_categories(
        soil_result,
        microbial_result
    )

    if mode.lower() == "scientific":
        return _scientific_report(
            crop,
            soil_result,
            microbial_result,
            risk_score,
            severity,
            categories
        )

    return _farmer_report(
        crop,
        risk_score,
        severity,
        categories
    )