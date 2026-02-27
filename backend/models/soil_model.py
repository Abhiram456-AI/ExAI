import pandas as pd

# Load dataset once (global, efficient)
DATA_PATH = "backend/data/agricultural_dataset.xlsx"
# Read Excel with second row as header (actual crop names)
raw_df = pd.read_excel(DATA_PATH, header=1)

# Rename first two columns properly
raw_df = raw_df.rename(columns={
    raw_df.columns[0]: "SlNo",
    raw_df.columns[1]: "Parameter"
})

# Set soil parameters as index
df = raw_df.set_index("Parameter")


def _resolve_column_name(crop: str, treatment: str) -> str:
    """
    Resolve correct column name based on crop and treatment.

    Rules:
    - Crop without suffix -> Dry (D)
    - Crop + 'T' suffix   -> Treated (T)
    - Matching is case-insensitive and space-tolerant
    """

    crop_clean = crop.strip().lower()
    treatment = treatment.upper()

    # Build normalized lookup: lowercase, stripped column names
    normalized_cols = {col.strip().lower(): col for col in df.columns}

    dry_key = crop_clean
    treated_key = f"{crop_clean} t"

    if treatment == "T" and treated_key in normalized_cols:
        return normalized_cols[treated_key]

    if treatment == "D" and dry_key in normalized_cols:
        return normalized_cols[dry_key]

    # Fallbacks
    if treated_key in normalized_cols:
        return normalized_cols[treated_key]

    if dry_key in normalized_cols:
        return normalized_cols[dry_key]

    raise ValueError(
        f"No valid column found for crop '{crop}'. "
        f"Available columns: {list(df.columns)}"
    )


def analyze_soil(crop: str, treatment: str = "D") -> dict:
    """
    Analyze soil condition for a given crop and treatment.
    Returns an explainable soil profile.
    """

    column = _resolve_column_name(crop, treatment)

    # Extract core parameters safely
    def get_value(param):
        try:
            return float(df.loc[param, column])
        except KeyError:
            return None

    pH = get_value("pH (1:2.5)")
    EC = get_value("EC (dS/m)")
    OC = get_value("Organic Carbon (%)")
    N = get_value("Available Nitrogen (Kg ha-1)")

    # ---- Interpretations ---- #

    # Acidity
    if pH is not None:
        if pH < 6.5:
            acidity = "Acidic"
        elif 6.5 <= pH <= 7.5:
            acidity = "Neutral"
        else:
            acidity = "Alkaline"
    else:
        acidity = "Unknown"

    # Salinity stress
    if EC is not None:
        salinity_stress = "High" if EC > 0.8 else "Low"
    else:
        salinity_stress = "Unknown"

    # Organic carbon status
    if OC is not None:
        nutrient_status = "Low Organic Carbon" if OC < 0.5 else "Adequate Organic Carbon"
    else:
        nutrient_status = "Unknown"

    # Nitrogen stress
    if N is not None:
        nitrogen_status = "Low Nitrogen" if N < 140 else "Adequate Nitrogen"
    else:
        nitrogen_status = "Unknown"

    # Overall stress level
    stress_factors = [
        acidity == "Acidic",
        salinity_stress == "High",
        nutrient_status == "Low Organic Carbon",
        nitrogen_status == "Low Nitrogen",
        treatment == "D"
    ]

    stress_level = "High" if stress_factors.count(True) >= 3 else "Moderate"

    # Microbial favorability
    if acidity == "Acidic" and treatment == "D":
        favors = "Bacteria"
    elif acidity == "Neutral" and treatment == "T":
        favors = "Balanced Microflora"
    else:
        favors = "Mixed"

    return {
        "crop": crop,
        "treatment": "Dry" if treatment == "D" else "Treated",
        "column_used": column,

        "soil_pH": pH,
        "acidity": acidity,
        "EC": EC,
        "salinity_stress": salinity_stress,

        "organic_carbon": OC,
        "nutrient_status": nutrient_status,

        "nitrogen": N,
        "nitrogen_status": nitrogen_status,

        "stress_level": stress_level,
        "favors": favors
    }