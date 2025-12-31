import os
import pandas as pd
import shutil

# ---------------------------------------------------------
# Unified working directory (GitHub-safe)
# ---------------------------------------------------------
WORKING_DIR = os.environ.get("WORKING_DIR", "data/testing-input-output")
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")
REPORT_PATH = os.path.join(WORKING_DIR, "correlations.md")
os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------
# Selected feature columns from the updated Analyzer (absolute + relative)
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "Brightness",
    "Mean_R", "Mean_G", "Mean_B",
    "Normalized_Blue",
    "Color_Temp_Proxy",
    "Texture",
    "Relative_Texture_Variance",
    "Edge_Density",
    "Shadow_Intensity",
    "Shadow_Direction_Variance",
    "Relative_Brightness_Variance"
]

# ---------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------
def compute_correlations(df):
    # Select only the defined feature columns that exist and are numeric
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if len(available_features) < 2:
        return pd.DataFrame()  # Not enough features
    
    df_features = df[available_features].apply(pd.to_numeric, errors="coerce")
    df_features = df_features.dropna()  # Remove rows with any NaN in features
    
    if len(df_features) < 2:
        return pd.DataFrame()  # Not enough valid rows for correlation
    
    corr = df_features.corr(method="pearson")
    corr = corr.round(3)
    return corr

def group_by_texture(df):
    if "Texture_Class" not in df.columns or df["Texture_Class"].nunique() == 0:
        return pd.DataFrame()
    return df.groupby("Texture_Class")[FEATURE_COLUMNS].mean(numeric_only=True).round(3)

def compute_statistics(df):
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        return pd.DataFrame()
    return df[available_features].describe().round(3)

def compute_data_density(df):
    if "Texture_Class" not in df.columns:
        return {}
    counts = df["Texture_Class"].value_counts()
    density = {}
    for cls, n in counts.items():
        if n < 5:
            density[cls] = f"Low Confidence (n={n})"
        elif n < 12:
            density[cls] = f"Medium Confidence (n={n})"
        else:
            density[cls] = f"High Confidence (n={n})"
    return density

def generate_narrative_insights(corr, density):
    insights = []
    if corr.empty:
        insights.append("Insufficient data for correlations (n < 2 valid rows).")
    else:
        # Find strong correlations (|r| > 0.5)
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 < col2:  # Avoid duplicates
                    r = corr.loc[col1, col2]
                    if abs(r) > 0.7:
                        strength = "very strong"
                    elif abs(r) > 0.5:
                        strength = "strong"
                    else:
                        continue
                    direction = "positive" if r > 0 else "negative"
                    insights.append(f"{col1} has {strength} {direction} correlation with {col2} (r = {r:.3f}).")
    # Add density note
    if density:
        insights.append("Data density per texture class:")
        for msg in density.values():
            insights.append(f"  - {msg}")
    else:
        insights.append("No texture classification data available.")
    return insights

# ---------------------------------------------------------
# Safe markdown fallback
# ---------------------------------------------------------
def df_to_markdown(df):
    if df.empty:
        return "_No data available_"
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(x) for x in row.values) + " |")
    return "\n".join([header, separator] + rows)

# ---------------------------------------------------------
# Markdown report
# ---------------------------------------------------------
def generate_markdown_report(corr, grouped, stats, density):
    insights = generate_narrative_insights(corr, density)
    
    with open(REPORT_PATH, "w") as f:
        f.write("# Correlation Report\n\n")
        
        f.write("## Human-Readable Insights\n")
        for insight in insights:
            f.write(f"- {insight}\n")
        f.write("\n")
        
        f.write("## Correlation Matrix (Selected Features)\n")
        if not corr.empty:
            f.write(df_to_markdown(corr) + "\n\n")
        else:
            f.write("_Insufficient valid data for correlation matrix_\n\n")
        
        f.write("## Grouped Averages by Texture Class\n")
        if not grouped.empty:
            f.write(df_to_markdown(grouped) + "\n\n")
        else:
            f.write("_No texture groups available_\n\n")
        
        f.write("## Summary Statistics\n")
        if not stats.empty:
            f.write(df_to_markdown(stats) + "\n\n")
        else:
            f.write("_No statistics available_\n\n")
        
        f.write("## Data Density & Confidence\n")
        if density:
            for cls, msg in density.items():
                f.write(f"- {cls}: {msg}\n")
        else:
            f.write("_No texture classification data_\n")

# ---------------------------------------------------------
# Image copying (visual traceability)
# ---------------------------------------------------------
def generate_correlated_images():
    for filename in os.listdir(WORKING_DIR):
        if filename.endswith("_analyzed.jpg"):
            name, ext = os.path.splitext(filename)
            base = name.replace("_analyzed", "")
            src = os.path.join(WORKING_DIR, filename)
            dst = os.path.join(WORKING_DIR, f"{base}_correlated{ext}")
            shutil.copy2(src, dst)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    if not os.path.isfile(CSV_PATH):
        print(f"❌ ERROR: field_data.csv not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from CSV")
    
    corr = compute_correlations(df)
    grouped = group_by_texture(df)
    stats = compute_statistics(df)
    density = compute_data_density(df)
    
    generate_markdown_report(corr, grouped, stats, density)
    print("✓ correlations.md generated")
    
    generate_correlated_images()
    print("✓ Correlated images created")
    
    print(f"🎉 Correlator complete — outputs in {WORKING_DIR}")

if __name__ == "__main__":
    main()


