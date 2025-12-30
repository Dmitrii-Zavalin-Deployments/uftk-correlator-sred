import os
import pandas as pd
import shutil

# ---------------------------------------------------------
# Unified working directory (GitHub‑safe)
# ---------------------------------------------------------

WORKING_DIR = os.environ.get("WORKING_DIR", "data/testing-input-output")
CSV_PATH = os.path.join(WORKING_DIR, "field_data.csv")
REPORT_PATH = os.path.join(WORKING_DIR, "correlations.md")

# Ensure directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------

def compute_correlations(df):
    return df.corr(numeric_only=True)

def group_by_texture(df):
    if "Texture_Class" in df.columns:
        return df.groupby("Texture_Class").mean(numeric_only=True)
    return pd.DataFrame()

def compute_statistics(df):
    return df.describe()

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

# ---------------------------------------------------------
# Markdown report
# ---------------------------------------------------------

def generate_markdown_report(corr, grouped, stats, density):
    with open(REPORT_PATH, "w") as f:
        f.write("# Correlation Report\n\n")

        f.write("## Correlation Matrix\n")
        f.write(corr.to_markdown() + "\n\n")

        f.write("## Grouped by Texture\n")
        if not grouped.empty:
            f.write(grouped.to_markdown() + "\n\n")
        else:
            f.write("_No texture groups available_\n\n")

        f.write("## Data Density\n")
        for cls, msg in density.items():
            f.write(f"- {cls}: {msg}\n")

# ---------------------------------------------------------
# Copy analyzed → correlated
# ---------------------------------------------------------

def generate_correlated_images():
    for filename in os.listdir(WORKING_DIR):
        if filename.endswith("_analyzed.jpg"):
            src = os.path.join(WORKING_DIR, filename)
            base = filename.replace("_analyzed", "")
            dst = os.path.join(WORKING_DIR, f"{base}_correlated.jpg")
            shutil.copy2(src, dst)
            print(f"✓ Created {dst}")

# ---------------------------------------------------------
# Update CSV with correlation summary
# ---------------------------------------------------------

def update_csv_with_summary(df, corr):
    df["Global_Correlation_Count"] = len(corr.columns)
    df.to_csv(CSV_PATH, index=False)
    print("✓ Updated field_data.csv with correlation summary")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"❌ ERROR: field_data.csv not found in {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    corr = compute_correlations(df)
    grouped = group_by_texture(df)
    stats = compute_statistics(df)
    density = compute_data_density(df)

    generate_markdown_report(corr, grouped, stats, density)
    print("✓ correlations.md generated")

    generate_correlated_images()

    update_csv_with_summary(df, corr)

    print(f"🎉 Correlator complete. Outputs written to {WORKING_DIR}")

if __name__ == "__main__":
    main()



