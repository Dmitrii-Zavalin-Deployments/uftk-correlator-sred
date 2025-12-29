import pandas as pd

def compute_correlations(df):
    return df.corr(numeric_only=True)

def group_by_texture(df):
    return df.groupby("texture_class").mean(numeric_only=True)

def compute_statistics(df):
    return df.describe()

def compute_data_density(df):
    counts = df["texture_class"].value_counts()
    density = {}
    for cls, n in counts.items():
        if n < 5:
            density[cls] = "Low Confidence (n={})".format(n)
        elif n < 12:
            density[cls] = "Medium Confidence (n={})".format(n)
        else:
            density[cls] = "High Confidence (n={})".format(n)
    return density

def generate_markdown_report(corr, grouped, stats, density):
    with open("correlations.md", "w") as f:
        f.write("# Correlation Report\n\n")
        f.write("## Correlation Matrix\n")
        f.write(corr.to_markdown() + "\n\n")
        f.write("## Grouped by Texture\n")
        f.write(grouped.to_markdown() + "\n\n")
        f.write("## Data Density\n")
        for cls, msg in density.items():
            f.write(f"- {cls}: {msg}\n")

def main():
    df = pd.read_csv("field_data.csv")
    corr = compute_correlations(df)
    grouped = group_by_texture(df)
    stats = compute_statistics(df)
    density = compute_data_density(df)
    generate_markdown_report(corr, grouped, stats, density)

if __name__ == "__main__":
    main()



