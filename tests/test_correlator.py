import os
import numpy as np
import pandas as pd
import pytest

import universal_field_toolkit_correlator_sred as correlator


# =========================================================
# Helpers
# =========================================================

def write_csv(path, df):
    df.to_csv(path, index=False)


# =========================================================
# compute_correlations
# =========================================================

def test_corr_basic_numeric():
    df = pd.DataFrame({
        "Brightness": [1, 2, 3],
        "Mean_R": [2, 4, 6],
        "NonNumeric": ["x", "y", "z"],
    })
    corr = correlator.compute_correlations(df)

    assert "Brightness" in corr.columns
    assert "Mean_R" in corr.columns
    assert "NonNumeric" not in corr.columns
    assert corr.loc["Brightness", "Mean_R"] == pytest.approx(1.0)


def test_corr_insufficient_rows():
    df = pd.DataFrame({"Brightness": [10], "Mean_R": [20]})
    assert correlator.compute_correlations(df).empty


def test_corr_missing_feature_columns():
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert correlator.compute_correlations(df).empty


def test_corr_rows_with_nan_are_dropped():
    df = pd.DataFrame({
        "Brightness": [1, np.nan, 3],
        "Mean_R": [2, 4, 6],
    })
    corr = correlator.compute_correlations(df)
    assert "Brightness" in corr.columns
    assert corr.loc["Brightness", "Mean_R"] == pytest.approx(1.0)


def test_corr_all_rows_invalid():
    df = pd.DataFrame({
        "Brightness": [np.nan, np.nan],
        "Mean_R": [np.nan, np.nan],
    })
    assert correlator.compute_correlations(df).empty


def test_corr_only_one_valid_feature():
    df = pd.DataFrame({"Brightness": [1, 2, 3]})
    assert correlator.compute_correlations(df).empty


# =========================================================
# group_by_texture
# =========================================================

def test_group_by_texture_basic():
    df = pd.DataFrame({
        "Texture_Class": ["smooth", "smooth", "grainy"],
        "Brightness": [10, 20, 30],
        "Mean_R": [100, 200, 300],
    })
    grouped = correlator.group_by_texture(df)

    assert set(grouped.index) == {"smooth", "grainy"}
    assert grouped.loc["smooth", "Brightness"] == pytest.approx(15.0)
    assert grouped.loc["grainy", "Brightness"] == pytest.approx(30.0)


def test_group_by_texture_missing_column():
    df = pd.DataFrame({"Brightness": [10, 20]})
    assert correlator.group_by_texture(df).empty


def test_group_by_texture_all_nan_classes():
    df = pd.DataFrame({
        "Texture_Class": [np.nan, np.nan],
        "Brightness": [10, 20],
    })
    assert correlator.group_by_texture(df).empty


def test_group_by_texture_missing_feature_columns():
    df = pd.DataFrame({
        "Texture_Class": ["smooth", "grainy"],
        "A": [1, 2],
    })
    grouped = correlator.group_by_texture(df)
    assert grouped.empty


def test_group_by_texture_single_row():
    df = pd.DataFrame({
        "Texture_Class": ["smooth"],
        "Brightness": [10],
        "Mean_R": [100],
    })
    grouped = correlator.group_by_texture(df)
    assert grouped.loc["smooth", "Brightness"] == 10


# =========================================================
# compute_statistics
# =========================================================

def test_stats_basic():
    df = pd.DataFrame({
        "Brightness": [1, 2, 3, 4],
        "Mean_R": [10, 20, 30, 40],
    })
    stats = correlator.compute_statistics(df)

    assert "Brightness" in stats.columns
    assert "Mean_R" in stats.columns
    assert stats.loc["mean", "Brightness"] == pytest.approx(2.5)


def test_stats_no_features():
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert correlator.compute_statistics(df).empty


def test_stats_with_nan():
    df = pd.DataFrame({
        "Brightness": [1, np.nan, 3],
        "Mean_R": [10, 20, np.nan],
    })
    stats = correlator.compute_statistics(df)
    assert "Brightness" in stats.columns
    assert "Mean_R" in stats.columns


def test_stats_single_row():
    df = pd.DataFrame({"Brightness": [10], "Mean_R": [20]})
    stats = correlator.compute_statistics(df)
    assert stats.loc["count", "Brightness"] == 1


# =========================================================
# compute_data_density
# =========================================================

def test_density_no_texture_column():
    df = pd.DataFrame({"x": [1, 2]})
    assert correlator.compute_data_density(df) == {}


def test_density_ignores_nan():
    df = pd.DataFrame({"Texture_Class": ["a", np.nan, "a"]})
    density = correlator.compute_data_density(df)
    assert density == {"a": "Low Confidence (n=2)"}


def test_density_empty_column():
    df = pd.DataFrame({"Texture_Class": []})
    assert correlator.compute_data_density(df) == {}


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"smooth": 1}, {"smooth": "Low Confidence (n=1)"}),
        ({"smooth": 4}, {"smooth": "Low Confidence (n=4)"}),
        ({"smooth": 5}, {"smooth": "Medium Confidence (n=5)"}),
        ({"smooth": 11}, {"smooth": "Medium Confidence (n=11)"}),
        ({"smooth": 12}, {"smooth": "High Confidence (n=12)"}),
    ],
)
def test_density_thresholds(counts, expected):
    rows = []
    for cls, n in counts.items():
        rows.extend([{"Texture_Class": cls} for _ in range(n)])
    df = pd.DataFrame(rows)
    assert correlator.compute_data_density(df) == expected


# =========================================================
# generate_markdown_report
# =========================================================

def test_report_full(tmp_path, monkeypatch):
    report_path = tmp_path / "correlations.md"
    monkeypatch.setattr(correlator, "REPORT_PATH", str(report_path))

    corr = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        columns=["Brightness", "Mean_R"],
        index=["Brightness", "Mean_R"],
    )
    grouped = pd.DataFrame(
        [[10.0, 20.0]],
        columns=["Brightness", "Texture"],
        index=["smooth"],
    )
    stats = pd.DataFrame(
        [[4, 2.5], [3, 1.5]],
        columns=["count", "mean"],
        index=["Brightness", "Mean_R"],
    )
    density = {"smooth": "High Confidence (n=12)"}

    correlator.generate_markdown_report(corr, grouped, stats, density)
    content = report_path.read_text()

    assert "Correlation Report" in content
    assert "Human-Readable Insights" in content
    assert "strong positive correlation" in content
    assert "High Confidence" in content


def test_report_empty_sections(tmp_path, monkeypatch):
    report_path = tmp_path / "correlations.md"
    monkeypatch.setattr(correlator, "REPORT_PATH", str(report_path))

    correlator.generate_markdown_report(
        corr=pd.DataFrame(),
        grouped=pd.DataFrame(),
        stats=pd.DataFrame(),
        density={}
    )

    content = report_path.read_text()
    assert "Insufficient data" in content
    assert "No texture groups available" in content
    assert "No statistics available" in content
    assert "No texture classification data available" in content


# =========================================================
# generate_correlated_images
# =========================================================

def test_copy_correlated_images(tmp_path, monkeypatch):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))

    (tmp_path / "img1_analyzed.jpg").write_bytes(b"jpeg1")
    (tmp_path / "img2_analyzed.jpeg").write_bytes(b"jpeg2")

    correlator.generate_correlated_images()

    assert (tmp_path / "img1_correlated.jpg").read_bytes() == b"jpeg1"
    assert (tmp_path / "img2_correlated.jpeg").read_bytes() == b"jpeg2"


def test_copy_correlated_ignores_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    os.makedirs(tmp_path / "subdir")
    correlator.generate_correlated_images()
    assert True  # Should not crash


# =========================================================
# main
# =========================================================

def test_main_missing_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(correlator, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(correlator, "REPORT_PATH", str(tmp_path / "correlations.md"))

    correlator.main()
    assert "field_data.csv not found" in capsys.readouterr().out


def test_main_minimal_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(correlator, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(correlator, "REPORT_PATH", str(tmp_path / "correlations.md"))

    df = pd.DataFrame({"Brightness": [10]})
    write_csv(tmp_path / "field_data.csv", df)

    correlator.main()

    assert (tmp_path / "correlations.md").is_file()


def test_main_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(correlator, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(correlator, "REPORT_PATH", str(tmp_path / "correlations.md"))

    df = pd.DataFrame({
        "Photo_Filename": ["img1_ingested.jpg", "img2_ingested.jpg"],
        "Brightness": [10, 20],
        "Mean_R": [100, 120],
        "Texture_Class": ["smooth", "grainy"],
    })
    write_csv(tmp_path / "field_data.csv", df)

    (tmp_path / "img1_analyzed.jpg").write_bytes(b"jpeg1")
    (tmp_path / "img2_analyzed.jpg").write_bytes(b"jpeg2")

    correlator.main()

    assert (tmp_path / "correlations.md").is_file()
    assert (tmp_path / "img1_correlated.jpg").is_file()
    assert (tmp_path / "img2_correlated.jpg").is_file()



