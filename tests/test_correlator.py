import os
import shutil
import textwrap

import numpy as np
import pandas as pd
import pytest

import universal_field_toolkit_correlator_sred as correlator


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def write_csv(path, df):
    df.to_csv(path, index=False)


# ---------------------------------------------------------
# compute_correlations
# ---------------------------------------------------------

def test_compute_correlations_numeric_only():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2, 4, 6],
        "c": ["x", "y", "z"],  # non-numeric
    })
    corr = correlator.compute_correlations(df)
    assert set(corr.columns) == {"a", "b"}
    assert set(corr.index) == {"a", "b"}
    # a and b perfectly correlated
    assert corr.loc["a", "b]"] == pytest.approx(1.0)


# ---------------------------------------------------------
# group_by_texture
# ---------------------------------------------------------

def test_group_by_texture_with_column():
    df = pd.DataFrame({
        "Texture_Class": ["smooth", "smooth", "grainy"],
        "Brightness": [10, 20, 30],
        "NonNumeric": ["a", "b", "c"],
    })
    grouped = correlator.group_by_texture(df)
    # Should group by texture and average numeric cols only
    assert set(grouped.index) == {"smooth", "grainy"}
    assert "Brightness" in grouped.columns
    assert "NonNumeric" not in grouped.columns
    assert grouped.loc["smooth", "Brightness"] == pytest.approx(15.0)
    assert grouped.loc["grainy", "Brightness"] == pytest.approx(30.0)


def test_group_by_texture_without_column():
    df = pd.DataFrame({
        "Brightness": [10, 20, 30],
    })
    grouped = correlator.group_by_texture(df)
    assert isinstance(grouped, pd.DataFrame)
    assert grouped.empty


# ---------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------

def test_compute_statistics_basic():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40],
    })
    stats = correlator.compute_statistics(df)
    # pandas describe returns count, mean, std, min, 25%, 50%, 75%, max
    assert "x" in stats.columns
    assert "y" in stats.columns
    assert "mean" in stats.index
    assert stats.loc["mean", "x"] == pytest.approx(2.5)
    assert stats.loc["mean", "y"] == pytest.approx(25.0)


# ---------------------------------------------------------
# compute_data_density
# ---------------------------------------------------------

def test_compute_data_density_no_texture_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    density = correlator.compute_data_density(df)
    assert density == {}


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"smooth": 1}, {"smooth": "Low Confidence (n=1)"}),
        ({"smooth": 4}, {"smooth": "Low Confidence (n=4)"}),
        ({"smooth": 5}, {"smooth": "Medium Confidence (n=5)"}),
        ({"smooth": 11}, {"smooth": "Medium Confidence (n=11)"}),
        ({"smooth": 12}, {"smooth": "High Confidence (n=12)"}),
        ({"smooth": 20}, {"smooth": "High Confidence (n=20)"}),
    ],
)
def test_compute_data_density_confidence_levels(counts, expected):
    rows = []
    for cls, n in counts.items():
        rows.extend([{"Texture_Class": cls} for _ in range(n)])
    df = pd.DataFrame(rows)
    density = correlator.compute_data_density(df)
    assert density == expected


def test_compute_data_density_multiple_classes():
    df = pd.DataFrame({
        "Texture_Class": ["a", "a", "a", "b", "b", "b", "b", "b",
                          "c"] * 2  # make some variety
    })
    density = correlator.compute_data_density(df)
    # Just ensure all keys present and messages contain class + n
    for cls, msg in density.items():
        assert cls in df["Texture_Class"].unique()
        assert "(n=" in msg


# ---------------------------------------------------------
# generate_markdown_report
# ---------------------------------------------------------

def test_generate_markdown_report_full(tmp_path, monkeypatch):
    # Patch REPORT_PATH
    report_path = tmp_path / "correlations.md"
    monkeypatch.setattr(correlator, "REPORT_PATH", str(report_path))

    corr = pd.DataFrame(
        [[1.0, 0.5],
         [0.5, 1.0]],
        columns=["a", "b"],
        index=["a", "b"],
    )
    grouped = pd.DataFrame(
        [[10.0, 20.0]],
        columns=["Brightness", "Texture"],
        index=["smooth"],
    )
    stats = pd.DataFrame(
        [[4, 2.5], [3, 1.5]],
        columns=["count", "mean"],
        index=["x", "y"],
    )
    density = {"smooth": "High Confidence (n=12)"}

    correlator.generate_markdown_report(corr, grouped, stats, density)

    assert report_path.is_file()
    content = report_path.read_text()
    # basic checks
    assert "# Correlation Report" in content
    assert "## Correlation Matrix" in content
    assert "## Grouped by Texture" in content
    assert "## Data Density" in content
    assert "smooth" in content
    assert "High Confidence (n=12)" in content


def test_generate_markdown_report_no_grouped(tmp_path, monkeypatch):
    report_path = tmp_path / "correlations.md"
    monkeypatch.setattr(correlator, "REPORT_PATH", str(report_path))

    corr = pd.DataFrame([[1.0]], columns=["a"], index=["a"])
    grouped = pd.DataFrame()  # empty
    stats = pd.DataFrame([[1]], columns=["count"], index=["x"])
    density = {}

    correlator.generate_markdown_report(corr, grouped, stats, density)

    content = report_path.read_text()
    assert "_No texture groups available_" in content
    # No bullet points if density empty
    assert "- " not in content.split("## Data Density")[1]


# ---------------------------------------------------------
# generate_correlated_images
# ---------------------------------------------------------

def test_generate_correlated_images_copies_analyzed_files(tmp_path, capsys, monkeypatch):
    # Patch WORKING_DIR
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))

    # Create some files
    analyzed1 = tmp_path / "img1_analyzed.jpg"
    analyzed2 = tmp_path / "img2_analyzed.jpg"
    other = tmp_path / "note.txt"

    analyzed1.write_bytes(b"fakejpeg1")
    analyzed2.write_bytes(b"fakejpeg2")
    other.write_text("ignore me")

    correlator.generate_correlated_images()

    out1 = tmp_path / "img1_correlated.jpg"
    out2 = tmp_path / "img2_correlated.jpg"
    assert out1.is_file()
    assert out2.is_file()
    assert out1.read_bytes() == b"fakejpeg1"
    assert out2.read_bytes() == b"fakejpeg2"

    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_generate_correlated_images_no_analyzed_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    # only non-analyzed file
    (tmp_path / "something.jpg").write_bytes(b"data")

    correlator.generate_correlated_images()
    captured = capsys.readouterr()
    # No "Created" lines expected
    assert "Created" not in captured.out


# ---------------------------------------------------------
# update_csv_with_summary
# ---------------------------------------------------------

def test_update_csv_with_summary_adds_column_and_saves(tmp_path, monkeypatch):
    # Patch CSV_PATH
    csv_path = tmp_path / "field_data.csv"
    monkeypatch.setattr(correlator, "CSV_PATH", str(csv_path))

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    corr = pd.DataFrame(
        np.eye(2),
        columns=["a", "b"],
        index=["a", "b"],
    )

    write_csv(csv_path, df)
    correlator.update_csv_with_summary(df, corr)

    # Re-read and validate
    updated = pd.read_csv(csv_path)
    assert "Global_Correlation_Count" in updated.columns
    # 2 columns in corr
    assert all(updated["Global_Correlation_Count"] == 2)


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def test_main_missing_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(correlator, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(correlator, "REPORT_PATH", str(tmp_path / "correlations.md"))

    correlator.main()
    captured = capsys.readouterr()
    assert "field_data.csv not found" in captured.out
    assert not (tmp_path / "correlations.md").exists()


def test_main_happy_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(correlator, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(correlator, "CSV_PATH", str(tmp_path / "field_data.csv"))
    monkeypatch.setattr(correlator, "REPORT_PATH", str(tmp_path / "correlations.md"))

    # Create CSV with numeric + Texture_Class
    df = pd.DataFrame({
        "Photo_Filename": ["img1_ingested.jpg", "img2_ingested.jpg"],
        "Brightness": [10, 20],
        "Mean_R": [100, 120],
        "Texture_Class": ["smooth", "grainy"],
    })
    write_csv(tmp_path / "field_data.csv", df)

    # Create analyzed images
    (tmp_path / "img1_analyzed.jpg").write_bytes(b"jpeg1")
    (tmp_path / "img2_analyzed.jpg").write_bytes(b"jpeg2")

    correlator.main()

    captured = capsys.readouterr()
    assert "correlations.md generated" in captured.out
    assert "Correlator complete" in captured.out

    # Check report exists
    assert (tmp_path / "correlations.md").is_file()

    # Check correlated images created
    assert (tmp_path / "img1_correlated.jpg").is_file()
    assert (tmp_path / "img2_correlated.jpg").is_file()

    # Check CSV updated with Global_Correlation_Count
    updated = pd.read_csv(tmp_path / "field_data.csv")
    assert "Global_Correlation_Count" in updated.columns
    assert all(updated["Global_Correlation_Count"] > 0)



