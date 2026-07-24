"""Unit tests for surg_mov_pred_logic.py -- exercises the tool's own logic
directly (column cleaning, ID detection, zip/model loading, prediction,
saving), independently of the HTTP layer.

Run just this module's tests:
    cd server && ./venv/bin/pytest tools/surg_mov_pred/test/
Or as part of the full suite:
    cd server && ./venv/bin/pytest
"""

import os
import zipfile

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tools.surg_mov_pred.src.surg_mov_pred_logic import (
    LoadData,
    Surg_mov_pred,
    clean_name,
    find_id_column,
    predict_all_targets,
    save_results,
)


# ---------------------------------------------------------------------------
# clean_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Age (years)", "Age_years"),
        ("SNA total", "SNA_Total"),
        ("  spaced  name  ", "spaced_name"),
        ('weird"name[1]', "weirdname1"),
        ("1stMeasure", "f_1stMeasure"),
        ("", "f_unnamed"),
        ("Jarabak's Ratio", "Jarabak's_Ratio"),
    ],
)
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


# ---------------------------------------------------------------------------
# find_id_column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "columns, expected",
    [
        (["PatientID", "Age"], "PatientID"),
        (["Patient ID", "Age"], "Patient ID"),
        (["#", "Age"], "#"),
        (["Subject_ID", "Age"], "Subject_ID"),
        (["Patient Number", "Age"], "Patient Number"),
        (["Age", "Weight"], None),
        (["patient_id_extra_stuff"], "patient_id_extra_stuff"),  # fallback match
    ],
)
def test_find_id_column(columns, expected):
    assert find_id_column(columns) == expected


# ---------------------------------------------------------------------------
# LoadData.extract_model
# ---------------------------------------------------------------------------

def _make_model_zip(tmp_path, package: dict, target_folder: str = "SomeTarget_Pred") -> str:
    model_dir = tmp_path / "model_src" / target_folder
    model_dir.mkdir(parents=True)
    joblib.dump(package, model_dir / "stacking_package.pkl")

    zip_path = tmp_path / "model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(model_dir / "stacking_package.pkl", f"{target_folder}/stacking_package.pkl")
    return str(zip_path)


def test_extract_model_loads_valid_package(tmp_path):
    package = {
        "target_name": "SomeTarget_Pred",
        "features_names": ["f1"],
        "scaler": StandardScaler(),
        "model": LinearRegression(),
    }
    zip_path = _make_model_zip(tmp_path, package)

    packages = LoadData.extract_model(zip_path)

    assert list(packages.keys()) == ["SomeTarget_Pred"]
    assert packages["SomeTarget_Pred"]["target_name"] == "SomeTarget_Pred"


def test_extract_model_raises_when_no_package_found(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")

    with pytest.raises(FileNotFoundError):
        LoadData.extract_model(str(zip_path))


# ---------------------------------------------------------------------------
# LoadData.load_input
# ---------------------------------------------------------------------------

def _make_input_zip(tmp_path, df: pd.DataFrame, zip_name: str = "input.zip") -> str:
    input_dir = tmp_path / "input_src"
    input_dir.mkdir(exist_ok=True)
    csv_path = input_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    zip_path = tmp_path / zip_name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, "data.csv")
    return str(zip_path)


def test_load_input_reads_csv_from_zip(tmp_path):
    df = pd.DataFrame({"PatientID": [1, 2], "f1": [10, 20]})
    zip_path = _make_input_zip(tmp_path, df)

    loaded = LoadData.load_input(zip_path)

    assert list(loaded.columns) == ["PatientID", "f1"]
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# predict_all_targets
# ---------------------------------------------------------------------------

def test_predict_all_targets_happy_path():
    scaler = StandardScaler().fit([[0], [10], [20]])
    model = LinearRegression().fit(scaler.transform([[0], [10], [20]]), [0, 100, 200])
    packages = {
        "target_a": {"features_names": ["f1"], "scaler": scaler, "model": model},
    }
    df = pd.DataFrame({"f1": [0, 10, 20]})

    results = predict_all_targets(df, packages)

    assert "target_a" in results.columns
    assert len(results) == 3


def test_predict_all_targets_skips_target_with_missing_features():
    scaler = StandardScaler().fit([[0], [10]])
    model = LinearRegression().fit(scaler.transform([[0], [10]]), [0, 1])
    packages = {
        "target_missing": {"features_names": ["does_not_exist"], "scaler": scaler, "model": model},
    }
    df = pd.DataFrame({"f1": [1, 2, 3]})

    results = predict_all_targets(df, packages)

    assert "target_missing" not in results.columns


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def test_save_results_writes_excel_and_csv(tmp_path):
    df = pd.DataFrame({"IDPatient": [1, 2], "target_a": [10.0, 20.0]})

    excel_path = save_results(df, str(tmp_path / "out"))

    assert os.path.exists(excel_path)
    assert excel_path.endswith(".xlsx")
    assert os.path.exists(str(tmp_path / "out" / "predictions_outputs.csv"))


# ---------------------------------------------------------------------------
# Surg_mov_pred.main -- full pipeline, model.zip + input.zip -> result file
# ---------------------------------------------------------------------------

def test_main_full_pipeline(tmp_path):
    scaler = StandardScaler().fit([[0], [10], [20]])
    model = LinearRegression().fit(scaler.transform([[0], [10], [20]]), [5, 15, 25])
    package = {
        "target_name": "f1_Pred",
        "features_names": ["f1"],
        "scaler": scaler,
        "model": model,
    }
    model_zip = _make_model_zip(tmp_path, package, target_folder="f1_Pred")

    df = pd.DataFrame({"PatientID": [1, 2, 3], "f1": [0, 10, 20]})
    input_zip = _make_input_zip(tmp_path, df)

    output_path = Surg_mov_pred.main(model_zip, input_zip)

    assert os.path.exists(output_path)
    result_df = pd.read_excel(output_path, index_col=0)
    assert "IDPatient" in result_df.columns
    assert "f1_Pred" in result_df.columns
    assert list(result_df["IDPatient"]) == [1, 2, 3]
