"""Unit tests for SurgMovPredLogic.py -- exercises the tool's own logic
directly (column cleaning, ID detection, model loading, prediction,
saving), independently of the HTTP layer.

Run just this module's tests:
    cd server && ./venv/bin/pytest tools/SurgMovPred/test/
Or as part of the full suite:
    cd server && ./venv/bin/pytest
"""

import os
import zipfile

# Set before anything imports config.Settings() (file_utils does, via the
# tool logic), so the suite runs regardless of the local environment.
os.environ.setdefault("API_TOKEN", "test-token")

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tools.SurgMovPred.src.SurgMovPredLogic import (
    LoadData,
    surgMovPred,
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


def test_extract_model_loads_from_folder(tmp_path):
    """A model served directly from the data store is a plain folder: it must
    be loaded as-is, without any zip extraction step."""
    package = {
        "target_name": "SomeTarget_Pred",
        "features_names": ["f1"],
        "scaler": StandardScaler(),
        "model": LinearRegression(),
    }
    model_dir = tmp_path / "SomeTarget_Pred"
    model_dir.mkdir()
    joblib.dump(package, model_dir / "stacking_package.pkl")

    packages = LoadData.extract_model(str(model_dir))

    assert list(packages.keys()) == ["SomeTarget_Pred"]


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


def test_load_input_reads_bare_tabular_file(tmp_path):
    """A server-side test file may be a single CSV/XLSX/ODS, not a zip."""
    df = pd.DataFrame({"PatientID": [1, 2], "f1": [10, 20]})
    xlsx_path = tmp_path / "patients.xlsx"
    df.to_excel(xlsx_path, index=False)

    loaded = LoadData.load_input(str(xlsx_path))

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
# surgMovPred.main -- full pipeline, model.zip + input.zip -> result file
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

    output_path = surgMovPred.main(model_zip, input_zip)

    assert os.path.exists(output_path)
    result_df = pd.read_excel(output_path, index_col=0)
    assert "IDPatient" in result_df.columns
    assert "f1_Pred" in result_df.columns
    assert list(result_df["IDPatient"]) == [1, 2, 3]


def test_main_full_pipeline_with_server_side_data(tmp_path, monkeypatch):
    """Data-store layout: the model is a plain folder and the input a bare
    XLSX, both served read-only -- nothing is zipped, and the output must land
    outside the (read-only) input directory."""
    import file_utils
    from config import settings

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "server_tmp"))

    scaler = StandardScaler().fit([[0], [10], [20]])
    model = LinearRegression().fit(scaler.transform([[0], [10], [20]]), [5, 15, 25])
    package = {
        "target_name": "f1_Pred",
        "features_names": ["f1"],
        "scaler": scaler,
        "model": model,
    }
    model_dir = tmp_path / "DATA" / "models" / "f1_Pred"
    model_dir.mkdir(parents=True)
    joblib.dump(package, model_dir / "stacking_package.pkl")

    testfiles_dir = tmp_path / "DATA" / "testfiles"
    testfiles_dir.mkdir()
    input_path = testfiles_dir / "patients.xlsx"
    pd.DataFrame({"PatientID": [1, 2, 3], "f1": [0, 10, 20]}).to_excel(
        input_path, index=False
    )

    # Simulate the read-only DATA mount so the tool has to fall back to its
    # TEMP_DIR scratch dir. When running as root (e.g. in the Docker test
    # container) the chmod has no effect and the scratch simply lands next to
    # the input; the assertions hold either way.
    testfiles_dir.chmod(0o555)
    try:
        output_path = surgMovPred.main(str(model_dir), str(input_path))

        assert os.path.exists(output_path)
        result_df = pd.read_excel(output_path, index_col=0)
        assert "f1_Pred" in result_df.columns
        assert list(result_df["IDPatient"]) == [1, 2, 3]
    finally:
        testfiles_dir.chmod(0o755)
