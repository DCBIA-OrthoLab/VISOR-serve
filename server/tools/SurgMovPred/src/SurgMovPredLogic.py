#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
import tempfile
import warnings
from pathlib import Path
import logging
import pandas as pd
import joblib

import file_utils

# The deployed sklearn version commonly differs from the one used to train the models.
# This is expected (models are tested for compatibility before being shipped) and not
# an actual error, so it shouldn't be surfaced as a CLI failure.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ===== Logging Configuration =====
logger = logging.getLogger("SurgMovPred_CLI")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("SurgMovPred_CLI.py initialization")


def clean_name(name: str) -> str:
    """Cleans a column name so it exactly matches the training-time format."""
    try:
        name = str(name).strip()
        name = re.sub(r'[\r\n\t]', ' ', name)

        # 1. Strip problematic quotes and brackets
        name = name.replace('"', '').replace('\\', '').replace('[', '').replace(']', '')

        # 2. Replace spaces and dashes with underscores
        # (keeping the apostrophe (') for Jarabak's and SM_A'_CP!)
        name = re.sub(r'[^0-9a-zA-Z_\']+', '_', name)

        # 3. Collapse multiple underscores
        name = re.sub(r'_+', '_', name).strip('_')

        # 4. Model-specific adjustment
        name = name.replace("total", "Total")

        if re.match(r'^\d', name):
            name = f'f_{name}'

        return name or 'f_unnamed'
    except Exception:
        logger.exception(f"Error cleaning column name '{name}'")
        return 'f_unnamed'


# Common naming conventions used by different users for the patient identifier column.
# Matched case-insensitively, with optional separator (space/underscore/dash) between words.
ID_COLUMN_PATTERNS = [
    r'^#$',
    r'^id$',
    r'^id[\s_-]?patient$',
    r'^patient[\s_-]?id$',
    r'^patient[\s_-]?(num|number|no)$',
    r'^subject[\s_-]?id$',
    r'^patient$',
    r'^subject$',
]


def find_id_column(columns) -> str:
    """
    Tries to identify which input column holds the patient identifier.
    Naming conventions vary a lot between users (e.g. '#', 'ID', 'PatientID', 'Patient Number'...),
    so this matches a broad set of common patterns instead of a single fixed name.

    Returns the matching column name, or None if nothing matched.
    """
    normalized = [(col, str(col).strip().lower()) for col in columns]

    for pattern in ID_COLUMN_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for col, norm in normalized:
            if regex.fullmatch(norm):
                return col

    # Fallback: any column whose name contains both "patient" and "id"
    for col, norm in normalized:
        if 'patient' in norm and 'id' in norm:
            return col

    return None


class LoadData:
    """Extracts the model package and loads the input data used by SuperMovPred.

    Zip extraction and CSV/XLSX/ODS loading are generic concerns, factored out
    into file_utils.py so other tools can reuse them; only the
    "find stacking_package.pkl inside the extracted model" part is specific
    to this tool and stays here.
    """

    @staticmethod
    def extract_model(model_path: str, scratch_dir: str = None) -> dict:
        """Loads every 'stacking_package.pkl' found under `model_path`.

        `model_path` is either a model folder served directly from the data
        store (no archive step, the usual case) or a zip archive that first
        needs extracting (e.g. a legacy packaged model). A zip is extracted
        into `scratch_dir` when given (the model itself may live in the
        read-only data store), next to the archive otherwise.
        """
        if os.path.isdir(model_path):
            logger.info(f"Loading model package from folder: {model_path}")
            return LoadData._load_model_packages(model_path)

        logger.info(f"Extracting model package: {model_path}")
        base_dir = scratch_dir or os.path.dirname(model_path)
        extract_dir = file_utils.extract_zip(
            model_path, os.path.join(base_dir, "model_extracted")
        )
        return LoadData._load_model_packages(extract_dir)

    @staticmethod
    def _load_model_packages(base_model_folder: str) -> dict:
        """Recursively walks the extracted model folder to load every 'stacking_package.pkl' file."""
        packages = {}
        package_files = list(Path(base_model_folder).glob("**/stacking_package.pkl"))

        if not package_files:
            raise FileNotFoundError(f"No 'stacking_package.pkl' model package found in {base_model_folder}")

        logger.info(f"Loading {len(package_files)} model(s)...")
        for pkl_path in package_files:
            try:
                package = joblib.load(pkl_path)
                packages[package['target_name']] = package
            except Exception:
                logger.exception(f"Unable to load package {pkl_path}")

        if not packages:
            raise RuntimeError(f"None of the {len(package_files)} model package(s) found in {base_model_folder} could be loaded.")

        logger.info(f"Successfully loaded {len(packages)} model package(s).")
        return packages

    @staticmethod
    def load_input(input_path: str, scratch_dir: str = None) -> pd.DataFrame:
        """Loads the input data: a zip archive of CSV/XLSX/ODS files, a folder
        of such files (e.g. served from the data store), or a single file.

        A zip is extracted into `scratch_dir` when given (the input may live
        in the read-only data store), next to the archive otherwise.
        """
        logger.info(f"Loading input data: {input_path}")
        if os.path.isdir(input_path):
            df = file_utils.load_tabular_directory(input_path)
        elif input_path.lower().endswith(".zip"):
            base_dir = scratch_dir or os.path.dirname(input_path)
            extract_dir = file_utils.extract_zip(
                input_path, os.path.join(base_dir, "input_extracted")
            )
            df = file_utils.load_tabular_directory(extract_dir)
        else:
            df = file_utils.load_tabular_file(input_path)
        logger.info(f"Input data loaded: {len(df)} rows")
        return df


def predict_all_targets(df: pd.DataFrame, packages: dict) -> pd.DataFrame:
    """
    Predicts the values for each loaded model, dynamically adapting to the input data.
    """
    try:
        # 1. Clean the input file's column names so they match the training-time format
        df_cleaned = df.copy()
        df_cleaned.columns = [clean_name(col) for col in df_cleaned.columns]

        # Collect predictions in a plain dict and build the DataFrame once at the end,
        # instead of inserting one column at a time (which fragments the DataFrame
        # and triggers pandas' PerformanceWarning).
        predictions_by_target = {}

        logger.info("Starting predictions for all target variables...")

        for target_name, pack in packages.items():
            try:
                expected_features = pack['features_names']
                scaler = pack['scaler']
                model = pack['model']

                # Resolve each expected feature to an actual column in the input file.
                # Some files provide T0 measurements without the "_T0" suffix used at training time.
                feature_source = {}
                missing_features = []
                for f in expected_features:
                    if f in df_cleaned.columns:
                        feature_source[f] = f
                    elif f.endswith('_T0') and f[:-3] in df_cleaned.columns:
                        feature_source[f] = f[:-3]
                    else:
                        missing_features.append(f)

                if missing_features:
                    logger.warning(f"⚠️ Model '{target_name}' skipped: missing {len(missing_features)} input feature(s) (e.g. {missing_features[:3]})")
                    continue

                # Extract and order the data according to this model's specific needs
                X_target = df_cleaned[[feature_source[f] for f in expected_features]]
                X_target.columns = expected_features

                # Standardize using the model's own scaler
                X_scaled = scaler.transform(X_target)
                X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features, index=df_cleaned.index)

                # Predict
                predictions_by_target[target_name] = model.predict(X_scaled_df)

            except Exception:
                logger.exception(f"Error predicting target '{target_name}'")

        results_df = pd.DataFrame(predictions_by_target, index=df_cleaned.index)

        logger.info(f"Predictions complete. {len(results_df.columns)} target(s) predicted out of {len(packages)} available.")
        return results_df

    except Exception:
        logger.exception("General error during prediction")
        raise


def save_results(df: pd.DataFrame, output_folder: str) -> str:
    """Saves the predictions table as Excel and CSV. Returns the Excel file path."""
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    excel_output = out_path / "predictions_outputs.xlsx"
    csv_output = out_path / "predictions_outputs.csv"

    logger.info(f"Saving results to: {out_path}")
    df.to_excel(excel_output, index=True)
    df.to_csv(csv_output, index=True)

    logger.info("Results saved successfully!")
    return str(excel_output)



class SurgMovPred:
    @staticmethod
    def main(model: str, input: str) -> str:
        try:
            logger.info("=== Surgical Movements Prediction Engine (Stacking Deploy) ===")

            # Scratch space for extraction and outputs. Preferably a subfolder
            # of the request's upload work dir (cleaned up by main.py with the
            # rest of that directory); when every input was served straight
            # from the read-only data store that dir isn't writable, so fall
            # back to a fresh TEMP_DIR folder (also cleaned up by main.py --
            # see file_utils.make_scratch_dir). Writability is probed by
            # actually creating the folder: os.access() is not reliable for
            # read-only mounts when running as root.
            try:
                scratch_dir = tempfile.mkdtemp(prefix="run_", dir=os.path.dirname(input))
            except OSError:
                scratch_dir = file_utils.make_scratch_dir("SurgMovPred_")

            # 1. Load every model package (a folder served from the data store,
            # or a zip archive to extract first)
            packages = LoadData.extract_model(model, scratch_dir)

            # 2. Load the input data (zip archive, folder, or single CSV / Excel / ODS file)
            df_input = LoadData.load_input(input, scratch_dir)

            # 3. Recover the patient identifier from the input so it can be carried over to the output
            id_column = find_id_column(df_input.columns)
            if id_column is not None:
                logger.info(f"Detected patient ID column: '{id_column}'")
                id_values = df_input[id_column].reset_index(drop=True)
            else:
                logger.warning("Could not detect a patient ID column in the input data; 'IDPatient' will be left empty in the output.")
                id_values = pd.Series([pd.NA] * len(df_input))

            # 4. Predict every target
            df_results = predict_all_targets(df_input, packages)
            df_results.insert(0, 'IDPatient', id_values.values)

            # 5. Save into the scratch dir chosen above; main.py schedules its
            # containing directory for cleanup after the response is streamed,
            # so the output is removed automatically without its own hook.
            output_dir = os.path.join(scratch_dir, "output")
            output_path = save_results(df_results, output_dir)

            logger.info("=== Process completed successfully ===")
            return output_path

        except Exception:
            # Each step above already logs its own exception with a full traceback;
            # re-raise so the tool server can report a clean failure.
            logger.exception("Process failed")
            raise
