

import json
from pathlib import Path

import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

from datapipeline.bike_data_pipeline import run_training_pipeline, run_inference_pipeline


def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative predictions
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def main():
    # 1) Get processed data from your DataOps pipeline
    X_train, y_train, preprocessor = run_training_pipeline()
    X_test = run_inference_pipeline(preprocessor)

    # 2) Train LightGBM (baseline config — can tune later)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3) Quick sanity evaluation on training data (MVP)
    preds_train = np.maximum(model.predict(X_train), 0)
    metrics = {
        "MAE": float(mean_absolute_error(y_train, preds_train)),
        "RMSE": float(np.sqrt(mean_squared_error(y_train, preds_train))),
        "R2": float(r2_score(y_train, preds_train)),
        "RMSLE": rmsle(y_train, preds_train),
    }

    print("LightGBM training-set metrics (MVP sanity check):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 4) Save artifacts (model + preprocessor + metadata)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_dir / "lightgbm_model.joblib")
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    meta = {
        "model_type": "LightGBM",
        "features_shape_train": [int(X_train.shape[0]), int(X_train.shape[1])],
        "features_shape_test": [int(X_test.shape[0]), int(X_test.shape[1])],
        "metrics": metrics,
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 5) Predict test set and generate submission file
    test_preds = np.round(np.maximum(model.predict(X_test), 0)).astype(int)

    import pandas as pd
    test_df = pd.read_csv("test.csv")
    submission = pd.DataFrame({"datetime": test_df["datetime"], "count": test_preds})
    submission.to_csv("modeltraining/submission_lightgbm.csv", index=False)

    print("Saved artifacts to ./artifacts/")
    print("Saved predictions to submission_lightgbm.csv")


if __name__ == "__main__":
    main()
