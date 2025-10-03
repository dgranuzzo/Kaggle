from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_DIR = Path(__file__).resolve().parent
CV_FOLDS = 5


def load_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y_train = train_df["SalePrice"]
    X_train = train_df.drop(columns=["SalePrice", "Id"])
    X_test = test_df.drop(columns=["Id"])
    test_ids = test_df["Id"]
    return X_train, y_train, X_test, test_ids, train_df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_lgbm_pipeline(features: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_xgb_pipeline(features: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_pipeline(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -cv_scores
    mean_rmse = float(rmse_scores.mean())
    std_rmse = float(rmse_scores.std())
    print(f"{name}: mean RMSE = {mean_rmse:.2f}, std = {std_rmse:.2f}")
    return mean_rmse, std_rmse


def save_predictions(name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, test_ids: pd.Series) -> Path:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    output_path = DATA_DIR / f"{name.lower()}_predictions.csv"
    pd.DataFrame({"Id": test_ids, "SalePrice": predictions}).to_csv(output_path, index=False)
    print(f"Saved {name} predictions to {output_path}")
    return output_path


def main() -> None:
    X_train, y_train, X_test, test_ids, _ = load_train_test(DATA_DIR)

    lgbm_pipeline = build_lgbm_pipeline(X_train)
    xgb_pipeline = build_xgb_pipeline(X_train)

    print("Evaluating pipelines with cross-validation...")
    lgbm_metrics = evaluate_pipeline("LightGBM", lgbm_pipeline, X_train, y_train)
    xgb_metrics = evaluate_pipeline("XGBoost", xgb_pipeline, X_train, y_train)

    print("\nFitting on full training data and generating submissions...")
    lgbm_path = save_predictions("LightGBM", lgbm_pipeline, X_train, y_train, X_test, test_ids)
    xgb_path = save_predictions("XGBoost", xgb_pipeline, X_train, y_train, X_test, test_ids)

    best_model = "LightGBM" if lgbm_metrics[0] <= xgb_metrics[0] else "XGBoost"
    print(f"\nBest model based on CV RMSE: {best_model}")
    print(f"Files created: {lgbm_path.name}, {xgb_path.name}")


if __name__ == "__main__":
    main()
