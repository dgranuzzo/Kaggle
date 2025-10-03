from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent
CV_FOLDS = 5


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    return train_df, train_df["SalePrice"], train_df.drop(columns=["SalePrice", "Id"])


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


def build_random_forest_pipeline(features: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_boosted_pipeline(features: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=None,
        max_iter=1500,
        l2_regularization=0.0,
        random_state=42,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_model(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -cv_scores
    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    return {
        "name": name,
        "rmse_scores": rmse_scores,
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "predictions": predictions,
    }


def plot_rmse_comparison(results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [res["name"] for res in results]
    means = [res["rmse_mean"] for res in results]
    stds = [res["rmse_std"] for res in results]

    ax.bar(names, means, yerr=stds, capsize=8, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Cross-validated RMSE Comparison")
    for idx, mean in enumerate(means):
        ax.text(idx, mean + stds[idx] + 500, f"{mean:,.0f}", ha="center", fontsize=10)
    fig.tight_layout()

    output_path = DATA_DIR / "model_rmse_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_predictions_vs_actual(y: pd.Series, results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    y_array = y.to_numpy()
    min_price, max_price = y_array.min(), y_array.max()

    colors = ["#1f77b4", "#ff7f0e"]
    for idx, res in enumerate(results):
        ax.scatter(
            y_array,
            res["predictions"],
            alpha=0.4,
            s=25,
            label=f"{res['name']}",
            color=colors[idx],
        )

    ax.plot([min_price, max_price], [min_price, max_price], "k--", linewidth=1)
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title("Cross-validated Predictions vs. Actual Prices")
    ax.legend()
    fig.tight_layout()

    output_path = DATA_DIR / "predictions_vs_actual.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    train_df, y, X = load_data(DATA_DIR)

    rf_pipeline = build_random_forest_pipeline(X)
    hgb_pipeline = build_boosted_pipeline(X)

    rf_results = evaluate_model("RandomForest", rf_pipeline, X, y)
    hgb_results = evaluate_model("HistGradientBoosting", hgb_pipeline, X, y)

    results = [rf_results, hgb_results]
    results.sort(key=lambda res: res["rmse_mean"])
    best_model = results[0]

    rmse_plot = plot_rmse_comparison(results)
    scatter_plot = plot_predictions_vs_actual(y, results)

    print("Model comparison complete.")
    for res in results:
        print(
            f"{res['name']}: mean RMSE = {res['rmse_mean']:.2f}, std = {res['rmse_std']:.2f}"
        )
    print(f"Best performing model: {best_model['name']}")
    print(f"Saved RMSE comparison chart to {rmse_plot}")
    print(f"Saved prediction scatter plot to {scatter_plot}")


if __name__ == "__main__":
    main()
