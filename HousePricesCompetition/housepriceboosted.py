from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df


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


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(features)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=None,
        max_iter=1500,
        l2_regularization=0.0,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    train_df, test_df = load_data(DATA_DIR)

    target_col = "SalePrice"
    id_col = "Id"

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col, id_col])
    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    pipeline = build_pipeline(X_train)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -cv_scores
    print(f"Cross-validated RMSE: mean={rmse_scores.mean():.2f}, std={rmse_scores.std():.2f}")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({id_col: test_ids, target_col: predictions})
    output_path = DATA_DIR / "hgb_predictions.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
