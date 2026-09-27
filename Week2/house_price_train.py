"""Reproducible House Prices workflow for the CRISP-DM assignment.

Run from the project directory:
    python house_price_train.py

Inputs: train.csv, test.csv
Outputs: submission.csv, model_comparison.csv, oof_predictions.csv,
         house_price_mlp.pt, house_price_preprocessor.joblib, figures/*.png
"""

from pathlib import Path
import copy
import json
import platform
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
N_SPLITS = 5
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 64
ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class HousePriceMLP(nn.Module):
    """Feed-forward regressor: input -> 128 -> 64 -> 32 -> one log-price."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def run_workflow(params=None, show_plots: bool = False, mlp_class=HousePriceMLP):
    """Run EDA, preprocessing, CV, model comparison, and submission checks."""
    params = params or {}
    data_params = params.get("data", {})
    train_params = params.get("train", {})
    validation_params = params.get("validation", {})
    experiment_params = params.get("experiment", {})
    seed = int(params.get("seed", SEED))
    n_splits = int(validation_params.get("n_splits", N_SPLITS))
    max_epochs = int(train_params.get("max_epochs", MAX_EPOCHS))
    patience = int(train_params.get("patience", PATIENCE))
    batch_size = int(train_params.get("batch_size", BATCH_SIZE))
    outlier_area_threshold = int(data_params.get("outlier_area_threshold", 4000))
    input_dir = Path(data_params.get("input_dir", ROOT))
    train_file = data_params.get("train_file", "train.csv")
    test_file = data_params.get("test_file", "test.csv")
    show_plots = bool(experiment_params.get("show_plots", show_plots))
    set_seed(seed)
    torch.set_num_threads(1)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng (DEVICE): {DEVICE}")

    # CRISP-DM 1: Business understanding and target definition.
    all_train_df = pd.read_csv(input_dir / train_file)
    test_df = pd.read_csv(input_dir / test_file)
    required_train = {"Id", "SalePrice"}
    assert required_train.issubset(all_train_df.columns)
    assert "Id" in test_df.columns
    assert set(test_df.columns) == set(all_train_df.columns) - {"SalePrice"}

    test_ids = test_df["Id"].copy()
    print(f"Số dòng/thuộc tính train gốc: {all_train_df.shape[0]} dòng; {all_train_df.shape[1] - 2} thuộc tính")
    print(f"Khoảng giá SalePrice: ${all_train_df.SalePrice.min():,.0f} - ${all_train_df.SalePrice.max():,.0f}")


    # CRISP-DM 2: Data understanding. Save plots and concise diagnostics for the report.
    raw_X = all_train_df.drop(columns=["Id", "SalePrice"])
    missing = (raw_X.isna().mean().sort_values(ascending=False) * 100).rename("missing_pct")
    print("Top 15 thuộc tính có tỉ lệ thiếu dữ liệu cao nhất (%):")
    print(missing.head(15).round(1).to_string())
    print("\nThống kê mô tả SalePrice:")
    print(all_train_df["SalePrice"].describe().round(0).to_string())
    outliers = all_train_df.loc[
        all_train_df["GrLivArea"] > outlier_area_threshold,
        ["Id", "GrLivArea", "SalePrice"],
    ]
    print(f"\nSố dòng có GrLivArea > {outlier_area_threshold:,} sqft: {len(outliers)}")
    print(outliers.to_string(index=False) if len(outliers) else "Không có dòng nào.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_train_df["SalePrice"], bins=35, color="#4C78A8", edgecolor="white")
    axes[0].set(title="SalePrice distribution", xlabel="SalePrice (USD)", ylabel="Count")
    axes[1].scatter(all_train_df["GrLivArea"], all_train_df["SalePrice"], s=14, alpha=0.55)
    axes[1].set(title="Living area vs SalePrice", xlabel="Above-ground living area (sq ft)", ylabel="SalePrice (USD)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_target_and_living_area.png", dpi=160)
    if show_plots:
        from IPython.display import display
        display(fig)
    plt.close(fig)

    # De Cock recommends examining and removing houses above 4,000 sq ft because
    # several are partial-sale outliers. Apply that documented rule to training rows;
    # preserve every test Id so the submission still covers the competition test set.
    keep_rows = all_train_df["GrLivArea"] <= outlier_area_threshold
    train_df = all_train_df.loc[keep_rows].reset_index(drop=True)
    train_ids = train_df["Id"].copy()
    y = np.log1p(train_df["SalePrice"].astype(float)).to_numpy()
    X = train_df.drop(columns=["Id", "SalePrice"])
    X_test = test_df.drop(columns=["Id"]).reindex(columns=X.columns)
    print(f"\nĐã loại {int((~keep_rows).sum())} dòng train vượt ngưỡng diện tích.")
    print(f"Kích thước train dùng để huấn luyện: {X.shape}; test: {X_test.shape}")


    # CRISP-DM 3: Data preparation. Imputation and encoding are learned inside each fold.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", encoder),
            ]), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


    # CRISP-DM 4: Modeling. Classical models and a PyTorch MLP use identical folds/features.
    sklearn_models = {
        "Ridge": Ridge(alpha=15.0),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=20000),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=2,
            min_samples_leaf=5, loss="huber", random_state=seed,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, max_features=0.8, min_samples_leaf=2,
            n_jobs=1, random_state=seed,
        ),
    }


    def fit_mlp_early_stopping(X_fit, y_fit, X_valid, y_valid, seed):
        """Fit on an inner training split and stop using inner validation loss."""
        set_seed(seed)
        target_mean = float(np.mean(y_fit))
        target_std = float(np.std(y_fit))
        y_fit_scaled = (y_fit - target_mean) / target_std
        y_valid_scaled = (y_valid - target_mean) / target_std
        model = mlp_class(X_fit.shape[1]).to(DEVICE)
        train_ds = TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit_scaled, dtype=torch.float32).reshape(-1, 1),
        )
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        x_valid_t = torch.tensor(X_valid, dtype=torch.float32, device=DEVICE)
        y_valid_t = torch.tensor(y_valid_scaled, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss, best_epoch, best_state, stale_epochs = float("inf"), 0, None, 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_loss = loss_fn(model(x_valid_t), y_valid_t).item()
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            valid_pred = model(x_valid_t).cpu().numpy().ravel() * target_std + target_mean
        return model, valid_pred, best_epoch


    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {name: np.full(len(X), np.nan) for name in [*sklearn_models, "PyTorch_MLP"]}
    best_epochs = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_fit, X_valid = X.iloc[fit_idx], X.iloc[valid_idx]
        y_fit, y_valid = y[fit_idx], y[valid_idx]
        print(f"\n--- Fold {fold}/{n_splits} ---")

        for name, estimator in sklearn_models.items():
            pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
            pipe.fit(X_fit, y_fit)
            oof[name][valid_idx] = pipe.predict(X_valid)

        inner_fit_idx, inner_valid_idx = train_test_split(
            np.arange(len(fit_idx)), test_size=0.15, random_state=seed + fold
        )
        inner_preprocessor = clone(preprocessor)
        X_inner_fit_t = inner_preprocessor.fit_transform(X_fit.iloc[inner_fit_idx])
        X_inner_valid_t = inner_preprocessor.transform(X_fit.iloc[inner_valid_idx])
        _, _, best_epoch = fit_mlp_early_stopping(
            X_inner_fit_t, y_fit[inner_fit_idx],
            X_inner_valid_t, y_fit[inner_valid_idx],
            seed=seed + fold,
        )
        # Refit this fold model on the full outer-fold training set for best_epoch epochs.
        fold_preprocessor = clone(preprocessor)
        X_fit_t = fold_preprocessor.fit_transform(X_fit)
        X_valid_t = fold_preprocessor.transform(X_valid)
        set_seed(seed + fold)
        fold_model = mlp_class(X_fit_t.shape[1]).to(DEVICE)
        fold_target_mean = float(np.mean(y_fit))
        fold_target_std = float(np.std(y_fit))
        y_fit_scaled = (y_fit - fold_target_mean) / fold_target_std
        ds = TensorDataset(
            torch.tensor(X_fit_t, dtype=torch.float32),
            torch.tensor(y_fit_scaled, dtype=torch.float32).reshape(-1, 1),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + fold))
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        for _ in range(best_epoch):
            fold_model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(fold_model(xb), yb)
                loss.backward()
                optimizer.step()
        fold_model.eval()
        with torch.no_grad():
            oof["PyTorch_MLP"][valid_idx] = fold_model(
                torch.tensor(X_valid_t, dtype=torch.float32, device=DEVICE)
            ).cpu().numpy().ravel() * fold_target_std + fold_target_mean
        best_epochs.append(best_epoch)
        print(f"  Số epoch tốt nhất của MLP (early stopping): {best_epoch}")


    # CRISP-DM 5: Evaluation. Report Kaggle RMSLE and dollar-scale error summaries.
    def regression_metrics(y_log, pred_log):
        pred_log = np.maximum(np.asarray(pred_log), 0.0)
        actual_price = np.expm1(y_log)
        predicted_price = np.maximum(np.expm1(pred_log), 0.0)
        return {
            "RMSLE": float(np.sqrt(mean_squared_error(y_log, pred_log))),
            "MAE_USD": float(mean_absolute_error(actual_price, predicted_price)),
            "Bias_USD_pred_minus_actual": float(np.mean(predicted_price - actual_price)),
            "Max_abs_error_USD": float(np.max(np.abs(predicted_price - actual_price))),
        }


    weights = {
        "Ridge": 0.30,
        "ElasticNet": 0.20,
        "GradientBoosting": 0.18,
        "ExtraTrees": 0.12,
        "PyTorch_MLP": 0.20,
    }
    oof["Blend"] = sum(weights[name] * oof[name] for name in weights)
    metrics = pd.DataFrame.from_dict(
        {name: regression_metrics(y, pred) for name, pred in oof.items()},
        orient="index",
    )
    metrics.index.name = "Model"
    metrics = metrics.sort_values("RMSLE")
    metrics.to_csv(ROOT / "model_comparison.csv")
    pd.DataFrame({"Id": train_ids, "SalePrice": train_df["SalePrice"],
                  **{f"oof_{name}": np.expm1(np.maximum(pred, 0)) for name, pred in oof.items()}}
                 ).to_csv(ROOT / "oof_predictions.csv", index=False)
    print("\nSo sánh 5-fold out-of-fold (giá trị càng thấp càng tốt):")
    print(metrics.round(4).to_string())
    print(f"\nSố epoch trung vị của MLP qua {n_splits} fold: {int(np.median(best_epochs))}")
    candidate_scores = metrics["RMSLE"]
    selected_model = candidate_scores.idxmin()
    print(f"=> Mô hình được chọn để nộp bài (RMSLE OOF thấp nhất): {selected_model}")


    # CRISP-DM 6: Deployment. Refit preprocessing/models, predict test, validate submission.
    test_log_predictions = {}
    for name, estimator in sklearn_models.items():
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
        pipe.fit(X, y)
        test_log_predictions[name] = pipe.predict(X_test)

    final_preprocessor = clone(preprocessor)
    X_full_t = final_preprocessor.fit_transform(X)
    X_test_t = final_preprocessor.transform(X_test)
    final_epochs = max(1, int(np.median(best_epochs)))
    set_seed(seed)
    final_mlp = mlp_class(X_full_t.shape[1]).to(DEVICE)
    final_target_mean = float(np.mean(y))
    final_target_std = float(np.std(y))
    y_full_scaled = (y - final_target_mean) / final_target_std
    final_ds = TensorDataset(
        torch.tensor(X_full_t, dtype=torch.float32),
        torch.tensor(y_full_scaled, dtype=torch.float32).reshape(-1, 1),
    )
    final_loader = DataLoader(final_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(final_mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(final_epochs):
        final_mlp.train()
        epoch_loss = 0.0
        for xb, yb in final_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(final_mlp(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if (epoch + 1) % 25 == 0 or epoch + 1 == final_epochs:
            print(f"MLP cuối cùng - epoch {epoch+1}/{final_epochs}; train MSE = {epoch_loss/len(X):.5f}")

    final_mlp.eval()
    with torch.no_grad():
        test_log_predictions["PyTorch_MLP"] = final_mlp(
            torch.tensor(X_test_t, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().ravel() * final_target_std + final_target_mean

    test_log_blend = sum(weights[name] * test_log_predictions[name] for name in weights)
    final_log_prediction = (test_log_blend if selected_model == "Blend"
                            else test_log_predictions[selected_model])
    submission = pd.DataFrame({
        "Id": test_ids.to_numpy(),
        "SalePrice": np.maximum(np.expm1(np.maximum(final_log_prediction, 0)), 0),
    })
    submission_path = ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False)

    assert list(submission.columns) == ["Id", "SalePrice"]
    assert len(submission) == len(test_df)
    assert submission["Id"].equals(test_ids.reset_index(drop=True))
    assert submission["SalePrice"].notna().all()
    assert np.isfinite(submission["SalePrice"]).all()
    assert (submission["SalePrice"] >= 0).all()
    assert submission["Id"].is_unique

    joblib.dump(final_preprocessor, ROOT / "house_price_preprocessor.joblib")
    torch.save({
        "model_state_dict": final_mlp.cpu().state_dict(),
        "input_dim": int(X_full_t.shape[1]),
        "architecture": [128, 64, 32],
        "dropout": [0.15, 0.10],
        "target_transform": "log1p(SalePrice)",
        "target_mean": final_target_mean,
        "target_std": final_target_std,
        "epochs": final_epochs,
            "seed": seed,
        "cv_best_epochs": best_epochs,
    }, ROOT / "house_price_mlp.pt")

    print(f"\nMô hình dùng để nộp bài: {selected_model}; đã lưu {submission_path} với {len(submission)} dòng hợp lệ.")
    print(submission.head().to_string(index=False))
    print(f"Khoảng giá dự đoán: ${submission.SalePrice.min():,.0f} - ${submission.SalePrice.max():,.0f}")
    print("Đã kiểm tra: schema, số dòng, thứ tự/tính duy nhất của Id, giá trị hữu hạn và không âm — tất cả hợp lệ.")

    run_summary = {
        "seed": seed,
        "cv": f"{n_splits}-fold shuffled KFold",
        "raw_training_rows": int(len(all_train_df)),
        "training_rows_after_outlier_rule": int(len(train_df)),
        "large_area_rows_removed_from_training": int((~keep_rows).sum()),
        "test_rows_predicted": int(len(submission)),
        "selected_model": selected_model,
        "selected_oof_rmsle": float(metrics.loc[selected_model, "RMSLE"]),
        "mlp_best_epochs_by_fold": [int(e) for e in best_epochs],
        "training_config": {
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "batch_size": batch_size,
            "n_splits": n_splits,
            "outlier_area_threshold": outlier_area_threshold,
        },
        "data_files": {"train": str(input_dir / train_file), "test": str(input_dir / test_file)},
        "metrics": metrics.round(8).to_dict(orient="index"),
        "submission_price_min": float(submission["SalePrice"].min()),
        "submission_price_max": float(submission["SalePrice"].max()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
    }
    (ROOT / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print("Đã lưu tóm tắt tái lập vào run_summary.json")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "outliers": outliers,
        "metrics": metrics,
        "submission": submission,
        "selected_model": selected_model,
        "figure_path": FIGURES / "eda_target_and_living_area.png",
    }


if __name__ == "__main__":
    run_workflow()
