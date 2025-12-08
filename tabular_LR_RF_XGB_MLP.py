
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
)


try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# Global maps


TARGET_MAP = {"home": 0, "draw": 1, "away": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}



# Utilities

def to_datetime_utc(series: pd.Series) -> pd.Series:
    """Safely convert to UTC datetimes."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def detect_history_indices(columns, base_prefix):
    """
    Detect indices for columns like 'home_team_history_goal_1', ..., '_10'

    base_prefix examples:
        'home_team_history_goal'
        'away_team_history_rating'
    """
    import re

    pattern = rf"^{base_prefix}_(\d+)$"
    idxs = []
    for c in columns:
        m = re.match(pattern, c)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))


def frac_true(x: np.ndarray) -> np.ndarray:
    """Fraction of ones per row, ignoring NaNs."""
    if x.size == 0:
        return np.zeros(x.shape[0])
    denom = np.sum(~np.isnan(x), axis=1)
    numer = np.nansum(x, axis=1)
    out = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    return out


# Extra feature helpers 


def add_coach_change_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flags indicating whether home/away team has changed coach
    relative to its 10-match history.

    This uses only the per-row history columns, so it does not leak future data.
    """
    for side in ["home", "away"]:
        curr = pd.to_numeric(df.get(f"{side}_team_coach_id", np.nan), errors="coerce")
        flags = []
        for i in range(1, 11):
            hist = pd.to_numeric(
                df.get(f"{side}_team_history_coach_{i}", np.nan),
                errors="coerce"
            )
            flags.append((hist.notna()) & (hist != curr))
        if flags:
            any_change = np.any(np.stack(flags, axis=1), axis=1).astype(int)
        else:
            any_change = np.zeros(len(df), dtype=int)
        df[f"{side}_has_coach_change"] = any_change
    return df


def add_league_goal_rate_proxy_from_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-match 'league_average_goal_proxy' using only the 10-match
    histories (both home and away) for that row.

    This is NOT a true league-wide aggregate across all matches, but a
    history-based proxy per sample, avoiding explicit leakage from future data.
    """
    N = len(df)
    home_hist_goals = []
    away_hist_goals = []

    for i in range(1, 11):
        hg = pd.to_numeric(df.get(f"home_team_history_goal_{i}", np.nan),
                           errors="coerce")
        hga = pd.to_numeric(df.get(f"home_team_history_opponent_goal_{i}", np.nan),
                            errors="coerce")
        ag = pd.to_numeric(df.get(f"away_team_history_goal_{i}", np.nan),
                           errors="coerce")
        aga = pd.to_numeric(df.get(f"away_team_history_opponent_goal_{i}", np.nan),
                            errors="coerce")

        home_hist_goals.append((hg + hga).values)
        away_hist_goals.append((ag + aga).values)

    if not home_hist_goals:
        df["league_average_goal_proxy"] = 2.5  # safe fallback
        return df

    all_hist = np.stack(home_hist_goals + away_hist_goals, axis=1)  # (N, 20)
    total_goals = np.nansum(all_hist, axis=1)
    total_games = np.sum(~np.isnan(all_hist), axis=1)

    avg_goals = np.divide(
        total_goals,
        total_games,
        out=np.zeros_like(total_goals, dtype=float),
        where=total_games > 0,
    )

    # Fill very small or zero values with a reasonable default
    avg_goals[~np.isfinite(avg_goals)] = np.nan
    df["league_average_goal_proxy"] = avg_goals
    mean_val = np.nanmean(df["league_average_goal_proxy"].values)
    if not np.isfinite(mean_val):
        mean_val = 2.5
    df["league_average_goal_proxy"].fillna(mean_val, inplace=True)
    return df


# Data loading & feature engineering


def load_and_merge(data_dir: Path) -> pd.DataFrame:
    """Load train.csv and train_target_and_scores.csv and merge on id."""
    train = pd.read_csv(data_dir / "train.csv")
    tts = pd.read_csv(data_dir / "train_target_and_scores.csv")

    if "id" not in train.columns or "id" not in tts.columns:
        raise ValueError(
            "Expected 'id' column in both train.csv and train_target_and_scores.csv."
        )

    df = train.merge(
        tts[["id", "score", "target"]], on="id", how="left", suffixes=("", "_tts")
    )

    # Keep only one target column
    if "target_tts" in df.columns:
        df.drop(columns=["target_tts"], inplace=True)

    return df


def build_features(df_raw: pd.DataFrame):
    """
    Implements Data Preparation and Feature Aggregation from the proposal,
    plus some additional engineered features inspired by Kaggle code.

    - chronological consistency
    - aggregated stats (goals, ratings, W/D/L rates, home%, cup%)
    - pairwise home/away comparative features
    - rest days, is_cup, league_id
    - coach-change flags (home/away)
    - league goal-rate proxy from histories
    - simple attack/defense strength and expected-goals proxies
    """
    df = df_raw.copy()

    # --- Chronological Consistency ---
    if "match_date" not in df.columns:
        raise ValueError("Expected 'match_date' in train.csv.")
    df["match_date"] = to_datetime_utc(df["match_date"])
    df.sort_values("match_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add coach-change flags and league goal proxy first
    df = add_coach_change_flags(df)
    df = add_league_goal_rate_proxy_from_history(df)

    cols = df.columns.tolist()

    # Detect history indices
    h_goal_idx = detect_history_indices(cols, "home_team_history_goal")
    h_ogoal_idx = detect_history_indices(cols, "home_team_history_opponent_goal")
    h_rating_idx = detect_history_indices(cols, "home_team_history_rating")
    h_opprating_idx = detect_history_indices(cols, "home_team_history_opponent_rating")
    h_homeflag_idx = detect_history_indices(cols, "home_team_history_is_play_home")
    h_cupflag_idx = detect_history_indices(cols, "home_team_history_is_cup")

    a_goal_idx = detect_history_indices(cols, "away_team_history_goal")
    a_ogoal_idx = detect_history_indices(cols, "away_team_history_opponent_goal")
    a_rating_idx = detect_history_indices(cols, "away_team_history_rating")
    a_opprating_idx = detect_history_indices(cols, "away_team_history_opponent_rating")
    a_homeflag_idx = detect_history_indices(cols, "away_team_history_is_play_home")
    a_cupflag_idx = detect_history_indices(cols, "away_team_history_is_cup")

    def get_block(df_, prefix, idxs, as_float=True):
        arrs = []
        for i in idxs:
            col = f"{prefix}_{i}"
            if col in df_.columns:
                arrs.append(pd.to_numeric(df_[col], errors="coerce").values)
            else:
                arrs.append(np.full(len(df_), np.nan))
        if not arrs:
            return np.empty((len(df_), 0))
        arr = np.vstack(arrs).T  # shape (n_samples, len(idxs))
        if as_float:
            arr = arr.astype(float)
        return arr

    def get_flag(df_, prefix, idxs):
        arrs = []
        for i in idxs:
            col = f"{prefix}_{i}"
            if col in df_.columns:
                arrs.append(
                    pd.to_numeric(df_[col], errors="coerce")
                    .fillna(0)
                    .astype(float)
                    .values
                )
            else:
                arrs.append(np.zeros(len(df_)))
        if not arrs:
            return np.empty((len(df_), 0))
        return np.vstack(arrs).T

    # Home blocks
    H_goal = get_block(df, "home_team_history_goal", h_goal_idx)
    H_ogoal = get_block(df, "home_team_history_opponent_goal", h_ogoal_idx)
    H_rating = get_block(df, "home_team_history_rating", h_rating_idx)
    H_opprating = get_block(
        df, "home_team_history_opponent_rating", h_opprating_idx
    )
    H_homeflag = get_flag(df, "home_team_history_is_play_home", h_homeflag_idx)
    H_cupflag = get_flag(df, "home_team_history_is_cup", h_cupflag_idx)

    # Away blocks
    A_goal = get_block(df, "away_team_history_goal", a_goal_idx)
    A_ogoal = get_block(df, "away_team_history_opponent_goal", a_ogoal_idx)
    A_rating = get_block(df, "away_team_history_rating", a_rating_idx)
    A_opprating = get_block(
        df, "away_team_history_opponent_rating", a_opprating_idx
    )
    A_homeflag = get_flag(df, "away_team_history_is_play_home", a_homeflag_idx)
    A_cupflag = get_flag(df, "away_team_history_is_cup", a_cupflag_idx)

    def agg_team(goals, opp_goals, rating, opp_rating, homeflag, cupflag, prefix):
        n = goals.shape[0]
        if goals.size == 0:
            return pd.DataFrame(
                {
                    f"{prefix}_{k}": np.zeros(n)
                    for k in [
                        "gf_mean",
                        "ga_mean",
                        "gd_mean",
                        "gf_var",
                        "ga_var",
                        "win_rate",
                        "draw_rate",
                        "loss_rate",
                        "pct_home",
                        "pct_cup",
                        "rating_mean",
                        "opp_rating_mean",
                    ]
                }
            )

        gf = goals
        ga = opp_goals
        gd = gf - ga

        win = (gf > ga).astype(float)
        draw = (gf == ga).astype(float)
        loss = (gf < ga).astype(float)

        gf_mean = np.nanmean(gf, axis=1)
        ga_mean = np.nanmean(ga, axis=1)
        gd_mean = np.nanmean(gd, axis=1)
        gf_var = np.nanvar(gf, axis=1)
        ga_var = np.nanvar(ga, axis=1)

        win_rate = frac_true(win)
        draw_rate = frac_true(draw)
        loss_rate = frac_true(loss)

        pct_home = frac_true(homeflag)
        pct_cup = frac_true(cupflag)

        rating_mean = np.nanmean(rating, axis=1) if rating.size > 0 else np.zeros(n)
        opp_rating_mean = (
            np.nanmean(opp_rating, axis=1) if opp_rating.size > 0 else np.zeros(n)
        )

        return pd.DataFrame(
            {
                f"{prefix}_gf_mean": gf_mean,
                f"{prefix}_ga_mean": ga_mean,
                f"{prefix}_gd_mean": gd_mean,
                f"{prefix}_gf_var": gf_var,
                f"{prefix}_ga_var": ga_var,
                f"{prefix}_win_rate": win_rate,
                f"{prefix}_draw_rate": draw_rate,
                f"{prefix}_loss_rate": loss_rate,
                f"{prefix}_pct_home": pct_home,
                f"{prefix}_pct_cup": pct_cup,
                f"{prefix}_rating_mean": rating_mean,
                f"{prefix}_opp_rating_mean": opp_rating_mean,
            }
        )

    home_agg = agg_team(
        H_goal, H_ogoal, H_rating, H_opprating, H_homeflag, H_cupflag, "home"
    )
    away_agg = agg_team(
        A_goal, A_ogoal, A_rating, A_opprating, A_homeflag, A_cupflag, "away"
    )

    # Base feature frame
    X = pd.concat([home_agg, away_agg], axis=1)

    # Pairwise differences: home - away
    for base in [
        "gf_mean",
        "ga_mean",
        "gd_mean",
        "gf_var",
        "ga_var",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "pct_home",
        "pct_cup",
        "rating_mean",
        "opp_rating_mean",
    ]:
        X[f"diff_{base}"] = X[f"home_{base}"] - X[f"away_{base}"]

    # Context features: is_cup, league_id, rest days
    if "is_cup" in df.columns:
        # 'False'/'True' strings -> 0/1
        X["is_cup"] = (
            df["is_cup"]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes"])
            .astype(int)
        )
    if "league_id" in df.columns:
        X["league_id"] = df["league_id"].astype("category").cat.codes

    # Rest days (days since last match) using history date_1
    def days_since_last(prefix):
        col = f"{prefix}_1"
        if col in df.columns:
            last_dt = to_datetime_utc(df[col])
            return (df["match_date"] - last_dt).dt.days.fillna(0).astype(float)
        return pd.Series(np.zeros(len(df)))

    X["home_rest_days"] = days_since_last("home_team_history_match_date")
    X["away_rest_days"] = days_since_last("away_team_history_match_date")
    X["diff_rest_days"] = X["home_rest_days"] - X["away_rest_days"]

    # Add coach-change flags
    X["home_has_coach_change"] = df["home_has_coach_change"].astype(float)
    X["away_has_coach_change"] = df["away_has_coach_change"].astype(float)

    # Add league goal-rate proxy
    X["league_average_goal_proxy"] = df["league_average_goal_proxy"].astype(float)

    # Attack/defense strength and simple expected-goals proxies
    eps = 1e-6
    X["home_attack_strength"] = X["home_gf_mean"] / (X["league_average_goal_proxy"] + eps)
    X["home_defense_weakness"] = X["home_ga_mean"] / (X["league_average_goal_proxy"] + eps)
    X["away_attack_strength"] = X["away_gf_mean"] / (X["league_average_goal_proxy"] + eps)
    X["away_defense_weakness"] = X["away_ga_mean"] / (X["league_average_goal_proxy"] + eps)

    X["home_expected_goals_proxy"] = (
        X["league_average_goal_proxy"]
        * X["home_attack_strength"]
        * X["away_defense_weakness"]
    )
    X["away_expected_goals_proxy"] = (
        X["league_average_goal_proxy"]
        * X["away_attack_strength"]
        * X["home_defense_weakness"]
    )

    # Target
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in merged data.")
    y = df["target"].map(TARGET_MAP)
    if y.isnull().any():
        raise ValueError("Unexpected target labels; expected {'home','draw','away'}.")

    return X, y.astype(int), df["match_date"].copy()


# -------------------------------------------------------------------
# Temporal splitting & class balance
# -------------------------------------------------------------------

def temporal_train_val_test_split(dates: pd.Series):
    """
    Implements temporal split as in the proposal:
    - Train: 2019–2020
    - Val:   early 2021 (Jan–Jun)
    - Test:  late 2021 (Jul–Dec)
    If the data does not cover those spans well, falls back to
    70% train, 15% val, 15% test by date quantiles.
    """
    d = to_datetime_utc(dates)

    train_mask = d.dt.year <= 2020
    val_mask = (d.dt.year == 2021) & (d.dt.month <= 6)
    test_mask = (d.dt.year == 2021) & (d.dt.month >= 7)

    # Fallback if val/test would be too small
    if val_mask.sum() < 1000 or test_mask.sum() < 1000:
        q1, q2 = d.quantile([0.7, 0.85])
        train_mask = d <= q1
        val_mask = (d > q1) & (d <= q2)
        test_mask = d > q2

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    return idx_train, idx_val, idx_test


def print_class_balance(y: pd.Series, name: str):
    vals, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\nClass balance in {name}:")
    for v, c in zip(vals, counts):
        print(f"  {INV_TARGET_MAP[v]}: {c} ({c / total:.3f})")


# -------------------------------------------------------------------
# Model training & evaluation
# -------------------------------------------------------------------

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Fit on X_train/y_train, evaluate on test, return metrics dict.
    Includes:
      - log loss
      - accuracy
      - balanced accuracy
      - full per-class precision/recall/F1/support
      - macro/weighted averages
      - confusion matrix
    """
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    preds = proba.argmax(axis=1)

    # Basic metrics
    ll = log_loss(y_test, proba, labels=[0, 1, 2])
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    # Detailed classification report (per-class and averages)
    report_dict = classification_report(
        y_test,
        preds,
        target_names=["home", "draw", "away"],
        output_dict=True,
        digits=4,
    )
    report_text = classification_report(
        y_test,
        preds,
        target_names=["home", "draw", "away"],
        digits=4,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])

    print(f"\n{name} – Test results:")
    print(f"  Log loss:          {ll:.4f}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print("  Classification report:")
    print(report_text)
    print("  Confusion matrix (rows=true, cols=pred):")
    print("          pred: home   draw   away")
    row_labels = ["home", "draw", "away"]
    for lbl, row in zip(row_labels, cm):
        print(f"    true {lbl:<4}: {row[0]:5d} {row[1]:6d} {row[2]:6d}")

    metrics = {
        "log_loss": float(ll),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),  # JSON-serializable
    }
    return metrics


def run_family_A_models(
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    calibrate_rf=False,
    fast=True,
):
    """
    Family A: Logistic Regression, Random Forest, LightGBM, XGBoost.

    fast=True  -> much lighter searches for LGBM/XGB 
    fast=False -> heavier searches 
    """
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    # =========================
    # 1) Logistic Regression
    # =========================
    lr_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    multi_class="multinomial",
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid_lr = {
        "clf__C": [0.1, 0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
    }

    print("\n[Family A] Logistic Regression – GridSearchCV with TimeSeriesSplit")
    lr_search = GridSearchCV(
        lr_pipe,
        param_grid=param_grid_lr,
        scoring="neg_log_loss",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )
    lr_search.fit(X_trainval, y_trainval)
    print("  Best params:", lr_search.best_params_)
    print(f"  Best CV log loss: {-lr_search.best_score_:.4f}")

    lr_best = lr_search.best_estimator_
    lr_metrics = evaluate_model(
        "Logistic Regression (tuned)", lr_best, X_trainval, y_trainval, X_test, y_test
    )
    results["logreg"] = {
        "cv_logloss": float(-lr_search.best_score_),
        "test": lr_metrics,
    }

    
    # 2) Random Forest
    
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", rf_base),
        ]
    )

    param_dist_rf = {
        "clf__n_estimators": [300, 500] if fast else [300, 400, 600],
        "clf__max_depth": [10, 15, None] if fast else [10, 15, 20, None],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
    }

    print("\n[Family A] Random Forest – RandomizedSearchCV with TimeSeriesSplit")
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=param_dist_rf,
        n_iter=10 if fast else 20,
        scoring="neg_log_loss",
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf_search.fit(X_trainval, y_trainval)
    print("  Best params:", rf_search.best_params_)
    print(f"  Best CV log loss: {-rf_search.best_score_:.4f}")

    rf_best = rf_search.best_estimator_
    rf_metrics = evaluate_model(
        "Random Forest (tuned)", rf_best, X_trainval, y_trainval, X_test, y_test
    )
    results["rf"] = {
        "cv_logloss": float(-rf_search.best_score_),
        "test": rf_metrics,
    }

    
    # 3) LightGBM
    
    if HAS_LGBM:
        lgbm_base = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        lgbm_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("clf", lgbm_base),
            ]
        )

        if fast:
            param_dist_lgbm = {
                "clf__n_estimators": [200, 400],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.03, 0.05],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.5, 0.8],
                "clf__min_child_samples": [20, 50],
            }
            n_iter_lgbm = 5
        else:
            param_dist_lgbm = {
                "clf__n_estimators": [300, 500, 700],
                "clf__num_leaves": [31, 63, 127],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.5, 0.8, 1.0],
                "clf__min_child_samples": [10, 20, 50],
            }
            n_iter_lgbm = 20

        print("\n[Family A] LightGBM – RandomizedSearchCV with TimeSeriesSplit")
        lgbm_search = RandomizedSearchCV(
            lgbm_pipe,
            param_distributions=param_dist_lgbm,
            n_iter=n_iter_lgbm,
            scoring="neg_log_loss",
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        lgbm_search.fit(X_trainval, y_trainval)
        print("  Best params:", lgbm_search.best_params_)
        print(f"  Best CV log loss: {-lgbm_search.best_score_:.4f}")

        lgbm_best = lgbm_search.best_estimator_
        lgbm_metrics = evaluate_model(
            "LightGBM (tuned)", lgbm_best, X_trainval, y_trainval, X_test, y_test
        )
        results["lgbm"] = {
            "cv_logloss": float(-lgbm_search.best_score_),
            "test": lgbm_metrics,
        }
    else:
        print("\n[Family A] LightGBM not installed; skipping that model.")

    
    # 4) XGBoost
    
    if HAS_XGB:
        xgb_base = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",   # fast CPU hist algorithm
            random_state=42,
            n_jobs=-1,
        )

        xgb_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                
                ("clf", xgb_base),
            ]
        )

        if fast:
            param_dist_xgb = {
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.03, 0.05],
                "clf__n_estimators": [200, 400],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5],
                "clf__min_child_weight": [1, 3],
                "clf__gamma": [0.0, 0.1],
                "clf__reg_alpha": [0.0, 1.0],
                "clf__reg_lambda": [1.0, 3.0],
            }
            n_iter_xgb = 5
        else:
            param_dist_xgb = {
                "clf__max_depth": [3, 4, 5, 6],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__n_estimators": [300, 500, 700],
                "clf__subsample": [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.3, 0.5, 0.8],
                "clf__min_child_weight": [1, 3, 5],
                "clf__gamma": [0.0, 0.1, 0.2],
                "clf__reg_alpha": [0.0, 1.0, 2.0],
                "clf__reg_lambda": [1.0, 3.0, 5.0],
            }
            n_iter_xgb = 20

        print("\n[Family A] XGBoost – RandomizedSearchCV with TimeSeriesSplit")
        xgb_search = RandomizedSearchCV(
            xgb_pipe,
            param_distributions=param_dist_xgb,
            n_iter=n_iter_xgb,
            scoring="neg_log_loss",
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        xgb_search.fit(X_trainval, y_trainval)
        print("  Best params:", xgb_search.best_params_)
        print(f"  Best CV log loss: {-xgb_search.best_score_:.4f}")

        xgb_best = xgb_search.best_estimator_
        xgb_metrics = evaluate_model(
            "XGBoost (tuned)", xgb_best, X_trainval, y_trainval, X_test, y_test
        )
        results["xgb"] = {
            "cv_logloss": float(-xgb_search.best_score_),
            "test": xgb_metrics,
        }
    else:
        print("\n[Family A] XGBoost not installed; skipping that model.")

    return results


def run_family_B_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Simple Feed-Forward MLP on aggregated features.
    Uses validation set for a quick check, then trains on train+val and
    evaluates on test.
    """
    print("\n[Family B] MLP – training on train, checking on val, evaluating on test")

    mlp = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=50,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )

    # Train on train only, evaluate on val
    mlp.fit(X_train, y_train)
    val_proba = mlp.predict_proba(X_val)
    val_logloss = log_loss(y_val, val_proba, labels=[0, 1, 2])
    val_acc = accuracy_score(y_val, val_proba.argmax(axis=1))
    print(f"  Val log loss: {val_logloss:.4f} | Val acc: {val_acc:.4f}")

    # Retrain on train+val for final test evaluation
    X_train_all = pd.concat([X_train, X_val], axis=0)
    y_train_all = pd.concat([y_train, y_val], axis=0)

    test_metrics = evaluate_model(
        "MLP (Family B)", mlp, X_train_all, y_train_all, X_test, y_test
    )
    return {
        "val_logloss": float(val_logloss),
        "val_acc": float(val_acc),
        "test": test_metrics,
    }



# Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing train.csv and train_target_and_scores.csv",
    )
    parser.add_argument(
        "--calibrate_rf",
        action="store_true",
        help="(Optional) Not used in tuned version; kept for compatibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("[1/6] Loading and merging data ...")
    df = load_and_merge(data_dir)

    print("[2/6] Building features ...")
    X, y, dates = build_features(df)

    print("[3/6] Temporal train/val/test split ...")
    idx_train, idx_val, idx_test = temporal_train_val_test_split(dates)

    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    print(f"  Train size: {len(X_train)}")
    print(f"  Val size:   {len(X_val)}")
    print(f"  Test size:  {len(X_test)}")

    print_class_balance(y_train, "train")
    print_class_balance(y_val, "val")
    print_class_balance(y_test, "test")

    # Combine train+val for Family A (CV on trainval, then test)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    print("\n[4/6] Running Family A models (LR, RF, LGBM, XGB) with tuning ...")
    family_A_results = run_family_A_models(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        calibrate_rf=args.calibrate_rf,
        fast=True,   
    )

    print("\n[5/6] Running Family B models (MLP) ...")
    family_B_results = run_family_B_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n[6/6] Done.")
    print("\nSummary (high-level):")
    for name, res in family_A_results.items():
        print(
            f"  {name}: "
            f"CV logloss={res['cv_logloss']:.4f}, "
            f"Test logloss={res['test']['log_loss']:.4f}, "
            f"Test acc={res['test']['accuracy']:.4f}, "
            f"Test bal_acc={res['test']['balanced_accuracy']:.4f}"
        )
    print(
        f"  MLP: Val logloss={family_B_results['val_logloss']:.4f}, "
        f"Test logloss={family_B_results['test']['log_loss']:.4f}, "
        f"Test acc={family_B_results['test']['accuracy']:.4f}, "
        f"Test bal_acc={family_B_results['test']['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
