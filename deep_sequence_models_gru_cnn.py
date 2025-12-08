

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)


# Global maps


TARGET_MAP = {"home": 0, "draw": 1, "away": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}



# Utilities


def set_seeds(seed: int = 42):
    """Make training as deterministic as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def detect_history_indices(columns, base_prefix):
    """
    Detect indices for columns like 'home_team_history_goal_1', ..., '_10'.
    """
    import re

    pattern = rf"^{base_prefix}_(\d+)$"
    idxs = []
    for c in columns:
        m = re.match(pattern, c)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))



# Data loading


def load_and_merge(data_dir: Path) -> pd.DataFrame:
    """
    Load train.csv and train_target_and_scores.csv, then merge on 'id'.
    """
    train = pd.read_csv(data_dir / "train.csv")
    tts = pd.read_csv(data_dir / "train_target_and_scores.csv")

    if "id" not in train.columns or "id" not in tts.columns:
        raise ValueError(
            "Expected 'id' column in both train.csv and train_target_and_scores.csv."
        )

    df = train.merge(
        tts[["id", "score", "target"]], on="id", how="left", suffixes=("", "_tts")
    )
    if "target_tts" in df.columns:
        df.drop(columns=["target_tts"], inplace=True)

    return df


# Sequence feature builder


def build_sequence_tensors(df_raw: pd.DataFrame):
    """
    Builds sequence tensors from up to 10-match histories.

    At each time step i, features include for HOME and AWAY:
      - goals scored
      - goals conceded
      - goal difference (gf - ga)
      - rating
      - opponent rating
      - is_play_home flag
      - is_cup flag

    Plus a mask feature indicating if this step is real history (1) or padding (0).

    Per-step feature dimension:
      home: 7 (gf, ga, gd, rating, opp_rating, is_home, is_cup)
      away: 7
      mask: 1
      total: 7 + 7 + 1 = 15

    Also builds simple static features:
      - is_cup (0/1 at match level)
      - league_id (categorical code)

    Returns:
      X_seq:       np.ndarray of shape (N, T, 15)
      static_feats np.ndarray of shape (N, 2)
      y:           np.ndarray of shape (N,)
      dates:       pd.Series of match_date for temporal split
    """
    df = df_raw.copy()

    if "match_date" not in df.columns:
        raise ValueError("Expected 'match_date' in train.csv.")
    df["match_date"] = to_datetime_utc(df["match_date"])
    df.sort_values("match_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols = df.columns.tolist()

    # Figure out which history indices are available
    h_idx = detect_history_indices(cols, "home_team_history_goal")
    a_idx = detect_history_indices(cols, "away_team_history_goal")
    common_steps = sorted(set(h_idx).intersection(a_idx))
    if not common_steps:
        raise ValueError("No common history indices detected for home/away histories.")

    # We use at most the first 10 steps
    steps = [i for i in common_steps if i <= 10][:10]
    seq_len = len(steps)

    N = len(df)
    feat_dim = 15
    X_seq = np.zeros((N, seq_len, feat_dim), dtype=np.float32)

    def get_col(name):
        if name in df.columns:
            return df[name].copy()
        else:
            return pd.Series([np.nan] * N, index=df.index)

    def to_num(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32).values

    for t, i in enumerate(steps):
        # Raw columns (before numeric conversion) – used to build mask
        h_goal_raw = get_col(f"home_team_history_goal_{i}")
        h_ogoal_raw = get_col(f"home_team_history_opponent_goal_{i}")
        h_rating_raw = get_col(f"home_team_history_rating_{i}")
        h_orating_raw = get_col(f"home_team_history_opponent_rating_{i}")
        h_is_home_raw = get_col(f"home_team_history_is_play_home_{i}")
        h_is_cup_raw = get_col(f"home_team_history_is_cup_{i}")

        a_goal_raw = get_col(f"away_team_history_goal_{i}")
        a_ogoal_raw = get_col(f"away_team_history_opponent_goal_{i}")
        a_rating_raw = get_col(f"away_team_history_rating_{i}")
        a_orating_raw = get_col(f"away_team_history_opponent_rating_{i}")
        a_is_home_raw = get_col(f"away_team_history_is_play_home_{i}")
        a_is_cup_raw = get_col(f"away_team_history_is_cup_{i}")

        # Mask step: 1 if any of these fields are non-null
        mask_step = (
            h_goal_raw.notna()
            | h_ogoal_raw.notna()
            | h_rating_raw.notna()
            | h_orating_raw.notna()
            | h_is_home_raw.notna()
            | h_is_cup_raw.notna()
            | a_goal_raw.notna()
            | a_ogoal_raw.notna()
            | a_rating_raw.notna()
            | a_orating_raw.notna()
            | a_is_home_raw.notna()
            | a_is_cup_raw.notna()
        ).astype(np.float32).values

        # Numeric conversion (fill NaNs with 0 for now)
        h_goal = to_num(h_goal_raw)
        h_ogoal = to_num(h_ogoal_raw)
        h_rating = to_num(h_rating_raw)
        h_orating = to_num(h_orating_raw)
        h_is_home = to_num(h_is_home_raw)
        h_is_cup = to_num(h_is_cup_raw)

        a_goal = to_num(a_goal_raw)
        a_ogoal = to_num(a_ogoal_raw)
        a_rating = to_num(a_rating_raw)
        a_orating = to_num(a_orating_raw)
        a_is_home = to_num(a_is_home_raw)
        a_is_cup = to_num(a_is_cup_raw)

        # Goal differences
        h_gd = h_goal - h_ogoal
        a_gd = a_goal - a_ogoal

        step_features = np.stack(
            [
                # home (7)
                h_goal,
                h_ogoal,
                h_gd,
                h_rating,
                h_orating,
                h_is_home,
                h_is_cup,
                # away (7)
                a_goal,
                a_ogoal,
                a_gd,
                a_rating,
                a_orating,
                a_is_home,
                a_is_cup,
                # mask (1)
                mask_step,
            ],
            axis=1,
        )  # shape (N, 15)

        X_seq[:, t, :] = step_features

    # Targets
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in merged data.")
    y = df["target"].map(TARGET_MAP).values
    if np.isnan(y).any():
        raise ValueError("Unexpected target labels; expected {'home','draw','away'}.")

    # Simple static features (no leakage: known at match time)
    static_feats = []

    # 1) is_cup as 0/1
    if "is_cup" in df.columns:
        is_cup = (
            df["is_cup"].astype(str)
            .str.lower()
            .isin(["1", "true", "t", "yes"])
        ).astype(np.float32).values
    else:
        is_cup = np.zeros(len(df), dtype=np.float32)
    static_feats.append(is_cup)

    # 2) league_id encoded as int
    if "league_id" in df.columns:
        league_codes = df["league_id"].astype("category").cat.codes.astype(np.float32)
    else:
        league_codes = np.zeros(len(df), dtype=np.float32)
    static_feats.append(league_codes)

    static_feats = np.stack(static_feats, axis=1)  # (N, 2)

    dates = df["match_date"].copy()
    return X_seq, static_feats, y.astype(int), dates



# Temporal split


def temporal_train_val_test_split(dates: pd.Series):
    """
    Temporal split used in the proposal:

    - Train: all matches up to and including 2020
    - Val:   2021-01 to 2021-06
    - Test:  2021-07 to 2021-12

    If val/test would be too small, falls back to a 70/15/15 split by date.
    """
    d = to_datetime_utc(dates)

    train_mask = d.dt.year <= 2020
    val_mask = (d.dt.year == 2021) & (d.dt.month <= 6)
    test_mask = (d.dt.year == 2021) & (d.dt.month >= 7)

    if val_mask.sum() < 1000 or test_mask.sum() < 1000:
        q1, q2 = d.quantile([0.7, 0.85])
        train_mask = d <= q1
        val_mask = (d > q1) & (d <= q2)
        test_mask = d > q2

    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]

    return idx_train, idx_val, idx_test


def print_class_balance(y, name: str):
    vals, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\nClass balance in {name}:")
    for v, c in zip(vals, counts):
        label = INV_TARGET_MAP[int(v)]
        print(f"  {label}: {c} ({c / total:.3f})")



# PyTorch dataset and models

class FootySeqDataset(Dataset):
    def __init__(self, X_seq, S_static, y):
        # X_seq:    (N, T, F)
        # S_static: (N, D_static)
        self.X_seq = torch.from_numpy(X_seq).float()
        self.S_static = torch.from_numpy(S_static).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.S_static[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        static_dim=2,
        hidden_dim=192,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(out_dim + static_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x_seq, x_static):
        # x_seq: (B, T, F); x_static: (B, D_static)
        out, h_n = self.gru(x_seq)
        # h_n: (num_layers * num_directions, B, H)
        if self.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            h_last = torch.cat([forward_last, backward_last], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1]  # (B, H)

        h_cat = torch.cat([h_last, x_static], dim=1)
        logits = self.fc(h_cat)
        return logits


class CNN1DClassifier(nn.Module):
    """
    Multi-kernel 1D CNN: conv with kernel sizes 3, 5, 7 in parallel, then concat.
    """
    def __init__(self, input_dim, static_dim=2, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=7, padding=3)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(64 * 3 + static_dim, num_classes)

    def forward(self, x_seq, x_static):
        # x_seq: (B, T, F); x_static: (B, D_static)
        x = x_seq.transpose(1, 2)  # (B, F, T)
        c3 = self.relu(self.conv3(x))
        c5 = self.relu(self.conv5(x))
        c7 = self.relu(self.conv7(x))
        c = torch.cat([c3, c5, c7], dim=1)  # (B, 64*3, T)
        c = self.pool(c).squeeze(-1)        # (B, 64*3)
        c = self.dropout(c)
        h_cat = torch.cat([c, x_static], dim=1)
        logits = self.fc(h_cat)
        return logits



# Training & evaluation loops


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=20,
    lr=1e-3,
    weight_decay=1e-5,
    patience=6,
):
    """
    Generic training loop for GRU/CNN models with:
    - class-weighted cross-entropy
    - ReduceLROnPlateau scheduler
    - early stopping based on validation loss
    """
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Train/val splits must be non-empty.")

    model.to(device)

    # Compute class weights from training labels
    all_targets = []
    for _, _, y_batch in train_loader:
        all_targets.append(y_batch)
    all_targets = torch.cat(all_targets)
    classes, counts = torch.unique(all_targets, return_counts=True)
    total = all_targets.size(0)
    weights = total / (len(classes) * counts.float())
    class_weights = torch.ones(3, device=device)
    for c, w in zip(classes, weights):
        class_weights[c] = w

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, S_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch, S_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, S_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                S_batch = S_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch, S_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_sequence_model(name, model, data_loader, device, verbose=True):
    """
    Evaluate GRU / CNN1D model on a given DataLoader.

    Returns:
      - log_loss
      - accuracy
      - balanced_accuracy
      - classification_report (dict with per-class + macro/weighted)
      - confusion_matrix (3x3 list)
    """
    model.to(device)
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X_batch, S_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)
            logits = model(X_batch, S_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    proba = torch.softmax(all_logits, dim=1).numpy()
    preds = proba.argmax(axis=1)

    ll = log_loss(all_labels, proba, labels=[0, 1, 2])
    acc = accuracy_score(all_labels, preds)
    bal_acc = balanced_accuracy_score(all_labels, preds)

    report_dict = classification_report(
        all_labels,
        preds,
        target_names=["home", "draw", "away"],
        digits=4,
        output_dict=True,
    )
    report_str = classification_report(
        all_labels,
        preds,
        target_names=["home", "draw", "away"],
        digits=4,
    )

    cm = confusion_matrix(all_labels, preds, labels=[0, 1, 2])

    if verbose:
        print(f"\n{name} – results:")
        print(f"  Log loss:          {ll:.4f}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced accuracy: {bal_acc:.4f}")
        print("  Classification report:\n", report_str)
        print("  Confusion matrix (rows=true, cols=pred):")
        print("          pred: home   draw   away")
        row_labels = ["home", "draw", "away"]
        for lbl, row in zip(row_labels, cm):
            print(f"    true {lbl:<4}: {row[0]:5d} {row[1]:6d} {row[2]:6d}")

    return {
        "log_loss": float(ll),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
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
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs for each model/config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(42)

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("[1/5] Loading and merging data ...")
    df = load_and_merge(data_dir)

    print("[2/5] Building sequence tensors ...")
    X_seq, static_feats, y, dates = build_sequence_tensors(df)
    print(f"  Raw sequence tensor shape:   {X_seq.shape}")        # (N, T, F)
    print(f"  Static feature shape:        {static_feats.shape}")  # (N, 2)

    print("[3/5] Temporal train/val/test split ...")
    idx_train, idx_val, idx_test = temporal_train_val_test_split(dates)

    X_train, X_val, X_test = X_seq[idx_train], X_seq[idx_val], X_seq[idx_test]
    S_train, S_val, S_test = static_feats[idx_train], static_feats[idx_val], static_feats[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    print(f"  Train size: {len(X_train)}")
    print(f"  Val size:   {len(X_val)}")
    print(f"  Test size:  {len(X_test)}")

    print_class_balance(y_train, "train")
    print_class_balance(y_val, "val")
    print_class_balance(y_test, "test")

   
    # Standardize sequence features based on training data
   
    N_tr, T, F = X_train.shape
    train_flat = X_train.reshape(-1, F)

    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std == 0.0] = 1.0

    # Do NOT normalize the mask (last dim); keep as raw 0/1
    mean[-1] = 0.0
    std[-1] = 1.0

    def standardize(x):
        return (x - mean[None, None, :]) / std[None, None, :]

    X_train = standardize(X_train)
    X_val = standardize(X_val)
    X_test = standardize(X_test)

    # Build datasets and loaders
    train_ds = FootySeqDataset(X_train, S_train, y_train)
    val_ds = FootySeqDataset(X_val, S_val, y_val)
    test_ds = FootySeqDataset(X_test, S_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    static_dim = S_train.shape[1]


    # GRU model: small hyperparameter search on val set
    
    print("\n[4/5] Training GRU sequence model(s) with small hyperparameter search ...")

    gru_configs = [
        {"hidden_dim": 192, "dropout": 0.3, "lr": 1e-3},
        {"hidden_dim": 256, "dropout": 0.3, "lr": 1e-3},
        {"hidden_dim": 256, "dropout": 0.25, "lr": 7e-4},
    ]

    best_gru_model = None
    best_gru_metric = np.inf
    best_gru_cfg = None

    for cfg in gru_configs:
        print(f"\n[GRU] Trying config: {cfg}")
        model = GRUClassifier(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=2,
            num_classes=3,
            dropout=cfg["dropout"],
            bidirectional=True,
        )
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            num_epochs=args.epochs,
            lr=cfg["lr"],
            weight_decay=1e-5,
            patience=6,
        )
        val_results = evaluate_sequence_model("GRU (val)", model, val_loader, device, verbose=False)
        print(f"  -> Val logloss={val_results['log_loss']:.4f}, acc={val_results['accuracy']:.4f}")

        # Select best by validation log loss
        if val_results["log_loss"] < best_gru_metric:
            best_gru_metric = val_results["log_loss"]
            best_gru_model = model
            best_gru_cfg = cfg

    print(f"\nBest GRU config on val: {best_gru_cfg}, val logloss={best_gru_metric:.4f}")
    gru_results_test = evaluate_sequence_model("GRU (best config, test)", best_gru_model, test_loader, device)

 
    # 1D CNN model: small hyperparameter search on val set
 
    print("\n[5/5] Training 1D CNN sequence model(s) with small hyperparameter search ...")

    cnn_configs = [
        {"dropout_p": 0.4, "lr": 5e-4},
        {"dropout_p": 0.3, "lr": 7e-4},
        {"dropout_p": 0.25, "lr": 1e-3},
    ]

    best_cnn_model = None
    best_cnn_metric = np.inf
    best_cnn_cfg = None

    for cfg in cnn_configs:
        print(f"\n[CNN] Trying config: {cfg}")
        model = CNN1DClassifier(
            input_dim=input_dim,
            static_dim=static_dim,
            num_classes=3,
            dropout_p=cfg["dropout_p"],
        )
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            num_epochs=args.epochs,
            lr=cfg["lr"],
            weight_decay=1e-5,
            patience=6,
        )
        val_results = evaluate_sequence_model("1D CNN (val)", model, val_loader, device, verbose=False)
        print(f"  -> Val logloss={val_results['log_loss']:.4f}, acc={val_results['accuracy']:.4f}")

        if val_results["log_loss"] < best_cnn_metric:
            best_cnn_metric = val_results["log_loss"]
            best_cnn_model = model
            best_cnn_cfg = cfg

    print(f"\nBest 1D CNN config on val: {best_cnn_cfg}, val logloss={best_cnn_metric:.4f}")
    cnn_results_test = evaluate_sequence_model("1D CNN (best config, test)", best_cnn_model, test_loader, device)

    print("\nSummary (sequence models, test set):")
    print(
        f"  GRU(best):   logloss={gru_results_test['log_loss']:.4f}, "
        f"accuracy={gru_results_test['accuracy']:.4f}, "
        f"bal_acc={gru_results_test['balanced_accuracy']:.4f}"
    )
    print(
        f"  CNN1D(best): logloss={cnn_results_test['log_loss']:.4f}, "
        f"accuracy={cnn_results_test['accuracy']:.4f}, "
        f"bal_acc={cnn_results_test['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
