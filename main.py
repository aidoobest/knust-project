"""
Fine-Tuning DistilBERT for ICT Misconception Detection in Open-Ended
Responses from Ghanaian Senior High School Students.

Compares a standard single-layer DistilBERT classification head against
an engineered two-layer non-linear head with a confidence-threshold output
mechanism, evaluated against six traditional ML baselines.

Author: Prince Aidoo
CAN-DO Research Lab, KNUST Department of Computer Science
"""

import random
import re
import os
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import norm

import matplotlib.pyplot as plt


# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "data/dataset1000.csv"          # adjust to your local/Kaggle path
FIGURE_DIR = "figures"
SEEDS = [42, 123, 2025]
THRESHOLD = 0.75
MAX_LENGTH = 128
BATCH_SIZE = 8

os.makedirs(FIGURE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEEDS[0])
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# ── Data loading and cleaning ──────────────────────────────────────────────
def load_and_clean_data(path):
    """
    Loads the raw corpus, drops unusable labels and missing text fields,
    and builds the combined question+answer input string.
    """
    df = pd.read_csv(path)

    # Drop rows with no usable label (keeps only Correct/Misconception)
    df = df[df["label"].notna()].copy()
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"Misconception": 1, "Correct": 0})
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # Drop the single instance with a missing response_text field
    df = df[df["response_text"].notna()].copy()

    df["combined_text"] = (
        "Question: " + df["question_text"] + " Answer: " + df["response_text"]
    )

    # Respondent identifier extracted from the trailing R-number in response_id,
    # used to prevent group leakage across train/val/test partitions.
    df["respondent_id"] = df["response_id"].str.extract(r"(R\d+)$")

    return df


def grouped_split(df, seed=SEEDS[0]):
    """
    Grouped 70/15/15 hold-out split, ensuring no respondent's rows are
    split across partitions.
    """
    set_seed(seed)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["respondent_id"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["respondent_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


# ── Model definitions ───────────────────────────────────────────────────────
class StandardDistilBERT(nn.Module):
    """Baseline: standard single-layer linear classification head."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"loss": outputs.loss, "logits": outputs.logits}


class EngineeredDistilBERT(nn.Module):
    """
    Engineered model: replaces the single linear head with
    Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, num_classes).
    """

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class RealDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            list(texts.astype(str)),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── BiLSTM-Attention baseline ───────────────────────────────────────────────
def simple_tokenize(text):
    return text.lower().split()


def build_vocab(texts, vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LENGTH):
    tokens = simple_tokenize(text)
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LENGTH):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


def make_collate_fn(vocab):
    def collate_fn(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        padded = pad_sequence(seqs, batch_first=True, padding_value=vocab["<PAD>"])
        return padded, lengths, torch.tensor(labels, dtype=torch.long)

    return collate_fn


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive attention (Bahdanau et al., 2015)."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        return self.classifier(context)


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate(model, loader, device, threshold=None):
    """
    Evaluates a DistilBERT-style model on a loader. If threshold is set,
    predictions below it are flagged as uncertain (-1) rather than assigned
    a binary label.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            probs = torch.softmax(outputs["logits"], dim=1)
            confidence, preds = torch.max(probs, dim=1)
            for i in range(len(batch["labels"])):
                all_labels.append(batch["labels"][i].item())
                all_probs.append(probs[i][1].item())
                if threshold is not None and confidence[i].item() < threshold:
                    all_preds.append(-1)
                else:
                    all_preds.append(preds[i].item())

    certain_preds = [p for p in all_preds if p != -1]
    certain_labels = [all_labels[i] for i, p in enumerate(all_preds) if p != -1]
    certain_probs = [all_probs[i] for i, p in enumerate(all_preds) if p != -1]

    macro_f1 = f1_score(certain_labels, certain_preds, average="macro")
    auc_roc = roc_auc_score(certain_labels, certain_probs) if len(set(certain_labels)) == 2 else None
    auc_pr = average_precision_score(certain_labels, certain_probs) if len(set(certain_labels)) == 2 else None

    return {
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "all_preds": all_preds,
        "certain_preds": certain_preds,
        "certain_labels": certain_labels,
        "certain_probs": certain_probs,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "uncertain_count": all_preds.count(-1),
        "total": len(all_labels),
    }


def aggregate(results_list, metric):
    vals = [r[metric] for r in results_list]
    return np.mean(vals), np.std(vals, ddof=1)


def mcnemar_with_effect_size(preds_a, preds_b, labels, name_a, name_b, alpha=0.05):
    """Runs McNemar's exact test with an odds-ratio effect size and 95% CI."""
    correct_a = [int(p == l) for p, l in zip(preds_a, labels)]
    correct_b = [int(p == l) for p, l in zip(preds_b, labels)]

    n01 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 0)
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 1)
    n11 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 1)
    n00 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 0)

    table = [[n11, n01], [n10, n00]]
    result = mcnemar(table, exact=True)

    if n01 == 0 or n10 == 0:
        odds_ratio = float("inf") if n10 == 0 and n01 > 0 else (0.0 if n01 == 0 and n10 > 0 else 1.0)
        ci_low, ci_high = float("nan"), float("nan")
    else:
        odds_ratio = n10 / n01
        se_log_or = np.sqrt(1 / n01 + 1 / n10)
        z = norm.ppf(1 - alpha / 2)
        ci_low = np.exp(np.log(odds_ratio) - z * se_log_or)
        ci_high = np.exp(np.log(odds_ratio) + z * se_log_or)

    print(f"McNemar's Test: {name_a} vs {name_b}")
    print(f"  n01={n01}, n10={n10}, n11={n11}, n00={n00}")
    print(f"  Exact p-value: {result.pvalue:.10f}")
    print(f"  Odds ratio: {odds_ratio:.3f} | 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    return {
        "pvalue": result.pvalue,
        "odds_ratio": odds_ratio,
        "ci": (ci_low, ci_high),
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n00": n00,
    }


# ── DistilBERT training loop ────────────────────────────────────────────────
def train_and_eval_condition(
    model_class, model_kwargs, condition_name, seed,
    train_loader, val_loader, test_loader, class_weights,
    threshold=None, epochs=5, patience=3,
):
    set_seed(seed)
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, patience_counter = 0, None, 0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = loss_fct(outputs["logits"], batch["labels"])
            loss.backward()
            optimizer.step()

        val_f1 = evaluate(model, val_loader, device)["macro_f1"]
        if val_f1 > best_f1:
            best_f1, best_state, patience_counter = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - train_start
    model.load_state_dict(best_state)

    infer_start = time.time()
    test_results = evaluate(model, test_loader, device, threshold=threshold)
    infer_time = time.time() - infer_start

    test_results.update({
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
        "seed": seed,
        "condition": condition_name,
    })
    print(f"  [{condition_name} | seed {seed}] Macro F1: {test_results['macro_f1']:.4f} "
          f"| Train: {train_time:.1f}s | Infer: {infer_time:.2f}s")
    return model, test_results


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    df = load_and_clean_data(DATA_PATH)
    print(f"Corpus size after cleaning: {df.shape}")

    train_df, val_df, test_df = grouped_split(df)
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

    # Class weights (inverse frequency, training partition only)
    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (2 * class_counts[i]) for i in range(2)], dtype=torch.float
    ).to(device)
    print("Class weights:", class_weights)

    # DistilBERT datasets/loaders
    train_loader = DataLoader(
        RealDataset(train_df["combined_text"], train_df["label"], tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        RealDataset(val_df["combined_text"], val_df["label"], tokenizer),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        RealDataset(test_df["combined_text"], test_df["label"], tokenizer),
        batch_size=BATCH_SIZE,
    )

    # ── Multi-seed DistilBERT training: Standard, Engineered, Engineered+Threshold
    conditions = {
        "Standard": (StandardDistilBERT, {"num_classes": 2}, None),
        "Engineered": (EngineeredDistilBERT, {"num_classes": 2, "dropout_rate": 0.3}, None),
        "Engineered+Threshold": (EngineeredDistilBERT, {"num_classes": 2, "dropout_rate": 0.3}, THRESHOLD),
    }
    all_results = {name: [] for name in conditions}
    all_models = {name: [] for name in conditions}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        for cond_name, (model_class, kwargs, thresh) in conditions.items():
            model, results = train_and_eval_condition(
                model_class, kwargs, cond_name, seed,
                train_loader, val_loader, test_loader, class_weights,
                threshold=thresh,
            )
            all_results[cond_name].append(results)
            all_models[cond_name].append(model)

    # ── Six traditional ML baselines ────────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_df["combined_text"])
    X_test_tfidf = tfidf.transform(test_df["combined_text"])
    y_train, y_test = train_df["label"].values, test_df["label"].values

    class_weight_map = {0: class_weights[0].item(), 1: class_weights[1].item()}
    sample_weight_train = np.array([class_weight_map[l] for l in y_train])

    def eval_sklearn_model(model, X_test, y_test, name):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        result = {
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "auc_roc": roc_auc_score(y_test, probs),
            "auc_pr": average_precision_score(y_test, probs),
            "preds": preds, "probs": probs, "labels": y_test,
        }
        print(f"{name} | Macro F1: {result['macro_f1']:.4f}")
        return result

    classical_results = {}
    for seed in SEEDS:
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        lr.fit(X_train_tfidf, y_train)
        classical_results.setdefault("Logistic Regression", []).append(
            eval_sklearn_model(lr, X_test_tfidf, y_test, "Logistic Regression"))

        svm = SVC(class_weight="balanced", probability=True, kernel="linear", random_state=seed)
        svm.fit(X_train_tfidf, y_train)
        classical_results.setdefault("SVM", []).append(
            eval_sklearn_model(svm, X_test_tfidf, y_test, "SVM"))

        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train, sample_weight=sample_weight_train)
        classical_results.setdefault("Naive Bayes", []).append(
            eval_sklearn_model(nb, X_test_tfidf, y_test, "Naive Bayes"))

        rf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=seed)
        rf.fit(X_train_tfidf, y_train)
        classical_results.setdefault("Random Forest", []).append(
            eval_sklearn_model(rf, X_test_tfidf, y_test, "Random Forest"))

        xgb = XGBClassifier(eval_metric="logloss", random_state=seed)
        xgb.fit(X_train_tfidf, y_train, sample_weight=sample_weight_train)
        classical_results.setdefault("XGBoost", []).append(
            eval_sklearn_model(xgb, X_test_tfidf, y_test, "XGBoost"))

    # BiLSTM-Attention
    vocab = build_vocab(train_df["combined_text"])
    collate_fn = make_collate_fn(vocab)
    bilstm_results = []
    for seed in SEEDS:
        set_seed(seed)
        train_ds = BiLSTMDataset(train_df["combined_text"], train_df["label"], vocab)
        val_ds = BiLSTMDataset(val_df["combined_text"], val_df["label"], vocab)
        test_ds = BiLSTMDataset(test_df["combined_text"], test_df["label"], vocab)

        bilstm_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        bilstm_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        bilstm_test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        model = BiLSTMAttention(vocab_size=len(vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        best_f1, best_state, patience_counter = 0, None, 0
        for epoch in range(15):
            model.train()
            for x, lengths, y in bilstm_train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fct(model(x, lengths), y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x, lengths, y in bilstm_val_loader:
                    preds = torch.argmax(model(x.to(device), lengths), dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.numpy())
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            if val_f1 > best_f1:
                best_f1, best_state, patience_counter = val_f1, copy.deepcopy(model.state_dict()), 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    break

        model.load_state_dict(best_state)
        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for x, lengths, y in bilstm_test_loader:
                logits = model(x.to(device), lengths)
                probs = torch.softmax(logits, dim=1)
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels.extend(y.numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())

        result = {
            "macro_f1": f1_score(test_labels, test_preds, average="macro"),
            "auc_roc": roc_auc_score(test_labels, test_probs),
            "auc_pr": average_precision_score(test_labels, test_probs),
            "preds": test_preds, "probs": test_probs, "labels": test_labels,
        }
        print(f"BiLSTM-Attention | seed {seed} | Macro F1: {result['macro_f1']:.4f}")
        bilstm_results.append(result)

    classical_results["BiLSTM-Attention"] = bilstm_results

    print("\nPipeline complete.")
    return all_results, all_models, classical_results, train_df, val_df, test_df


if __name__ == "__main__":
    main()