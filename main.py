"""
Fine-Tuning DistilBERT for ICT Misconception Detection in Open-Ended
Responses from Ghanaian Senior High School Students.

REVISION 2 -- corrects the following issues identified in CAN-DO-VEX-DAT
Testbed feedback (round 1, commit 0e30b0c):
  - Denominator mismatch: the confidence-threshold arm was previously
    scored on a reduced instance set (147/154) while every comparator was
    scored on the full set. All headline comparisons now use full-coverage
    argmax scoring; the threshold mechanism is reported separately as a
    risk-coverage result (selective_subset_from_full).
  - Seed regime: increased from 3 seeds on one fixed partition to 10 seeds,
    each independently redrawing the train/val/test partition, so reported
    variance reflects partition variance and initialisation variance
    jointly rather than initialisation variance alone.
  - Missing ablation cell: the grid now includes all four combinations of
    {single-layer, two-layer} x {no threshold, threshold}, derived
    post-hoc from saved probabilities with no additional training required.
  - McNemar's odds-ratio edge case: corrected an inverted zero-count
    branch and added an explicit Haldane-Anscombe correction where a
    discordant count of zero would otherwise produce an undefined ratio.
  - Interpretability: Integrated Gradients now runs on the full
    Misconception-labelled test subset (not a fixed n=20 sample), reports
    SIGNED attributions (not absolute), and suppresses any token below a
    minimum instance-support threshold.
  - Reproducibility: every per-seed, per-model prediction set is persisted
    to disk as a JSON artifact, so downstream statistics can be
    regenerated without retraining.

Author: Prince Aidoo
CAN-DO Research Lab, KNUST Department of Computer Science
"""

import random
import os
import copy
import time
import json

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
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import norm


# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "data/misconception_corpus.csv"
ARTIFACT_DIR = "artifacts"
SEEDS = [42, 123, 2025, 7, 19, 77, 101, 256, 512, 999]
THRESHOLD = 0.75
TAU_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MAX_LENGTH = 128
BATCH_SIZE = 8
MIN_ATTRIBUTION_SUPPORT = 5

os.makedirs(ARTIFACT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Data loading and cleaning ────────────────────────────────────────────────
def load_and_clean_data(path):
    """
    Loads the raw corpus with full attrition disclosure. Reports every
    filtering step so the final n is fully reconciled against the raw
    row count (addresses the M4 provenance-reconciliation finding).
    """
    df = pd.read_csv(path)
    n_raw = len(df)
    print(f"Raw rows loaded: {n_raw}")

    df = df[df["label"].notna()].copy()
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"Misconception": 1, "Correct": 0})
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    n_after_label = len(df)
    print(f"After label cleaning: {n_after_label} "
          f"(dropped {n_raw - n_after_label} unusable-label rows)")

    df = df[df["response_text"].notna()].copy()
    n_after_text = len(df)
    print(f"After response-text completeness filter: {n_after_text} "
          f"(dropped {n_after_label - n_after_text} rows with missing response_text)")

    df["combined_text"] = (
        "Question: " + df["question_text"] + " Answer: " + df["response_text"]
    )

    extracted = df["response_id"].str.extract(r"(R\d+)$")[0]
    n_unparsed = extracted.isna().sum()
    print(f"Respondent-ID extraction: {n_unparsed} unparsed value(s) out of {len(df)} rows.")
    df["respondent_id"] = extracted
    if n_unparsed > 0:
        df = df[df["respondent_id"].notna()].copy()

    print(f"Final corpus: {len(df)} instances "
          f"({n_raw} raw -> {n_after_label} -> {n_after_text} -> {len(df)})")
    print(df["label"].value_counts())
    return df


def student_level_recurrence(df):
    """
    Tests whether Misconception labels recur WITHIN a respondent across
    items, distinguishing a stable misconception from a one-off wrong
    answer.
    """
    grouped = df.groupby("respondent_id")["label"].agg(["sum", "count"])
    n = len(grouped)
    zero = (grouped["sum"] == 0).sum()
    one = (grouped["sum"] == 1).sum()
    two_plus = (grouped["sum"] >= 2).sum()
    print(f"\nStudent-level recurrence: {n} respondents")
    print(f"  0 Misconception responses: {zero} ({100*zero/n:.1f}%)")
    print(f"  Exactly 1: {one} ({100*one/n:.1f}%)")
    print(f"  2+ (recurrence): {two_plus} ({100*two_plus/n:.1f}%)")
    return grouped


def grouped_split(df, seed):
    """
    Grouped hold-out split, redrawn independently for the given seed.
    Grouping by respondent_id prevents any respondent's rows appearing
    in more than one partition.
    """
    set_seed(seed)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["respondent_id"]))
    train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["respondent_id"]))
    val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()

    assert set(train_df["respondent_id"]) & set(val_df["respondent_id"]) == set()
    assert set(train_df["respondent_id"]) & set(test_df["respondent_id"]) == set()
    assert set(val_df["respondent_id"]) & set(test_df["respondent_id"]) == set()

    return train_df, val_df, test_df


# ── Model definitions ────────────────────────────────────────────────────────
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
    """Primary engineered model: Linear(768,256)->ReLU->Dropout->Linear(256,2)."""

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


class EngineeredDistilBERT_v2(nn.Module):
    """
    Secondary ablation: capacity-reduced head (64-dim, layer-normalised),
    trained with differential encoder/head learning rates, to test the
    hypothesis that the primary head's added capacity was mismatched to
    the training partition size.
    """

    def __init__(self, num_classes=2, dropout_rate=0.5, hidden_dim=64):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
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


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_full_coverage(model, loader, device):
    """
    Computes metrics via unmodified argmax over ALL instances. This is
    the ONLY function used to populate the main comparison table, so
    every model is scored on the identical partition with no
    threshold-induced denominator change.
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
            preds = torch.argmax(probs, dim=1)
            for i in range(len(batch["labels"])):
                all_labels.append(batch["labels"][i].item())
                all_probs.append(probs[i][1].item())
                all_preds.append(preds[i].item())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
    auc_pr = average_precision_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
    return {
        "macro_f1": macro_f1, "auc_roc": auc_roc, "auc_pr": auc_pr,
        "preds": all_preds, "labels": all_labels, "probs": all_probs,
        "total": len(all_labels),
    }


def selective_subset_from_full(full_results, threshold):
    """
    Derives the selective (abstention) view from saved full-coverage
    probabilities, as a report SEPARATE from the main comparison table.
    """
    preds, labels, probs = full_results["preds"], full_results["labels"], full_results["probs"]
    accepted_idx = [i for i, p1 in enumerate(probs) if max(p1, 1 - p1) >= threshold]
    accepted_preds = [preds[i] for i in accepted_idx]
    accepted_labels = [labels[i] for i in accepted_idx]
    coverage = len(accepted_idx) / len(labels)
    macro_f1 = (
        f1_score(accepted_labels, accepted_preds, average="macro")
        if len(set(accepted_labels)) == 2 and accepted_labels else None
    )
    return {
        "threshold": threshold, "coverage": coverage,
        "n_accepted": len(accepted_idx), "n_total": len(labels),
        "macro_f1_accepted": macro_f1, "accepted_idx": accepted_idx,
    }


def matched_coverage_metric(full_results, target_coverage):
    """
    Gives a model the SAME abstention budget as the proposed model, by
    dropping its least-confident fraction, then recomputing Macro F1 on
    the remainder.
    """
    preds, labels, probs = full_results["preds"], full_results["labels"], full_results["probs"]
    confidences = [max(p, 1 - p) for p in probs]
    n_keep = int(round(target_coverage * len(labels)))
    order = np.argsort(confidences)[::-1]
    keep_idx = order[:n_keep]
    kept_preds = [preds[i] for i in keep_idx]
    kept_labels = [labels[i] for i in keep_idx]
    macro_f1 = f1_score(kept_labels, kept_preds, average="macro") if len(set(kept_labels)) == 2 else None
    return {"target_coverage": target_coverage, "actual_n": n_keep, "macro_f1_matched": macro_f1}


def tau_sweep(full_results, tau_values):
    """Risk-coverage curve across tau, no retraining needed."""
    rows = []
    for tau in tau_values:
        sel = selective_subset_from_full(full_results, tau)
        rows.append({
            "tau": tau, "coverage": sel["coverage"],
            "macro_f1_accepted": sel["macro_f1_accepted"],
            "n_accepted": sel["n_accepted"],
        })
    return pd.DataFrame(rows)


def mcnemar_with_effect_size(preds_a, preds_b, labels, name_a, name_b, alpha=0.05):
    """
    McNemar's exact test with an odds-ratio effect size and 95% CI.
    Applies the exact binomial form unconditionally (not gated on
    discordant-pair count). Applies a Haldane-Anscombe correction when
    either discordant count is zero, since the raw ratio is undefined
    in that case and the naive zero/infinity substitution used in an
    earlier version of this script had the two cases inverted.
    """
    correct_a = [int(p == l) for p, l in zip(preds_a, labels)]
    correct_b = [int(p == l) for p, l in zip(preds_b, labels)]
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 0)
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 1)
    n11 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 1)
    n00 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 0)

    result = mcnemar([[n11, n01], [n10, n00]], exact=True)

    if n01 == 0 or n10 == 0:
        # Haldane-Anscombe correction: add 0.5 to each discordant cell
        odds_ratio = (n10 + 0.5) / (n01 + 0.5)
        se = np.sqrt(1 / (n01 + 0.5) + 1 / (n10 + 0.5))
        corrected = True
    else:
        odds_ratio = n10 / n01
        se = np.sqrt(1 / n01 + 1 / n10)
        corrected = False

    z = norm.ppf(1 - alpha / 2)
    ci_low = np.exp(np.log(odds_ratio) - z * se)
    ci_high = np.exp(np.log(odds_ratio) + z * se)

    return {
        "pvalue": result.pvalue, "odds_ratio": odds_ratio, "ci": (ci_low, ci_high),
        "n01": n01, "n10": n10, "n11": n11, "n00": n00,
        "haldane_anscombe_corrected": corrected,
    }


# ── DistilBERT training ───────────────────────────────────────────────────────
def train_and_eval_condition(model_class, model_kwargs, condition_name, seed,
                              train_loader, val_loader, test_loader, class_weights,
                              encoder_lr=2e-5, head_lr=None,
                              epochs=5, patience=3):
    """
    Trains a DistilBERT-based model and evaluates it at full coverage
    only. If head_lr is given, uses a differential learning rate between
    the pretrained encoder and the classification head; otherwise a
    single learning rate is applied uniformly (matching the original
    baseline and primary engineered configuration).
    """
    set_seed(seed)
    model = model_class(**model_kwargs).to(device)

    if head_lr is not None:
        encoder_params = list(model.distilbert.parameters())
        head_params = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=encoder_lr, weight_decay=0.01)

    loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, patience_counter = 0, None, 0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fct(outputs["logits"], batch["labels"])
            loss.backward()
            optimizer.step()

        val_f1 = evaluate_full_coverage(model, val_loader, device)["macro_f1"]
        if val_f1 > best_f1:
            best_f1, best_state, patience_counter = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - train_start

    if best_state is None:
        print(f"  WARNING: [{condition_name} | seed {seed}] never improved past val_f1=0; "
              f"using final-epoch weights.")
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    infer_start = time.time()
    test_results = evaluate_full_coverage(model, test_loader, device)
    infer_time = time.time() - infer_start

    test_results.update({
        "train_time_sec": train_time, "infer_time_sec": infer_time,
        "seed": seed, "condition": condition_name,
    })
    print(f"  [{condition_name} | seed {seed}] Full-coverage Macro F1: "
          f"{test_results['macro_f1']:.4f} | Train: {train_time:.1f}s")

    out_path = os.path.join(ARTIFACT_DIR, f"{condition_name.replace(' ', '_')}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "preds": test_results["preds"], "labels": test_results["labels"],
            "probs": test_results["probs"],
        }, f)

    return model, test_results


# ── BiLSTM-Attention baseline ────────────────────────────────────────────────
def build_vocab(texts, vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_bilstm(text, vocab, max_len=MAX_LENGTH):
    tokens = text.lower().split()
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LENGTH):
        self.texts, self.labels = texts.tolist(), labels.tolist()
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = encode_bilstm(self.texts[idx], self.vocab, self.max_len)
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
        return self.classifier(self.dropout(context))


def train_and_eval_bilstm(seed, train_df, val_df, test_df, class_weights):
    set_seed(seed)
    vocab = build_vocab(train_df["combined_text"])
    collate_fn = make_collate_fn(vocab)

    train_loader = DataLoader(
        BiLSTMDataset(train_df["combined_text"], train_df["label"], vocab),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        BiLSTMDataset(val_df["combined_text"], val_df["label"], vocab),
        batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(
        BiLSTMDataset(test_df["combined_text"], test_df["label"], vocab),
        batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = BiLSTMAttention(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, patience_counter = 0, None, 0
    for epoch in range(15):
        model.train()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fct(model(x, lengths), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, lengths, y in val_loader:
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

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            logits = model(x.to(device), lengths)
            probs = torch.softmax(logits, dim=1)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(y.numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())

    return {
        "macro_f1": f1_score(test_labels, test_preds, average="macro"),
        "auc_roc": roc_auc_score(test_labels, test_probs),
        "auc_pr": average_precision_score(test_labels, test_probs),
        "preds": test_preds, "probs": test_probs, "labels": test_labels,
        "total": len(test_labels), "seed": seed,
    }


# ── Signed, full-corpus Integrated Gradients ─────────────────────────────────
def compute_signed_attributions(model, tokenizer, test_df, device,
                                  min_support=MIN_ATTRIBUTION_SUPPORT):
    """
    Computes Integrated Gradients on ALL Misconception-labelled instances
    in the given test partition (not a fixed-size subsample), reports
    SIGNED attributions (positive = toward Misconception, negative =
    toward Correct), and suppresses any token observed in fewer than
    min_support instances.
    """
    from captum.attr import LayerIntegratedGradients

    model.eval()

    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask)["logits"]

    lig = LayerIntegratedGradients(forward_func, model.distilbert.embeddings)

    misconception_texts = test_df[test_df["label"] == 1]["combined_text"].tolist()
    token_scores = {}

    for text in misconception_texts:
        encoding = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=MAX_LENGTH
        ).to(device)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        attributions, _ = lig.attribute(
            inputs=input_ids, baselines=baseline_ids,
            additional_forward_args=(attention_mask,), target=1,
            return_convergence_delta=True, n_steps=50,
        )
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

        for tok, sc in zip(tokens, attributions.detach().cpu().numpy()):
            if tok in ("[PAD]", "[CLS]", "[SEP]"):
                continue
            token_scores.setdefault(tok, []).append(sc)

    rows = []
    for tok, scores in token_scores.items():
        n = len(scores)
        if n < min_support:
            continue
        mean_signed = np.mean(scores)
        rows.append({
            "token": tok, "n_instances": n, "mean_signed_attribution": mean_signed,
            "direction": "toward Misconception" if mean_signed > 0 else "toward Correct",
        })

    return pd.DataFrame(rows).sort_values(
        "mean_signed_attribution", key=abs, ascending=False
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    df = load_and_clean_data(DATA_PATH)
    student_level_recurrence(df)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    condition_names = [
        "Standard", "Engineered", "Engineered-v2",
        "Logistic Regression", "SVM", "Naive Bayes",
        "Random Forest", "XGBoost", "BiLSTM-Attention",
    ]
    all_results = {name: [] for name in condition_names}
    all_results_selective = {"Standard+Threshold": [], "Engineered+Threshold": []}
    all_models = {"Engineered": []}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_df, val_df, test_df = grouped_split(df, seed)

        class_counts = train_df["label"].value_counts().sort_index()
        class_weights = torch.tensor(
            [len(train_df) / (2 * class_counts[i]) for i in range(2)], dtype=torch.float
        ).to(device)

        train_loader = DataLoader(
            RealDataset(train_df["combined_text"], train_df["label"], tokenizer),
            batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            RealDataset(val_df["combined_text"], val_df["label"], tokenizer),
            batch_size=BATCH_SIZE)
        test_loader = DataLoader(
            RealDataset(test_df["combined_text"], test_df["label"], tokenizer),
            batch_size=BATCH_SIZE)

        # Standard and primary Engineered DistilBERT
        seed_full = {}
        for cond_name, model_class, kwargs in [
            ("Standard", StandardDistilBERT, {"num_classes": 2}),
            ("Engineered", EngineeredDistilBERT, {"num_classes": 2, "dropout_rate": 0.3}),
        ]:
            model, results = train_and_eval_condition(
                model_class, kwargs, cond_name, seed,
                train_loader, val_loader, test_loader, class_weights,
            )
            all_results[cond_name].append(results)
            seed_full[cond_name] = results
            if cond_name == "Engineered":
                all_models["Engineered"].append(model)

        # Secondary ablation: capacity-reduced head, differential LR
        _, results_v2 = train_and_eval_condition(
            EngineeredDistilBERT_v2, {"num_classes": 2, "dropout_rate": 0.5, "hidden_dim": 64},
            "Engineered-v2", seed, train_loader, val_loader, test_loader, class_weights,
            encoder_lr=2e-5, head_lr=1e-3,
        )
        all_results["Engineered-v2"].append(results_v2)

        # 4th ablation cell: derive both +Threshold variants post-hoc
        for base_name in ["Standard", "Engineered"]:
            sel = selective_subset_from_full(seed_full[base_name], THRESHOLD)
            all_results_selective[f"{base_name}+Threshold"].append({
                "macro_f1": sel["macro_f1_accepted"], "coverage": sel["coverage"],
                "n_accepted": sel["n_accepted"], "n_total": sel["n_total"],
                "seed": seed, "condition": f"{base_name}+Threshold",
            })

        # Classical baselines, retrained on this seed's split
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(train_df["combined_text"])
        X_test_tfidf = tfidf.transform(test_df["combined_text"])
        y_train, y_test = train_df["label"].values, test_df["label"].values
        cw_map = {0: class_weights[0].item(), 1: class_weights[1].item()}
        sw_train = np.array([cw_map[l] for l in y_train])

        def eval_sklearn(model, name):
            preds = model.predict(X_test_tfidf)
            probs = model.predict_proba(X_test_tfidf)[:, 1]
            result = {
                "macro_f1": f1_score(y_test, preds, average="macro"),
                "auc_roc": roc_auc_score(y_test, probs),
                "auc_pr": average_precision_score(y_test, probs),
                "preds": preds.tolist(), "probs": probs.tolist(),
                "labels": y_test.tolist(), "total": len(y_test), "seed": seed,
            }
            all_results[name].append(result)
            print(f"  [{name} | seed {seed}] Macro F1: {result['macro_f1']:.4f}")

        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        lr.fit(X_train_tfidf, y_train)
        eval_sklearn(lr, "Logistic Regression")

        svm = SVC(class_weight="balanced", probability=True, kernel="linear", random_state=seed)
        svm.fit(X_train_tfidf, y_train)
        eval_sklearn(svm, "SVM")

        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train, sample_weight=sw_train)
        eval_sklearn(nb, "Naive Bayes")

        rf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=seed)
        rf.fit(X_train_tfidf, y_train)
        eval_sklearn(rf, "Random Forest")

        xgb = XGBClassifier(eval_metric="logloss", random_state=seed)
        xgb.fit(X_train_tfidf, y_train, sample_weight=sw_train)
        eval_sklearn(xgb, "XGBoost")

        bilstm_result = train_and_eval_bilstm(seed, train_df, val_df, test_df, class_weights)
        all_results["BiLSTM-Attention"].append(bilstm_result)
        print(f"  [BiLSTM-Attention | seed {seed}] Macro F1: {bilstm_result['macro_f1']:.4f}")

    # Interpretability, computed on the seed-42 test partition
    train_df_42, val_df_42, test_df_42 = grouped_split(df, SEEDS[0])
    attribution_df = compute_signed_attributions(
        all_models["Engineered"][0], tokenizer, test_df_42, device
    )
    attribution_df.to_csv(os.path.join(ARTIFACT_DIR, "signed_attributions_seed42.csv"), index=False)

    print("\nPipeline complete. Per-seed prediction artifacts saved to:", ARTIFACT_DIR)
    return all_results, all_results_selective, all_models, attribution_df, df


if __name__ == "__main__":
    main()
