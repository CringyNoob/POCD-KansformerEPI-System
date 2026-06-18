"""
Comprehensive metrics utilities.

Computes the standard evaluation metrics used across all stages:
  BCE, MSE, Frobenius, F1, Accuracy, Balanced Accuracy,
  Precision, Recall, AUROC, AUPR, MCC

Usage:
    from src.metrics import compute_all_metrics, format_metrics_report
"""

import numpy as np

NOT_AVAILABLE_MSG = "Score not available, as the model is not designed to produce this metric."


def _safe(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return None on any error."""
    try:
        val = fn(*args, **kwargs)
        if isinstance(val, float) and np.isnan(val):
            return None
        return val
    except Exception:
        return None


def compute_all_metrics(
    labels,
    probs,
    preds=None,
    bce_loss_val=None,
    mse_loss_val=None,
    frob_loss_val=None,
    is_multilabel=False,
):
    """
    Compute all 11 metrics from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    labels : array-like, shape (N,)
        Ground-truth binary labels (0 or 1).
    probs : array-like, shape (N,)
        Predicted probabilities (after sigmoid).
    preds : array-like, shape (N,), optional
        Binary predictions (>= 0.5). Computed from probs if not given.
    bce_loss_val : float, optional
        Pre-computed BCE loss value.
    mse_loss_val : float, optional
        Pre-computed MSE loss value.
    frob_loss_val : float, optional
        Pre-computed Frobenius penalty value.
    is_multilabel : bool
        If True, binary classification metrics (F1, Acc, BalAcc, Precision,
        Recall, MCC) are marked as not available.

    Returns
    -------
    dict with keys:
        bce, mse, frobenius, f1, accuracy, balanced_accuracy,
        precision, recall, auroc, aupr, mcc
        Values are float or None (if not applicable).
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )

    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()
    if preds is None:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.asarray(preds).ravel().astype(int)
    labels_int = labels.astype(int)

    result = {
        "bce": bce_loss_val,
        "mse": mse_loss_val,
        "frobenius": frob_loss_val,
    }

    # Ranking metrics (always applicable for binary or flattened multi-label)
    result["auroc"] = _safe(roc_auc_score, labels_int, probs)
    result["aupr"] = _safe(average_precision_score, labels_int, probs)

    if is_multilabel:
        # Binary classification metrics don't apply to multi-label
        for key in ("f1", "accuracy", "balanced_accuracy", "precision", "recall", "mcc"):
            result[key] = None
    else:
        result["f1"] = _safe(f1_score, labels_int, preds, zero_division=0)
        result["accuracy"] = _safe(accuracy_score, labels_int, preds)
        result["balanced_accuracy"] = _safe(balanced_accuracy_score, labels_int, preds)
        result["precision"] = _safe(precision_score, labels_int, preds, zero_division=0)
        result["recall"] = _safe(recall_score, labels_int, preds, zero_division=0)
        result["mcc"] = _safe(matthews_corrcoef, labels_int, preds)

    return result


def format_metrics_report(m, prefix="", indent=2):
    """
    Format a metrics dict as a readable multi-line string.

    Parameters
    ----------
    m : dict
        Output of compute_all_metrics().
    prefix : str
        Optional prefix like "[Train]" or "[Val]".
    indent : int
        Number of leading spaces.

    Returns
    -------
    str
    """
    pad = " " * indent
    pfx = f"{prefix} " if prefix else ""
    lines = []

    def _fmt(key, label):
        val = m.get(key)
        if val is None:
            lines.append(f"{pad}{pfx}{label:20s}: {NOT_AVAILABLE_MSG}")
        else:
            lines.append(f"{pad}{pfx}{label:20s}: {val:.4f}")

    _fmt("bce", "BCE Loss")
    _fmt("mse", "MSE Loss")
    _fmt("frobenius", "Frobenius Loss")
    _fmt("auroc", "AUROC")
    _fmt("aupr", "AUPR")
    _fmt("accuracy", "Accuracy")
    _fmt("balanced_accuracy", "Balanced Accuracy")
    _fmt("precision", "Precision")
    _fmt("recall", "Recall")
    _fmt("f1", "F1 Score")
    _fmt("mcc", "MCC")

    return "\n".join(lines)


def format_epoch_line(epoch, max_epochs, train_m, val_m, elapsed):
    """
    Format a compact single-line epoch summary with key metrics.

    Parameters
    ----------
    epoch : int
    max_epochs : int
    train_m : dict from compute_all_metrics
    val_m : dict from compute_all_metrics
    elapsed : float, seconds

    Returns
    -------
    str
    """

    def _v(m, key):
        v = m.get(key)
        return f"{v:.4f}" if v is not None else "N/A"

    return (
        f"Epoch {epoch:3d}/{max_epochs} | "
        f"AUROC {_v(train_m,'auroc')}/{_v(val_m,'auroc')} | "
        f"AUPR {_v(train_m,'aupr')}/{_v(val_m,'aupr')} | "
        f"Acc {_v(train_m,'accuracy')}/{_v(val_m,'accuracy')} | "
        f"F1 {_v(train_m,'f1')}/{_v(val_m,'f1')} | "
        f"MCC {_v(train_m,'mcc')}/{_v(val_m,'mcc')} | "
        f"BalAcc {_v(train_m,'balanced_accuracy')}/{_v(val_m,'balanced_accuracy')} | "
        f"{elapsed:.0f}s"
    )


def metrics_to_history_row(epoch, train_m, val_m, elapsed, test_m=None):
    """
    Build a history row dict containing all metrics for JSON serialization.
    """
    row = {"epoch": epoch, "time": round(elapsed, 1)}
    for prefix, m in [("train", train_m), ("val", val_m)]:
        for key in ("bce", "mse", "frobenius", "auroc", "aupr", "accuracy",
                     "balanced_accuracy", "precision", "recall", "f1", "mcc"):
            row[f"{prefix}_{key}"] = m.get(key)

    if test_m is not None:
        for key in ("bce", "mse", "frobenius", "auroc", "aupr", "accuracy",
                     "balanced_accuracy", "precision", "recall", "f1", "mcc"):
            row[f"test_{key}"] = test_m.get(key)

    return row
