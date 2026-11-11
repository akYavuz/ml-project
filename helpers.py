# helpers.py - shared functions used by both notebooks
import os
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)

SEEDS = [42, 52, 62]

PARAM_ORDER = {"dt": ["depth2", "depth3", "depth5", "depth10", "depthNone"], # max_depth: how many questions the tree can ask in a row, low = underfitting, high = overfitting
               "knn": ["k1", "k3", "k5", "k7", "k11"]} # k: how many nearest neighbors we look at, small k = overfitting, usually odd numbers so the votes dont tie


# Source: assignment 4.1, slide "validation methods" p.3 (train_test_split)
def make_splits(X, y):
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42) # we use stratify=y because our dataset is imbalanced so we keep the class distribution in the train 
    splits = {}                                                                                                  # and test sets similar to the original dataset
    for seed in SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=0.125, stratify=y_rest, random_state=seed) #  we use 0.125 to get 10% (0.125 * 0.8 = 0.1)
        splits[seed] = {"X_train": X_train, "X_val": X_val,
                        "y_train": y_train, "y_val": y_val}
    return splits, X_test, y_test


def split_summary(splits, y_test, y):
    rows = []
    for seed in SEEDS:
        parts = {"train": splits[seed]["y_train"], "val": splits[seed]["y_val"],
                 "test": y_test}
        for name, y_part in parts.items():
            row = {"seed": seed, "subset": name, "n": len(y_part),
                   "percent": round(100 * len(y_part) / len(y), 1)}
            class_counts = y_part.value_counts().sort_index().to_dict()
            row.update({f"class_{c}": n for c, n in class_counts.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def check_splits(splits, X_test, X):
    all_idx = set(X.index)
    test_idx = set(X_test.index)
    for seed in SEEDS:
        train_idx = set(splits[seed]["X_train"].index)
        val_idx = set(splits[seed]["X_val"].index)
        no_overlap = not (train_idx & val_idx or train_idx & test_idx or val_idx & test_idx)
        sizes_ok = len(train_idx) + len(val_idx) + len(test_idx) == len(all_idx)
        same_test = (all_idx - train_idx - val_idx) == test_idx          # so we never touch the test set during training/validation
        print(f"seed {seed}: no overlap = {no_overlap}, sizes add up = {sizes_ok}, "
              f"same test set = {same_test}")


# Source: assignment 7.1
def get_configs(seed):
    configs = []
    for depth in [2, 3, 5, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        configs.append(("dt", "depth" + str(depth), model))
    for k in [1, 3, 5, 7, 11]:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
        configs.append(("knn", "k" + str(k), model))
    configs.append(("nb", "default", GaussianNB()))
    return configs


def get_versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def plot_param_comparison(summary, dataset_name, algorithm, save_path):
    order = PARAM_ORDER[algorithm]
    colors = {"unscaled": "tab:blue", "scaled": "tab:orange"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for representation in ["unscaled", "scaled"]:
        part = summary[(summary["dataset"] == dataset_name)
                       & (summary["algorithm"] == algorithm)
                       & (summary["representation"] == representation)]
        part = part.set_index("parameter").loc[order]
        x = range(len(order))
        ax.errorbar(x, part["train_macro_f1_mean"], yerr=part["train_macro_f1_std"],
                    color=colors[representation], linestyle="--", marker="o",
                    capsize=3, label=f"train ({representation})")
        ax.errorbar(x, part["val_macro_f1_mean"], yerr=part["val_macro_f1_std"],
                    color=colors[representation], linestyle="-", marker="s",
                    capsize=3, label=f"validation ({representation})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([p.replace("depth", "").replace("k", "") for p in order])
    ax.set_xlabel("max_depth" if algorithm == "dt" else "n_neighbors (k)")
    ax.set_ylabel("Mean macro-F1 (error bars = std over 3 seeds)")
    ax.set_title(f"{dataset_name}: {'Decision Tree' if algorithm == 'dt' else 'KNN'}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()


# Source: slide "validation methods" p.25-26 (confusion_matrix + heatmap)
def plot_confusion(y_test, pred, target_names, title, save_path):
    cm = confusion_matrix(y_test, pred, labels=sorted(y_test.unique()))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    return cm


# class_pos = row/column of the chosen class in cm
def manual_check(cm, class_pos, class_name, report_dict):
    TP = cm[class_pos, class_pos]
    FN = cm[class_pos, :].sum() - TP
    FP = cm[:, class_pos].sum() - TP
    TN = cm.sum() - TP - FP - FN
    print(f"{class_name}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan     # nan = undefined
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    report = report_dict[class_name]
    return pd.DataFrame({"manual": [precision, recall, f1],
                         "classification_report": [report["precision"], report["recall"],
                                                   report["f1-score"]]},
                        index=["precision", "recall", "f1"])


def plot_roc_breast_cancer(model, X, y_test, save_path):
    proba = model.predict_proba(X)
    column = list(model.classes_).index(0)        # probability of malignant (class 0)
    y_malignant = (y_test == 0).astype(int)
    fpr, tpr, _ = roc_curve(y_malignant, proba[:, column])
    auc = roc_auc_score(y_malignant, proba[:, column])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"malignant vs benign (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Breast Cancer - test ROC (malignant = positive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    return auc


def plot_roc_wine(model, X, y_test, target_names, save_path):
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    class_aucs = {}
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        class_aucs[target_names[i]] = roc_auc_score(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{target_names[i]} vs rest (AUC = {class_aucs[target_names[i]]:.3f})")
    macro_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Wine - test ROC (macro OvR AUC = {macro_auc:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    return class_aucs, macro_auc


def list_errors(y_test, pred, target_names):
    names = dict(enumerate(target_names))
    errors = pd.DataFrame({"actual": y_test, "predicted": pred}, index=y_test.index)
    errors = errors[errors["actual"] != errors["predicted"]]
    return errors.replace(names)


# every feature gets its own small boxplot because the scales are very different
def draw_boxplots(X, per_figure, prefix, title):
    columns = list(X.columns)
    for number, start in enumerate(range(0, len(columns), per_figure), start=1):
        group = columns[start:start + per_figure]
        n_rows = (len(group) + 4) // 5
        fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, column in zip(axes, group):
            ax.boxplot(X[column])
            ax.set_title(column, fontsize=8)
            ax.set_xticks([])
        for ax in axes[len(group):]:
            ax.axis("off")
        fig.suptitle(f"{title} - boxplot {number}")
        fig.tight_layout()
        fig.savefig(f"{prefix}_boxplot_{number}.png", dpi=150)
        plt.show()


def plot_cluster_pair(X_part, y_test, clusters, feature1, feature2, names, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    class_names = y_test.map(dict(enumerate(names)))
    cluster_names = ["cluster " + str(c) for c in clusters]
    sns.scatterplot(x=X_part[feature1].values, y=X_part[feature2].values,
                    hue=class_names.values, ax=axes[0])
    axes[0].set_title("True classes")
    sns.scatterplot(x=X_part[feature1].values, y=X_part[feature2].values,
                    hue=cluster_names, ax=axes[1])
    axes[1].set_title("Predicted clusters")
    for ax in axes:
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()


RESULTS_DIR = "../results"
CSV_ROWS = {"classifier_runs": 132, "classifier_summary": 44, "classifier_test": 2,
            "kmeans_runs": 12, "kmeans_summary": 4, "kmeans_test": 4}
DATASETS = ["breast_cancer", "wine"]


def save_part(table, name, dataset):
    os.makedirs(RESULTS_DIR + "/parts", exist_ok=True)
    table.to_csv(f"{RESULTS_DIR}/parts/{dataset}_{name}.csv", index=False)


def combine_results():
    for name, expected in CSV_ROWS.items():
        parts = []
        for dataset in DATASETS:
            path = f"{RESULTS_DIR}/parts/{dataset}_{name}.csv"
            if os.path.exists(path):
                parts.append(pd.read_csv(path))
        if len(parts) < 2:
            print(f"{name}: waiting for the other notebook")
            continue
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(f"{RESULTS_DIR}/{name}.csv", index=False)
        print(f"{name}.csv: {len(combined)} rows (expected {expected})")