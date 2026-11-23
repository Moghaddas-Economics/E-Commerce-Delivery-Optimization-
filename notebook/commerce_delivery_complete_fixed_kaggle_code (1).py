# ============================================================
# Commerce Delivery Optimization using Predictive & Prescriptive Analytics
# COMPLETE KAGGLE CODE
# Data Mining + Data Cleaning + Modeling + Tables + PDF Figures + Workflow
# Dataset: Customer Analytics by prachi13
# Expected file: /kaggle/input/datasets/prachi13/customer-analytics/Train.csv
# ============================================================

# ============================================================
# SECTION 0 - LIBRARIES AND SETTINGS
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ============================================================
# SECTION 0.1 - OUTPUT FOLDERS
# ============================================================

BASE_DIR = "/kaggle/working/commerce_delivery_outputs"

FIG_DIR = os.path.join(BASE_DIR, "Figures")
TABLE_DIR = os.path.join(BASE_DIR, "Tables")
RESULT_DIR = os.path.join(BASE_DIR, "Results")
CODE_DIR = os.path.join(BASE_DIR, "Codes")
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")
GITHUB_DIR = os.path.join(BASE_DIR, "GitHub")
WORKFLOW_DIR = os.path.join(BASE_DIR, "Workflow")
SOURCE_DIR = os.path.join(BASE_DIR, "Source")

for folder in [
    BASE_DIR, FIG_DIR, TABLE_DIR, RESULT_DIR, CODE_DIR,
    DATASET_DIR, GITHUB_DIR, WORKFLOW_DIR, SOURCE_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# SECTION 0.2 - STYLE
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#607D8B"]
BLUE = "#2196F3"
ORANGE = "#FF5722"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
RED = "#E53935"
GREY = "#607D8B"


def save_pdf(fig, filename):
    """Save a matplotlib figure as PDF in the Figures folder."""
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {path}")


def save_table_pdf(df, filename, title=None, max_rows=30):
    """Save a pandas dataframe as a simple PDF table."""
    show_df = df.copy()
    if len(show_df) > max_rows:
        show_df = show_df.head(max_rows)

    fig_height = max(2.5, 0.35 * len(show_df) + 1.5)
    fig_width = min(18, max(8, 1.2 * len(show_df.columns)))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#37474F")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 == 0 else "#F5F5F5")

    path = os.path.join(TABLE_DIR, filename)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table PDF: {path}")


# ============================================================
# SECTION 0.3 - DATA PATH FIX
# ============================================================

DATA_PATH = "/kaggle/input/datasets/prachi13/customer-analytics/Train.csv"

if not os.path.exists(DATA_PATH):
    print("The expected path was not found:")
    print(DATA_PATH)
    print("\nSearching for CSV files inside /kaggle/input ...")
    csv_files = []
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirname, filename))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV file found in /kaggle/input. Please add the Kaggle dataset: Customer Analytics by prachi13."
        )

    print("CSV files found:")
    for f in csv_files:
        print("  ", f)

    DATA_PATH = csv_files[0]
    print("\nUsing this file automatically:")
    print(DATA_PATH)

print("Dataset path used:")
print(DATA_PATH)


# ============================================================
# SECTION 1 - DATA MINING AND VARIABLE DESCRIPTION
# ============================================================

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()

print("=" * 70)
print("COMMERCE DELIVERY OPTIMIZATION - COMPLETE REPORT")
print("=" * 70)
print(f"Raw dataset loaded: {df.shape[0]:,} rows and {df.shape[1]} columns")


# ============================================================
# TABLE 1 - DATASET OVERVIEW
# ============================================================

dataset_overview = pd.DataFrame({
    "Item": [
        "Dataset name",
        "Source",
        "File name",
        "Number of records",
        "Number of variables",
        "Target variable",
        "Project aim"
    ],
    "Value": [
        "Customer Analytics / E-Commerce Shipping Data",
        "Kaggle dataset by prachi13",
        os.path.basename(DATA_PATH),
        f"{df.shape[0]:,}",
        f"{df.shape[1]}",
        "Reached.on.Time_Y.N",
        "Predict late delivery and support delivery optimization"
    ]
})

dataset_overview.to_csv(os.path.join(TABLE_DIR, "Table_01_Dataset_Overview.csv"), index=False)
save_table_pdf(dataset_overview, "Table_01_Dataset_Overview.pdf", "Table 1. Dataset Overview")


# ============================================================
# TABLE 2 - VARIABLE DICTIONARY
# ============================================================

variable_rows = [
    ["ID", "Identifier", "Unique record number. It is removed before modeling.", "Drop"],
    ["Warehouse_block", "Nominal categorical", "Warehouse dispatch block.", "Predictor"],
    ["Mode_of_Shipment", "Nominal categorical", "Shipping mode: Ship, Flight, or Road.", "Predictor"],
    ["Customer_care_calls", "Discrete numeric", "Number of customer care calls.", "Predictor"],
    ["Customer_rating", "Ordinal numeric", "Customer rating from 1 to 5.", "Predictor"],
    ["Cost_of_the_Product", "Continuous numeric", "Product cost in monetary units.", "Predictor"],
    ["Prior_purchases", "Discrete numeric", "Number of previous purchases.", "Predictor"],
    ["Product_importance", "Ordinal categorical", "Product priority: low, medium, or high.", "Predictor"],
    ["Gender", "Nominal categorical", "Customer gender.", "Predictor"],
    ["Discount_offered", "Continuous numeric", "Discount offered to the customer.", "Predictor"],
    ["Weight_in_gms", "Continuous numeric", "Weight of the shipment in grams.", "Predictor"],
    ["Reached.on.Time_Y.N", "Binary", "1 means late delivery. 0 means on-time delivery.", "Target"],
]

variable_dictionary = pd.DataFrame(
    variable_rows,
    columns=["Variable", "Type", "Simple description", "Role"]
)

variable_dictionary.to_csv(os.path.join(TABLE_DIR, "Table_02_Variable_Dictionary.csv"), index=False)
save_table_pdf(variable_dictionary, "Table_02_Variable_Dictionary.pdf", "Table 2. Variable Dictionary")


# ============================================================
# TABLE 3 - SUMMARY STATISTICS
# ============================================================

numeric_cols_raw = df.select_dtypes(include=np.number).columns.tolist()
summary_statistics = df[numeric_cols_raw].describe().T.reset_index()
summary_statistics.rename(columns={"index": "Variable"}, inplace=True)
summary_statistics = summary_statistics.round(2)

summary_statistics.to_csv(os.path.join(TABLE_DIR, "Table_03_Summary_Statistics.csv"), index=False)
save_table_pdf(summary_statistics, "Table_03_Summary_Statistics.pdf", "Table 3. Summary Statistics")


# ============================================================
# FIGURE 1 - MISSING VALUES
# ============================================================

missing_counts = df.isnull().sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(missing_counts.index, missing_counts.values, color=BLUE, edgecolor="white")
ax.set_title("Figure 1. Missing Values Audit", fontweight="bold")
ax.set_xlabel("Dataset variables")
ax.set_ylabel("Number of missing values")
ax.tick_params(axis="x", rotation=45)
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        int(bar.get_height()),
        ha="center",
        va="bottom",
        fontsize=8
    )
plt.tight_layout()
save_pdf(fig, "Figure_01_Missing_Values_Audit.pdf")
plt.show()


# ============================================================
# FIGURE 2 - TARGET DISTRIBUTION
# ============================================================

target_raw = "Reached.on.Time_Y.N"
target_counts = df[target_raw].value_counts().sort_index()
target_labels = ["On-Time (0)", "Late (1)"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(target_labels, [target_counts.get(0, 0), target_counts.get(1, 0)],
              color=[GREEN, ORANGE], edgecolor="white")
ax.set_title("Figure 2. Delivery Status Distribution", fontweight="bold")
ax.set_xlabel("Delivery status")
ax.set_ylabel("Number of shipments")
for bar in bars:
    value = int(bar.get_height())
    pct = value / len(df) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        f"{value:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )
plt.tight_layout()
save_pdf(fig, "Figure_02_Target_Distribution.pdf")
plt.show()


# ============================================================
# FIGURE 3 - NUMERIC VARIABLE DISTRIBUTIONS
# ============================================================

numeric_for_distribution = [
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms"
]

existing_numeric = [c for c in numeric_for_distribution if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(existing_numeric):
    ax = axes[i]
    ax.hist(df[col], bins=30, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(), color=RED, linestyle="--", linewidth=1.5, label=f"Mean = {df[col].mean():.2f}")
    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend()

for j in range(len(existing_numeric), len(axes)):
    axes[j].axis("off")

fig.suptitle("Figure 3. Numeric Variable Distributions", fontsize=15, fontweight="bold")
plt.tight_layout()
save_pdf(fig, "Figure_03_Numeric_Variable_Distributions.pdf")
plt.show()


# ============================================================
# FIGURE 4 - CATEGORICAL VARIABLE DISTRIBUTIONS
# ============================================================

categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
existing_categorical = [c for c in categorical_cols if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(existing_categorical):
    ax = axes[i]
    counts = df[col].value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[:len(counts)], edgecolor="white")
    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of orders")
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=8
        )

for j in range(len(existing_categorical), len(axes)):
    axes[j].axis("off")

fig.suptitle("Figure 4. Categorical Variable Distributions", fontsize=15, fontweight="bold")
plt.tight_layout()
save_pdf(fig, "Figure_04_Categorical_Variable_Distributions.pdf")
plt.show()


# ============================================================
# FIGURE 5 - CORRELATION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[numeric_cols_raw].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Figure 5. Correlation Matrix of Numeric Variables", fontweight="bold", pad=12)
ax.set_xlabel("Numeric variables")
ax.set_ylabel("Numeric variables")
plt.tight_layout()
save_pdf(fig, "Figure_05_Correlation_Matrix.pdf")
plt.show()


# ============================================================
# SECTION 2 - DATA CLEANING AND PREPROCESSING
# ============================================================

df_clean = df.copy()

rename_map = {
    "Warehouse_block": "warehouse_block",
    "Mode_of_Shipment": "mode_of_shipment",
    "Customer_care_calls": "customer_care_calls",
    "Customer_rating": "customer_rating",
    "Cost_of_the_Product": "cost_of_product",
    "Prior_purchases": "prior_purchases",
    "Product_importance": "product_importance",
    "Gender": "gender",
    "Discount_offered": "discount_offered",
    "Weight_in_gms": "weight_in_gms",
    "Reached.on.Time_Y.N": "late_delivery"
}

if "ID" in df_clean.columns:
    df_clean.drop(columns=["ID"], inplace=True)

df_clean.rename(columns=rename_map, inplace=True)

# Missing values
for col in df_clean.select_dtypes(include=np.number).columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in df_clean.select_dtypes(include="object").columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Duplicates
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index(drop=True, inplace=True)


# ============================================================
# FIGURE 6 - OUTLIER ANALYSIS BEFORE AND AFTER
# ============================================================

outlier_cols = [
    "customer_care_calls",
    "customer_rating",
    "cost_of_product",
    "prior_purchases",
    "discount_offered",
    "weight_in_gms"
]
outlier_cols = [c for c in outlier_cols if c in df_clean.columns]

df_before_outliers = df_clean.copy()
outlier_summary = []

for col in outlier_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    outlier_summary.append({
        "Feature": col,
        "Lower fence": round(lower, 2),
        "Upper fence": round(upper, 2),
        "Outliers before treatment": int(n_out),
        "Treatment": "IQR winsorization"
    })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df.to_csv(os.path.join(TABLE_DIR, "Table_04_Outlier_Treatment.csv"), index=False)
save_table_pdf(outlier_df, "Table_04_Outlier_Treatment.pdf", "Table 4. Outlier Treatment Summary")

fig, axes = plt.subplots(2, len(outlier_cols), figsize=(3.5 * len(outlier_cols), 8))
if len(outlier_cols) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for i, col in enumerate(outlier_cols):
    axes[0, i].boxplot(df_before_outliers[col], patch_artist=True,
                       boxprops=dict(facecolor=ORANGE, alpha=0.65))
    axes[0, i].set_title(f"{col}\nBefore", fontsize=9, fontweight="bold")
    axes[0, i].set_xlabel("Before treatment")
    axes[0, i].set_ylabel("Value")

    axes[1, i].boxplot(df_clean[col], patch_artist=True,
                       boxprops=dict(facecolor=GREEN, alpha=0.65))
    axes[1, i].set_title(f"{col}\nAfter", fontsize=9, fontweight="bold")
    axes[1, i].set_xlabel("After treatment")
    axes[1, i].set_ylabel("Value")

fig.suptitle("Figure 6. Outlier Detection Before and After Winsorization", fontsize=15, fontweight="bold")
plt.tight_layout()
save_pdf(fig, "Figure_06_Outlier_Analysis_Before_After.pdf")
plt.show()


# ============================================================
# ENCODING AND SCALING
# ============================================================

df_encoded = df_clean.copy()

# Standardize strings
for col in ["gender", "product_importance"]:
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower()

# Binary encoding
if "gender" in df_encoded.columns:
    df_encoded["gender"] = df_encoded["gender"].map({"m": 1, "male": 1, "f": 0, "female": 0})

# Ordinal encoding
if "product_importance" in df_encoded.columns:
    df_encoded["product_importance"] = df_encoded["product_importance"].map({"low": 0, "medium": 1, "high": 2})

# One-hot encoding
ohe_cols = [c for c in ["warehouse_block", "mode_of_shipment"] if c in df_encoded.columns]
df_encoded = pd.get_dummies(df_encoded, columns=ohe_cols, drop_first=True)

# Convert bool to int
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Save cleaned encoded data
df_encoded.to_csv(os.path.join(RESULT_DIR, "ecommerce_cleaned_encoded.csv"), index=False)

# Scaled version
scale_cols = [
    "customer_care_calls",
    "cost_of_product",
    "discount_offered",
    "weight_in_gms",
    "prior_purchases"
]
scale_cols = [c for c in scale_cols if c in df_encoded.columns]

df_scaled = df_encoded.copy()
scaler = StandardScaler()
df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
df_scaled.to_csv(os.path.join(RESULT_DIR, "ecommerce_cleaned_scaled.csv"), index=False)

cleaning_summary = pd.DataFrame({
    "Step": [
        "Drop ID",
        "Rename variables",
        "Missing value check",
        "Duplicate check",
        "Outlier treatment",
        "Categorical encoding",
        "Feature scaling",
        "Save cleaned data"
    ],
    "Result": [
        "ID removed if present",
        "Columns renamed with simple names",
        "Numeric median and categorical mode used if needed",
        "Duplicate rows removed",
        "IQR winsorization applied",
        "Gender, product importance, warehouse, and mode encoded",
        "StandardScaler applied for selected variables",
        "Encoded and scaled CSV files saved"
    ]
})
cleaning_summary.to_csv(os.path.join(TABLE_DIR, "Table_05_Cleaning_Summary.csv"), index=False)
save_table_pdf(cleaning_summary, "Table_05_Cleaning_Summary.pdf", "Table 5. Data Cleaning Summary")


# ============================================================
# SECTION 3 - MODELING
# ============================================================

TARGET = "late_delivery"
FEATURES = [c for c in df_encoded.columns if c != TARGET]

X = df_encoded[FEATURES]
y = df_encoded[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model_scaler = StandardScaler()
X_train_sc = model_scaler.fit_transform(X_train)
X_test_sc = model_scaler.transform(X_test)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=42), False),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    "AdaBoost": (AdaBoostClassifier(n_estimators=100, random_state=42), False),
    "KNN": (KNeighborsClassifier(n_neighbors=7), True),
    "SVM": (SVC(probability=True, random_state=42), True),
    "Naive Bayes": (GaussianNB(), True),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=42), False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, (model, needs_scaling) in models.items():
    print(f"Training model: {name}")

    Xtr = X_train_sc if needs_scaling else X_train.values
    Xte = X_test_sc if needs_scaling else X_test.values

    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "CV Accuracy Mean": round(cv_scores.mean(), 4),
        "CV Accuracy Std": round(cv_scores.std(), 4),
        "_model": model,
        "_needs_scaling": needs_scaling,
        "_y_pred": y_pred,
        "_y_proba": y_proba
    })

results_df = pd.DataFrame(results)
results_table = results_df.drop(columns=["_model", "_needs_scaling", "_y_pred", "_y_proba"])
results_table = results_table.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

results_table.to_csv(os.path.join(RESULT_DIR, "model_results_comparison.csv"), index=False)
results_table.to_csv(os.path.join(TABLE_DIR, "Table_06_Model_Results.csv"), index=False)
save_table_pdf(results_table, "Table_06_Model_Results.pdf", "Table 6. Model Performance Comparison")


# ============================================================
# FIGURE 7 - MODEL COMPARISON
# ============================================================

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

fig, axes = plt.subplots(1, len(metrics), figsize=(24, 6))
model_names = results_table["Model"].tolist()

for ax, metric in zip(axes, metrics):
    vals = results_table[metric].values
    colors = [GREEN if value == max(vals) else BLUE for value in vals]
    bars = ax.barh(model_names, vals, color=colors, edgecolor="white")
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Machine learning model")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()

    for bar, value in zip(bars, vals):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8
        )

fig.suptitle("Figure 7. Model Comparison Across Evaluation Metrics", fontsize=15, fontweight="bold")
plt.tight_layout()
save_pdf(fig, "Figure_07_Model_Comparison.pdf")
plt.show()


# ============================================================
# FIGURE 8 - ROC CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random classifier (AUC = 0.50)")

for i, row in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, row["_y_proba"])
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        color=PALETTE[i % len(PALETTE)],
        label=f"{row['Model']} (AUC = {row['ROC-AUC']:.3f})"
    )

ax.set_title("Figure 8. ROC Curves for All Models", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
save_pdf(fig, "Figure_08_ROC_Curves.pdf")
plt.show()


# ============================================================
# BEST MODEL ANALYSIS
# ============================================================

best_name = results_table.iloc[0]["Model"]
best_row = next(row for row in results if row["Model"] == best_name)

best_model = best_row["_model"]
best_y_pred = best_row["_y_pred"]
best_y_proba = best_row["_y_proba"]

best_X_train = X_train_sc if best_row["_needs_scaling"] else X_train.values
best_X_test = X_test_sc if best_row["_needs_scaling"] else X_test.values

report_dict = classification_report(
    y_test,
    best_y_pred,
    target_names=["On-Time (0)", "Late (1)"],
    output_dict=True
)
classification_report_df = pd.DataFrame(report_dict).T.round(4)
classification_report_df.to_csv(os.path.join(RESULT_DIR, "classification_report_best_model.csv"))
save_table_pdf(
    classification_report_df.reset_index().rename(columns={"index": "Class"}),
    "Table_07_Classification_Report_Best_Model.pdf",
    f"Table 7. Classification Report for Best Model: {best_name}"
)


# ============================================================
# FIGURE 9 - BEST MODEL CONFUSION MATRIX AND LEARNING CURVE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Late"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion Matrix: {best_name}", fontweight="bold")
axes[0].set_xlabel("Predicted class")
axes[0].set_ylabel("Actual class")

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    best_X_train,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
    n_jobs=-1
)

axes[1].plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training accuracy", color=BLUE)
axes[1].plot(train_sizes, val_scores.mean(axis=1), marker="s", label="Cross-validation accuracy", color=GREEN)
axes[1].fill_between(
    train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.15,
    color=BLUE
)
axes[1].fill_between(
    train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.15,
    color=GREEN
)
axes[1].set_title(f"Learning Curve: {best_name}", fontweight="bold")
axes[1].set_xlabel("Training set size")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

fig.suptitle("Figure 9. Best Model Evaluation", fontsize=15, fontweight="bold")
plt.tight_layout()
save_pdf(fig, "Figure_09_Best_Model_Evaluation.pdf")
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_imp = RandomForestClassifier(n_estimators=200, random_state=42)
rf_imp.fit(X_train.values, y_train)

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_imp.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

feature_importance.to_csv(os.path.join(RESULT_DIR, "feature_importance.csv"), index=False)
feature_importance.to_csv(os.path.join(TABLE_DIR, "Table_08_Feature_Importance.csv"), index=False)
save_table_pdf(feature_importance, "Table_08_Feature_Importance.pdf", "Table 8. Random Forest Feature Importance")


# ============================================================
# FIGURE 10 - FEATURE IMPORTANCE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))
top_imp = feature_importance.head(15).iloc[::-1]
bars = ax.barh(top_imp["Feature"], top_imp["Importance"], color=BLUE, edgecolor="white")
ax.set_title("Figure 10. Feature Importance from Random Forest", fontweight="bold")
ax.set_xlabel("Importance score")
ax.set_ylabel("Input feature")

for bar, value in zip(bars, top_imp["Importance"]):
    ax.text(
        value + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        fontsize=8
    )

plt.tight_layout()
save_pdf(fig, "Figure_10_Feature_Importance.pdf")
plt.show()


# ============================================================
# HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train.values, y_train)

tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test.values)
y_proba_tuned = tuned_model.predict_proba(X_test.values)[:, 1]

tuned_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [
        round(accuracy_score(y_test, y_pred_tuned), 4),
        round(precision_score(y_test, y_pred_tuned), 4),
        round(recall_score(y_test, y_pred_tuned), 4),
        round(f1_score(y_test, y_pred_tuned), 4),
        round(roc_auc_score(y_test, y_proba_tuned), 4)
    ]
})

tuned_results.to_csv(os.path.join(RESULT_DIR, "tuned_random_forest_results.csv"), index=False)
save_table_pdf(tuned_results, "Table_09_Tuned_Random_Forest_Results.pdf", "Table 9. Tuned Random Forest Results")

best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(os.path.join(RESULT_DIR, "best_random_forest_parameters.csv"), index=False)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = X_test.copy()
test_predictions["actual_late_delivery"] = y_test.values
test_predictions["predicted_late_delivery"] = y_pred_tuned
test_predictions["probability_late_delivery"] = np.round(y_proba_tuned, 4)
test_predictions["correct_prediction"] = (
    test_predictions["actual_late_delivery"] == test_predictions["predicted_late_delivery"]
).astype(int)

test_predictions.to_csv(os.path.join(RESULT_DIR, "test_predictions.csv"), index=False)


# ============================================================
# SAVE LINKS FOR OVERLEAF / GITHUB
# ============================================================

with open(os.path.join(DATASET_DIR, "dataset_link.txt"), "w") as f:
    f.write("Dataset Name: Customer Analytics\n")
    f.write("Dataset Owner: prachi13\n")
    f.write("Dataset Link: https://www.kaggle.com/datasets/prachi13/customer-analytics\n")

with open(os.path.join(CODE_DIR, "kaggle_notebook_link.txt"), "w") as f:
    f.write("Kaggle Notebook Link:\n")
    f.write("https://www.kaggle.com/code/maryammoghaddasbayat/dataanalytics/edit\n")

with open(os.path.join(GITHUB_DIR, "github_repository_link.txt"), "w") as f:
    f.write("Paste the GitHub repository link here.\n")


# ============================================================
# SECTION 4 - WORKFLOW FIGURE
# This must be the final generated figure.
# ============================================================

workflow_steps = [
    ("01", "Dataset\nAcquisition", "Kaggle dataset\nis added to notebook"),
    ("02", "Data Mining", "Variable description\nand first audit"),
    ("03", "Data Quality\nAssessment", "Missing values\nand duplicates"),
    ("04", "Data Cleaning", "Rename columns\nand remove ID"),
    ("05", "Outlier\nTreatment", "IQR winsorization\nfor numeric features"),
    ("06", "Exploratory\nAnalysis", "Distributions\nand correlation"),
    ("07", "Feature\nEncoding", "Binary, ordinal\nand one-hot encoding"),
    ("08", "Data Splitting\nand Scaling", "Train-test split\nand StandardScaler"),
    ("09", "Model\nDevelopment", "Nine machine\nlearning models"),
    ("10", "Model\nEvaluation", "Accuracy, F1\nand ROC-AUC"),
    ("11", "Hyperparameter\nTuning", "GridSearchCV for\nRandom Forest"),
    ("12", "Feature\nImportance", "Main drivers of\nlate delivery"),
    ("13", "Predictive\nAnalytics", "Late delivery\nrisk prediction"),
    ("14", "Prescriptive\nAnalytics", "Business action\nrecommendations")
]

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")

phase_colors = {
    "data": "#E3F2FD",
    "clean": "#FFF3E0",
    "model": "#E8F5E9",
    "business": "#F3E5F5"
}
edge_colors = {
    "data": "#1976D2",
    "clean": "#F57C00",
    "model": "#388E3C",
    "business": "#7B1FA2"
}

positions = [
    (1, 10), (5.2, 10), (9.4, 10), (13.6, 10),
    (1, 7.2), (5.2, 7.2), (9.4, 7.2), (13.6, 7.2),
    (1, 4.4), (5.2, 4.4), (9.4, 4.4), (13.6, 4.4),
    (4.2, 1.6), (10.2, 1.6)
]

def phase_for_index(i):
    if i <= 3:
        return "data"
    if i <= 7:
        return "clean"
    if i <= 11:
        return "model"
    return "business"

box_w = 3.2
box_h = 1.5

for i, (num, title, subtitle) in enumerate(workflow_steps, start=1):
    x, y = positions[i - 1]
    phase = phase_for_index(i)

    box = patches.FancyBboxPatch(
        (x, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        linewidth=2,
        edgecolor=edge_colors[phase],
        facecolor=phase_colors[phase]
    )
    ax.add_patch(box)

    circle = patches.Circle(
        (x + 0.35, y + box_h - 0.35),
        0.28,
        facecolor=edge_colors[phase],
        edgecolor="white",
        linewidth=1.5
    )
    ax.add_patch(circle)

    ax.text(
        x + 0.35,
        y + box_h - 0.35,
        num,
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    ax.text(
        x + box_w / 2,
        y + 0.88,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#263238"
    )

    ax.text(
        x + box_w / 2,
        y + 0.33,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#455A64"
    )

# Arrows
arrow_pairs = [
    (0, 1), (1, 2), (2, 3),
    (3, 7),
    (7, 6), (6, 5), (5, 4),
    (4, 8),
    (8, 9), (9, 10), (10, 11),
    (11, 13),
    (13, 12)
]

for start, end in arrow_pairs:
    x1, y1 = positions[start]
    x2, y2 = positions[end]

    start_point = (x1 + box_w, y1 + box_h / 2)
    end_point = (x2, y2 + box_h / 2)

    if abs(x1 - x2) < 0.1:
        start_point = (x1 + box_w / 2, y1)
        end_point = (x2 + box_w / 2, y2 + box_h)

    if x2 < x1:
        start_point = (x1, y1 + box_h / 2)
        end_point = (x2 + box_w, y2 + box_h / 2)

    ax.annotate(
        "",
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(arrowstyle="->", lw=2, color="#546E7A")
    )

# Phase labels
ax.text(2.6, 11.6, "Data Understanding", ha="center", fontsize=12, fontweight="bold", color="#1976D2")
ax.text(9.0, 8.9, "Data Preparation", ha="center", fontsize=12, fontweight="bold", color="#F57C00")
ax.text(9.0, 6.1, "Machine Learning", ha="center", fontsize=12, fontweight="bold", color="#388E3C")
ax.text(9.0, 0.7, "Decision Support", ha="center", fontsize=12, fontweight="bold", color="#7B1FA2")

ax.set_title(
    "Figure 11. Complete Workflow of Commerce Delivery Optimization Framework",
    fontsize=17,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

workflow_pdf_1 = os.path.join(FIG_DIR, "Figure_11_Project_Workflow.pdf")
workflow_pdf_2 = os.path.join(WORKFLOW_DIR, "Figure_11_Project_Workflow.pdf")
fig.savefig(workflow_pdf_1, format="pdf", bbox_inches="tight")
fig.savefig(workflow_pdf_2, format="pdf", bbox_inches="tight")

plt.show()
