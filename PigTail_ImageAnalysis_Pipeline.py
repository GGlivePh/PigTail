# ============================================================
# PigTail image analysis pipeline
# Grum Gebreyesus Teklewold
# Center for Quantitative Genetics and Genomics
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from skimage.morphology import skeletonize
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelBinarizer

# ============================================================
# 1. Data paths
# ============================================================

DATA_ROOT = "/data/pigtail"

FEATURE_FILE = os.path.join(DATA_ROOT, "image_features.csv")
GROUND_TRUTH_FILE = os.path.join(DATA_ROOT, "taillength_ground_truth_Raw_Clean.txt")

OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. Load merged feature table
# ============================================================

df_feat = pd.read_csv(FEATURE_FILE)
gt = pd.read_csv(GROUND_TRUTH_FILE, sep=r"\s+", engine="python")

gt["TagID"] = gt["TagID"].astype(str).str.replace(".0", "")
df_feat["TagID"] = df_feat["TagID"].astype(str)

df_feat = df_feat.merge(gt[["TagID", "TailLength_cm", "TailThickness_cm", "TailDocking", "HerdId"]],
                        on="TagID", how="left")

df_feat.rename(columns={
    "TailLength_cm": "TailLength_cm_gt",
    "TailThickness_cm": "TailThickness_cm_gt"
}, inplace=True)

print("Merged rows:", len(df_feat))

# ============================================================
# 3. Feature engineering
# ============================================================

eps = 1e-6

L = df_feat["TailLength_cm_skeleton"]
Tmean = df_feat["TailThickness_cm_mean"]
Tmax = df_feat["TailThickness_cm_max"]
A = df_feat["TailArea_cm2"]
P = df_feat["TailPerimeter_cm"]

df_feat["ThicknessRatio_mean_vs_len"] = Tmean / (L + eps)
df_feat["ThicknessRatio_mean_vs_sqrtArea"] = Tmean / (np.sqrt(A) + eps)
df_feat["ThicknessRatio_mean_vs_perimeter"] = Tmean / (P + eps)

df_feat["ThicknessRatio_max_vs_len"] = Tmax / (L + eps)
df_feat["ThicknessRatio_max_vs_sqrtArea"] = Tmax / (np.sqrt(A) + eps)
df_feat["ThicknessRatio_max_vs_perimeter"] = Tmax / (P + eps)

df_feat["TaperIndex"] = Tmax / (Tmean + eps)

# ============================================================
# 4. Tail length regression (Random Forest)
# ============================================================

reg_features = [
    "TailLength_cm_skeleton","TailThickness_cm_mean","TailThickness_cm_max",
    "TailArea_cm2","TailPerimeter_cm","TailBBoxWidth_cm","TailBBoxHeight_cm",
    "TailAspectRatio","TailConvexArea_cm2","TailSolidity","TailExtent",
    "TailEccentricity","TailTortuosity","TailCompactness","TailCircularity",
    "ThicknessRatio_mean_vs_len","ThicknessRatio_mean_vs_sqrtArea",
    "ThicknessRatio_mean_vs_perimeter",
    "ThicknessRatio_max_vs_len","ThicknessRatio_max_vs_sqrtArea",
    "ThicknessRatio_max_vs_perimeter","TaperIndex"
]

df_reg = df_feat.dropna(subset=reg_features + ["TailLength_cm_gt"])
X = df_reg[reg_features]
y = df_reg["TailLength_cm_gt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_len = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_len.fit(X_train, y_train)

y_pred_test = rf_len.predict(X_test)

print("Length R²:", r2_score(y_test, y_pred_test))
print("Length RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

df_feat["PredTailLength_cm"] = rf_len.predict(df_feat[reg_features].fillna(0))

# ============================================================
# 5. Docking classification (RF)
# ============================================================

clf_features = reg_features

df_clf = df_feat.dropna(subset=clf_features + ["TailDocking"])
Xc = df_clf[clf_features]
yc = df_clf["TailDocking"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

rf_clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
rf_clf.fit(Xc_train, yc_train)

yc_pred = rf_clf.predict(Xc_test)

print(classification_report(yc_test, yc_pred))

cm = confusion_matrix(yc_test, yc_pred, labels=["Intact","Half","Quarter"])

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Intact","Half","Quarter"],
            yticklabels=["Intact","Half","Quarter"],
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.title("Docking classification")
plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix_rf.png"))
plt.close()

# ============================================================
# 6. Normal-tail model (intact only)
# ============================================================

df_intact = df_feat[df_feat["TailDocking"]=="Intact"].dropna(subset=reg_features + ["TailLength_cm_gt"])

Xn = df_intact[reg_features]
yn = df_intact["TailLength_cm_gt"]

rf_normal = RandomForestRegressor(n_estimators=300, random_state=42)
rf_normal.fit(Xn, yn)

df_feat["PredictedNormalTail_cm"] = rf_normal.predict(df_feat[reg_features].fillna(0))
df_feat["FracMissing_Pred"] = 1 - df_feat["PredTailLength_cm"] / (df_feat["PredictedNormalTail_cm"] + eps)

# ============================================================
# 7. Fraction-missing threshold model
# ============================================================

df_thresh = df_feat.dropna(subset=["FracMissing_Pred","TailDocking"])
df_thresh = df_thresh[df_thresh["TailDocking"]!="Intact"]

best_acc=0
best_t=0

for t in np.linspace(0.2,0.8,200):
    pred = np.where(df_thresh["FracMissing_Pred"]<t,"Half","Quarter")
    acc = np.mean(pred==df_thresh["TailDocking"])
    if acc>best_acc:
        best_acc=acc
        best_t=t

print("Best threshold:",best_t,"Accuracy:",best_acc)

# ============================================================
# 8. Herd-level validation
# ============================================================

herd_stats=[]
for herd,g in df_feat.groupby("HerdId"):
    if len(g)<10: continue
    herd_stats.append({
        "Herd":herd,
        "R2":r2_score(g["TailLength_cm_gt"],g["PredTailLength_cm"]),
        "RMSE":np.sqrt(mean_squared_error(g["TailLength_cm_gt"],g["PredTailLength_cm"]))
    })

herd_stats=pd.DataFrame(herd_stats)
print(herd_stats)

herd_stats.to_csv(os.path.join(OUTPUT_DIR,"herd_performance.csv"),index=False)

# ============================================================
# 9. Save models
# ============================================================

joblib.dump(rf_len, os.path.join(OUTPUT_DIR,"rf_tail_length.pkl"))
joblib.dump(rf_clf, os.path.join(OUTPUT_DIR,"rf_docking.pkl"))
joblib.dump(rf_normal, os.path.join(OUTPUT_DIR,"rf_normal_tail.pkl"))

print("Pipeline completed successfully")
