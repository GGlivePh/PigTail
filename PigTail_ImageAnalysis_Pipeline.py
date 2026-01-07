# ============================================================
# PigTail image analysis pipeline
# Grum Gebreyesus Teklewold
# Center for Quantitative Genetics and Genomics
# ============================================================
import os, json, cv2, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ============================================================
# 1. Paths and constants
# ============================================================

BASE = "/data/pigtail"
COCO1 = os.path.join(BASE, "Raw_72_Train.json")
COCO2 = os.path.join(BASE, "Corrected_Val_Pre_Annotations_Raw_New.json")
GT_PATH = os.path.join(BASE, "taillength_ground_truth_Raw_Clean.txt")
OUTPUT = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT, exist_ok=True)

RULER_WIDTH_CM = 5.4
RULER_HEIGHT_CM = 8.57

# ============================================================
# 2. COCO helpers
# ============================================================

def coco_seg_to_polygons(seg):
    if isinstance(seg, list):
        return seg
    return []

def polygons_to_mask(polygons, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly).reshape(-1,2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

# ============================================================
# 3. Tail geometry
# ============================================================

def compute_tail_length_px(mask):
    skel = skeletonize(mask > 0)
    return float(skel.sum())

def compute_tail_thickness_px(mask):
    dist = distance_transform_edt(mask > 0)
    vals = dist[dist > 0]
    return 2*vals.mean(), 2*vals.max()

def compute_advanced_features(mask):
    props = regionprops(mask.astype(int))[0]
    area = props.area
    perim = props.perimeter
    bbox_w = props.bbox[3] - props.bbox[1]
    bbox_h = props.bbox[2] - props.bbox[0]
    convex = props.convex_area

    return {
        "area_px": area,
        "perimeter_px": perim,
        "bbox_w_px": bbox_w,
        "bbox_h_px": bbox_h,
        "convex_area_px": convex,
        "solidity": area/convex if convex>0 else np.nan,
        "extent": area/(bbox_w*bbox_h) if bbox_w*bbox_h>0 else np.nan,
        "eccentricity": props.eccentricity,
        "compactness": perim**2/(4*np.pi*area) if area>0 else np.nan,
        "circularity": 4*np.pi*area/(perim**2) if perim>0 else np.nan,
    }

# ============================================================
# 4. Ruler calibration
# ============================================================

def cm_per_pixel_from_ruler_mask(ruler_mask):
    cnts,_ = cv2.findContours(ruler_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)==0:
        return None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return ((RULER_WIDTH_CM/w)+(RULER_HEIGHT_CM/h))/2

# ============================================================
# 5. Load COCO annotations
# ============================================================

with open(COCO1) as f: coco1=json.load(f)
with open(COCO2) as f: coco2=json.load(f)

images = coco1["images"] + coco2["images"]
annotations = coco1["annotations"] + coco2["annotations"]

# ============================================================
# 6. Feature extraction
# ============================================================

rows = []

for img in images:
    w,h = img["width"], img["height"]
    fname = img["file_name"]
    anns = [a for a in annotations if a["image_id"]==img["id"]]

    tail_polys = [p for a in anns if a["category_id"]==1 for p in coco_seg_to_polygons(a["segmentation"])]
    ruler_polys = [p for a in anns if a["category_id"]==2 for p in coco_seg_to_polygons(a["segmentation"])]

    if not tail_polys or not ruler_polys:
        continue

    tail_mask = polygons_to_mask(tail_polys,w,h)
    ruler_mask = polygons_to_mask(ruler_polys,w,h)

    cm_px = cm_per_pixel_from_ruler_mask(ruler_mask)
    if cm_px is None:
        continue

    Lpx = compute_tail_length_px(tail_mask)
    Tmean,Tmax = compute_tail_thickness_px(tail_mask)
    feats = compute_advanced_features(tail_mask)

    rows.append({
        "Filename": fname,
        "TagID": os.path.splitext(fname)[0],
        "TailLength_cm_skeleton": Lpx * cm_px,
        "TailThickness_cm_mean": Tmean * cm_px,
        "TailThickness_cm_max": Tmax * cm_px,
        "TailArea_cm2": feats["area_px"] * cm_px**2,
        "TailPerimeter_cm": feats["perimeter_px"] * cm_px,
        "TailBBoxWidth_cm": feats["bbox_w_px"] * cm_px,
        "TailBBoxHeight_cm": feats["bbox_h_px"] * cm_px,
        "TailConvexArea_cm2": feats["convex_area_px"] * cm_px**2,
        "TailSolidity": feats["solidity"],
        "TailExtent": feats["extent"],
        "TailEccentricity": feats["eccentricity"],
        "TailCompactness": feats["compactness"],
        "TailCircularity": feats["circularity"]
    })

df_feat = pd.DataFrame(rows)
print("Extracted tails:", len(df_feat))

# ============================================================
# 7. Merge with ground truth
# ============================================================

gt = pd.read_csv(GT_PATH, sep=r"\s+", engine="python")
gt["TagID"] = gt["TagID"].astype(str).str.replace(".0","")

df_feat["TagID"] = df_feat["TagID"].astype(str)
df_feat = df_feat.merge(gt, on="TagID", how="inner")
print("Merged rows:", len(df_feat))

# ============================================================
# 8. Feature engineering
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
# 9. Tail length regression
# ============================================================

reg_features = [
    "TailLength_cm_skeleton","TailThickness_cm_mean","TailThickness_cm_max",
    "TailArea_cm2","TailPerimeter_cm","TailBBoxWidth_cm","TailBBoxHeight_cm",
    "TailConvexArea_cm2","TailSolidity","TailExtent","TailEccentricity",
    "TailCompactness","TailCircularity",
    "ThicknessRatio_mean_vs_len","ThicknessRatio_mean_vs_sqrtArea",
    "ThicknessRatio_mean_vs_perimeter",
    "ThicknessRatio_max_vs_len","ThicknessRatio_max_vs_sqrtArea",
    "ThicknessRatio_max_vs_perimeter","TaperIndex"
]

df_reg = df_feat.dropna(subset=reg_features + ["TailLength_cm_gt"])
X = df_reg[reg_features].values
y = df_reg["TailLength_cm_gt"].values

Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)

rf_len = RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
rf_len.fit(Xtr,ytr)

df_feat["PredTailLength_cm"] = rf_len.predict(df_feat[reg_features].fillna(0).values)

print("Test R2:", r2_score(yte, rf_len.predict(Xte)))
print("Test RMSE:", np.sqrt(mean_squared_error(yte, rf_len.predict(Xte))))

# ============================================================
# 10. Docking classification
# ============================================================

clf_features = reg_features
df_clf = df_feat.dropna(subset=clf_features+["TailDocking"])

Xc = df_clf[clf_features].values
yc = df_clf["TailDocking"].values

Xtr,Xte,ytr,yte = train_test_split(Xc,yc,test_size=0.2,random_state=42,stratify=yc)

rf_clf = RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1,class_weight="balanced")
rf_clf.fit(Xtr,ytr)
yp = rf_clf.predict(Xte)

print(classification_report(yte,yp))
cm = confusion_matrix(yte,yp,labels=["Intact","Half","Quarter"])

sns.heatmap(cm,annot=True,fmt="d",xticklabels=["Intact","Half","Quarter"],yticklabels=["Intact","Half","Quarter"])
plt.savefig(os.path.join(OUTPUT,"confusion_standard_RF.png"))
plt.close()

# ============================================================
# 11. Fraction-missing model
# ============================================================

df_intact = df_feat[df_feat["TailDocking"]=="Intact"].dropna(subset=reg_features+["TailLength_cm_gt"])
Xn = df_intact[reg_features].values
yn = df_intact["TailLength_cm_gt"].values

rf_norm = RandomForestRegressor(n_estimators=300,random_state=42)
rf_norm.fit(Xn,yn)

df_feat["PredictedNormalTail_cm"] = rf_norm.predict(df_feat[reg_features].fillna(0).values)
df_feat["FracMissing_GT"] = 1 - df_feat["TailLength_cm_gt"]/df_feat["PredictedNormalTail_cm"]

# ============================================================
# 12. Save outputs
# ============================================================

df_feat.to_csv(os.path.join(OUTPUT,"all_features_and_predictions.csv"),index=False)
joblib.dump(rf_len,os.path.join(OUTPUT,"rf_length.pkl"))
joblib.dump(rf_clf,os.path.join(OUTPUT,"rf_classifier.pkl"))
joblib.dump(rf_norm,os.path.join(OUTPUT,"rf_normal.pkl"))

print("Pipeline finished. All outputs written to:", OUTPUT)


