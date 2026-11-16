"""
Phần mô hình của web Đất & Cây (không phụ thuộc Streamlit, để tự kiểm tra được).
Khớp notebook 04b: Random Forest tự cài đặt (thuat_toan.MyRandomForest), tham số chọn ở mục 7,
và mục 15b.4: học thêm các bản sao nhiễu (chỉ trong bộ nhớ, dữ liệu gốc không đổi).
"""
import os, pickle
import numpy as np
import pandas as pd
from thuat_toan import MyRandomForest, do_quan_trong_hoan_vi

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache_models")
SEED = 42
FEATS = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
SOIL = ["N", "P", "K", "pH"]
TARGET = "Crop"

# Tham số đã chọn ở notebook (mục 7). Web dùng vote="soft" để top-3 có xác suất mượt (mục 2.4).
THAM_SO = {"criterion": "gini", "max_depth": None, "min_samples_leaf": 1, "max_features": 3,
           "n_estimators": 100, "min_samples_split": 2, "vote": "soft", "random_state": SEED}

# chế độ huấn luyện: nhãn -> mức nhiễu khi học (0 = chỉ dữ liệu sạch). Khớp mục 15b.4.
MODES = {"Học thêm nhiễu 10% (mặc định)": 0.10, "Học thêm nhiễu 20%": 0.20, "Học dữ liệu sạch": 0.0}
DEFAULT_MODE = "Học thêm nhiễu 10% (mặc định)"
N_BAN = 5   # số bản sao nhiễu cho mỗi dòng train (như notebook)


def them_nhieu(df, muc, rng):
    """Bản SAO của df: mỗi chỉ số nhân hệ số ngẫu nhiên đều trong [1-a, 1+a]. Nhãn không đổi. Độ ẩm chặn trên 100."""
    out = df.copy()
    for c in FEATS:
        out[c] = out[c].to_numpy(float) * rng.uniform(1 - muc, 1 + muc, len(out))
    out["humidity"] = out["humidity"].clip(upper=100)
    return out


def doc_du_lieu():
    tr = pd.read_csv(os.path.join(BASE, "train.csv"))
    te = pd.read_csv(os.path.join(BASE, "test.csv"))
    return tr, te


def tao_tap_huan_luyen(tr, muc):
    """Trả về (DataFrame 7 chỉ số, nhãn). muc=0: chỉ dữ liệu sạch. Dữ liệu gốc `tr` không bị sửa."""
    if muc <= 0:
        return tr[FEATS].reset_index(drop=True), tr[TARGET].to_numpy()
    seed = SEED + 7 + (1 if abs(muc - 0.10) < 1e-9 else 2)   # đúng seed của notebook 15b.4
    rng = np.random.default_rng(seed)
    X = pd.concat([tr[FEATS]] + [them_nhieu(tr, muc, rng)[FEATS] for _ in range(N_BAN)], ignore_index=True)
    y = np.tile(tr[TARGET].to_numpy(), N_BAN + 1)
    assert len(X) == len(tr) * (N_BAN + 1) == len(y)
    return X, y


def lay_mo_hinh(tr, muc, cols):
    """Huấn luyện (hoặc nạp từ cache_models/) rừng cho bộ chỉ số `cols` với mức nhiễu học `muc`."""
    cols = [c for c in FEATS if c in cols]
    os.makedirs(CACHE_DIR, exist_ok=True)
    f = os.path.join(CACHE_DIR, f"rf_n{int(round(muc * 100))}_{'-'.join(cols)}.pkl")
    if os.path.exists(f):
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    X, y = tao_tap_huan_luyen(tr, muc)
    p = dict(THAM_SO); p["max_features"] = min(p["max_features"], len(cols))
    m = MyRandomForest(**p).fit(X[cols], y)
    try:
        with open(f, "wb") as fh:
            pickle.dump(m, fh)
    except Exception:
        pass
    return m


def danh_gia(m, X, y):
    P = m.predict_proba(X)
    cls = m.classes_
    top = np.argsort(-P, axis=1)
    pred = cls[top[:, 0]]
    top3 = float(np.mean([yy in cls[t[:3]] for yy, t in zip(y, top)]))
    f1 = {}
    for c in cls:
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(y == c), 1)
        f1[c] = float(2 * p * r / (p + r)) if p + r else 0.0
    return {"acc": float(np.mean(pred == y)), "top3": top3, "f1": f1, "macro": float(np.mean(list(f1.values()))),
            "sai": int(np.sum(pred != y)), "n": len(y)}


def do_ben(m, te, muc_list=(0.05, 0.10, 0.20), n_lan=5):
    """Accuracy trên test gốc và trên bản sao test bị làm lệch (nhiễu giả lập). Trả {muc: (tb, độ lệch)}."""
    y = te[TARGET].to_numpy()
    cols = m.feature_names_
    out = {0.0: (float(np.mean(m.predict(te[cols]) == y)), 0.0)}
    for i, mu in enumerate(muc_list, 1):
        a = [float(np.mean(m.predict(them_nhieu(te, mu, np.random.default_rng(SEED + 100 * i + j))[cols]) == y)) for j in range(n_lan)]
        out[mu] = (float(np.mean(a)), float(np.std(a)))
    return out


def quan_trong_hoan_vi(m, te):
    hv = do_quan_trong_hoan_vi(m, te[FEATS], te[TARGET].to_numpy(), n_repeats=5, random_state=SEED).set_index("Chỉ số")
    return hv["Accuracy giảm (TB)"].to_dict()
