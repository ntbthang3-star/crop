"""
Đất & Cây - Hệ thống hỗ trợ chọn cây trồng theo đất và khí hậu (giao diện kính olive, ảnh thật).
Mô hình: MyRandomForest tự cài đặt (thuat_toan.py), huấn luyện theo notebook 04b (có học thêm bản sao nhiễu).
Chạy:  streamlit run app.py
"""
import os, io, re, base64, unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import mo_hinh_web as M
from mo_hinh_web import FEATS, SOIL, TARGET
from noi_dung import (LBL, SHORT, LIMIT, CROPS, GROUPS, PERENNIAL, GROUP_SEASON, STEPS_ANNUAL, STEPS_PERENNIAL, ADV)

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Đất & Cây", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")
S = st.session_state

# ----------------------------------------------------------------------
# Số liệu đã đo trong notebook 04b (hiển thị ở "Phòng thí nghiệm"; chạy lại notebook thì sửa ở đây)
# ----------------------------------------------------------------------
NB_B10 = [("N", .1705), ("P", .1886), ("K", .3023), ("Nhiệt độ", .1727), ("Độ ẩm", .3045), ("pH", .1136), ("Lượng mưa", .3114),
          ("Nhóm đất (N+P+K+pH)", .7909), ("Nhóm khí hậu (3 chỉ số)", .9432), ("Cả 7 chỉ số", .9932)]
NB_NOISE = ["Số sạch", "Lệch 5%", "Lệch 10%", "Lệch 20%"]
NB_SUBSET = [("2 chỉ số: độ ẩm + mưa", [.773, .667, .562, .403]), ("3 chỉ số: K + độ ẩm + mưa", [.959, .889, .788, .605]),
             ("4 chỉ số: N + K + độ ẩm + mưa", [.989, .955, .898, .738]), ("Đủ 7 chỉ số", [.993, .977, .944, .840])]
NB_TANG = [("Học dữ liệu sạch (1760 dòng)", [.993, .977, .944, .840]), ("Học thêm nhiễu 10% (10560 dòng)", [.986, .984, .979, .896]),
           ("Học thêm nhiễu 20% (10560 dòng)", [.984, .979, .971, .942])]
NB_TREES = [("1 cây", .9429), ("5 cây", .9632), ("10 cây", .981), ("25 cây", .9932), ("50 cây", .996), ("100 cây (chọn)", .9949), ("200 cây", .996)]
NB_CALIB = [("0,40 – 0,60", 4, "54%", "75%"), ("0,60 – 0,80", 12, "73%", "83%"), ("0,80 – 0,95", 45, "89%", "100%"), ("0,95 – 1,00", 379, "99%", "100%")]
ACC_SUBSET = {frozenset(["N", "P", "K", "pH"]): .79, frozenset(["humidity", "rainfall"]): .77, frozenset(["K", "humidity", "rainfall"]): .96,
              frozenset(["N", "K", "humidity", "rainfall"]): .99, frozenset(FEATS): .99}
MEASURE_ORDER = ["humidity", "rainfall", "K", "N", "P", "temperature", "pH"]   # nên đo thêm theo thứ tự này
CONFUSED = {"rice": "jute", "jute": "rice", "blackgram": "maize", "maize": "blackgram", "lentil": "mothbeans", "mothbeans": "lentil"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
:root{--out:#11180f;--app:#1c2716;--card:rgba(62,77,44,.62);--ink:#f2f3e8;--mut:#c0c7ae;--fo:#d4e66b;--li:#273316;--cl:#ffb59c;--mo:#8cae39;--acc:rgba(14,22,11,.42);--line:rgba(224,232,195,.2);}
html,body,[class*="css"],.stMarkdown,button,input,label{font-family:'Be Vietnam Pro',system-ui,sans-serif !important;}
.stApp{background:radial-gradient(ellipse at 15% 0%,rgba(124,151,54,.28),transparent 40%),linear-gradient(135deg,#11180f,#202a17 52%,#121a11) fixed;color:var(--ink);}
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"],header[data-testid="stHeader"],footer{display:none !important;}
.block-container{max-width:1240px;padding-top:.8rem;padding-bottom:3rem;}
h1,h2,h3,h4,h5,.fd{font-family:'Bricolage Grotesque','Be Vietnam Pro',sans-serif !important;color:var(--ink);font-weight:600 !important;letter-spacing:-.01em;}
[data-testid="stCaptionContainer"],.stCaption{color:var(--mut) !important;}
em{font-style:normal;color:var(--fo);}
/* thanh điều hướng */
div[role="radiogroup"]{gap:.3rem;flex-wrap:wrap;}
div[role="radiogroup"] label{background:transparent;border:1px solid var(--line);border-radius:99px;padding:.22rem .9rem;cursor:pointer;margin:0;}
div[role="radiogroup"] label > div:first-child{display:none;}
div[role="radiogroup"] label:has(input:checked){background:rgba(206,228,125,.2);border-color:rgba(214,236,101,.55);}
div[role="radiogroup"] label *{color:var(--mut) !important;font-size:.88rem;}
div[role="radiogroup"] label:has(input:checked) *{color:var(--ink) !important;}
.logo{display:flex;align-items:center;gap:9px;font:600 21px 'Bricolage Grotesque',sans-serif;color:var(--ink);padding-top:.2rem;}
.logo i{width:22px;height:22px;border-radius:6px 6px 6px 2px;background:var(--fo);display:block;}
.navline{border-bottom:1px solid var(--line);margin:.2rem 0 1.1rem;}
/* nút */
.stButton > button{width:100%;border-radius:99px;border:1px solid var(--line);background:rgba(25,35,20,.55);color:var(--ink);font-weight:600;font-size:.84rem;padding:.4rem .8rem;transition:all .15s;}
.stButton > button:hover{border-color:var(--fo);background:rgba(151,177,69,.25);color:var(--ink);}
.stButton > button[kind="primary"]{background:var(--fo);border-color:var(--fo);color:var(--li);}
.stButton > button[kind="primary"]:hover{background:#e3f27f;color:var(--li);}
/* thẻ kính */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--line) !important;border-radius:18px;backdrop-filter:blur(16px);box-shadow:0 12px 32px rgba(0,0,0,.18);}
.glass{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;backdrop-filter:blur(16px);box-shadow:0 12px 32px rgba(0,0,0,.18);}
.eyebrow{display:inline-block;color:var(--fo);font-size:11px;font-weight:700;letter-spacing:.17em;margin-bottom:8px;text-transform:uppercase;}
.note{font-size:13.5px;background:var(--acc);border-radius:12px;padding:10px 13px;margin:.5rem 0;color:var(--ink);}
.note.w{background:rgba(255,150,110,.14);}
.tag{display:inline-block;background:rgba(214,236,101,.14);border-radius:99px;padding:2px 12px;font-size:13px;margin:0 6px 6px 0;color:var(--fo);}
/* hero có ảnh nền thật */
.hero{position:relative;border-radius:30px;overflow:hidden;min-height:440px;border:1px solid var(--line);background-size:cover;background-position:center;box-shadow:0 20px 55px -20px rgba(0,0,0,.6);}
.hero .ov{position:absolute;inset:0;background:linear-gradient(100deg,rgba(14,22,11,.93) 0%,rgba(24,34,16,.72) 50%,rgba(24,34,16,.12) 100%);}
.hero .tx{position:relative;padding:clamp(22px,4vw,48px);max-width:660px;}
.hero h1{font-size:clamp(34px,5.2vw,60px);font-weight:700 !important;line-height:1.06;margin:.3rem 0 .9rem;}
.hero p{color:#dfe6cc;max-width:52ch;line-height:1.65;margin:0 0 1.2rem;}
.hs{display:flex;gap:10px;flex-wrap:wrap;}
.hs div{background:rgba(226,235,196,.13);border:1px solid rgba(255,255,255,.2);backdrop-filter:blur(8px);border-radius:16px;padding:9px 16px;}
.hs b{display:block;font:700 26px 'Bricolage Grotesque',sans-serif;}
.hs span{font-size:12px;color:var(--mut);}
.sec{font:600 clamp(26px,3.4vw,36px) 'Bricolage Grotesque',sans-serif;text-align:center;margin:.2rem 0 .3rem;}
.sub{text-align:center;color:var(--mut);font-size:.92rem;margin-bottom:1.2rem;}
.center{text-align:center;margin-top:2.4rem;}
.step{border-top:3px solid var(--mo);background:var(--card);border-radius:14px;padding:1rem 1rem 1.1rem;height:100%;}
.step .n{font:700 2rem 'Bricolage Grotesque',sans-serif;color:var(--fo);line-height:1;}
.step h4{margin:.5rem 0 .3rem;font-size:1.1rem;}
.step p{margin:0;color:var(--mut);font-size:.88rem;line-height:1.6;}
.pcard{position:relative;border-radius:20px;overflow:hidden;height:250px;background-size:cover;background-position:center;border:1px solid var(--line);box-shadow:0 8px 24px rgba(0,0,0,.3);margin-bottom:.45rem;}
.pcard.sm{height:150px;}
.pcard .ov{position:absolute;inset:0;background:linear-gradient(180deg,rgba(10,18,8,0) 38%,rgba(10,18,8,.9) 100%);}
.pcard .tg{position:absolute;top:12px;left:12px;font-size:.65rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600;color:#fff;background:rgba(14,22,11,.6);backdrop-filter:blur(4px);padding:.2rem .6rem;border-radius:99px;}
.pcard .tx{position:absolute;left:16px;right:16px;bottom:14px;color:#fff;}
.pcard .nm{font:600 1.4rem 'Bricolage Grotesque',sans-serif;line-height:1.15;}
.pcard.sm .nm{font-size:1.1rem;}
.pcard .la{font-style:italic;font-size:.76rem;color:#dbe6c6;}
.pcard .mt{font-size:.74rem;color:#e9efdc;margin-top:.3rem;}
.pcard .pp{position:absolute;top:10px;right:12px;font-weight:700;color:var(--li);background:var(--fo);border-radius:99px;padding:.1rem .6rem;font-size:.8rem;}
.mini{display:flex;flex-wrap:wrap;gap:5px;margin-top:.35rem;}
.mini i{font-style:normal;background:rgba(14,22,11,.5);border-radius:99px;padding:2px 9px;font-size:11.5px;color:#e9efdc;}
.rescard{position:relative;border-radius:24px;overflow:hidden;height:300px;background-size:cover;background-position:center;border:1px solid var(--line);box-shadow:0 14px 40px rgba(0,0,0,.4);}
.rescard .ov{position:absolute;inset:0;background:linear-gradient(180deg,rgba(10,18,8,.05) 30%,rgba(10,18,8,.92) 100%);}
.rescard .tx{position:absolute;left:22px;right:22px;bottom:18px;color:#fff;}
.rescard .rk{font-size:.7rem;letter-spacing:.16em;color:var(--fo);font-weight:700;}
.rescard h2{color:#fff;font-size:2.5rem;margin:.1rem 0;line-height:1.1;}
.rescard .la{font-style:italic;color:#dbe6c6;font-size:.88rem;}
.pill{position:absolute;top:14px;right:14px;border-radius:99px;padding:.25rem .85rem;font-size:.8rem;font-weight:700;}
.pill.hi{background:#d4e66b;color:#273316;} .pill.mid{background:#f6dc8a;color:#5d4605;} .pill.lo{background:#ffb59c;color:#5a1d0b;}
.pb{display:flex;align-items:center;gap:.7rem;margin:.5rem 0;font-size:.88rem;}
.pb .th{width:34px;height:34px;border-radius:50%;flex:none;background-size:cover;background-position:center;background-color:#3e4d2c;border:1px solid var(--line);}
.pb .l{width:78px;flex:none;font-weight:600;}
.pb .t{flex:1;background:rgba(255,255,255,.12);border-radius:9px;height:10px;overflow:hidden;}
.pb .f{height:100%;background:linear-gradient(90deg,#829f36,#d5eb61);border-radius:9px;}
.pb .v{width:46px;text-align:right;font-weight:700;font-variant-numeric:tabular-nums;}
.strip{margin:.5rem 0 .95rem;} .strip .sl{font-size:.8rem;font-weight:600;margin-bottom:.35rem;}
.strip .out{color:var(--cl);} .strip .ok{color:var(--fo);}
.strip .tr{position:relative;height:9px;background:var(--line);border-radius:9px;}
.strip .rg{position:absolute;top:0;height:9px;background:var(--mo);opacity:.75;border-radius:9px;}
.strip .md{position:absolute;top:-3px;width:2px;height:15px;background:var(--ink);}
.strip .dot{position:absolute;top:-2px;width:13px;height:13px;margin-left:-6px;border-radius:50%;border:2px solid #11180f;}
.strip .dot.bad{background:var(--cl);} .strip .dot.good{background:var(--fo);}
.strip .ax{display:flex;justify-content:space-between;font-size:.68rem;color:var(--mut);margin-top:.25rem;}
.adv{border-left:3px solid var(--cl);background:rgba(255,150,110,.14);border-radius:8px;padding:.65rem .9rem;margin:.45rem 0;font-size:.86rem;}
.adv.okk{border-left-color:var(--fo);background:rgba(214,236,101,.12);}
.banner{position:relative;border-radius:26px;overflow:hidden;min-height:300px;background-size:cover;background-position:center;margin-bottom:1rem;border:1px solid var(--line);}
.banner .ov{position:absolute;inset:0;background:linear-gradient(90deg,rgba(10,18,8,.9) 0%,rgba(10,18,8,.4) 70%,rgba(10,18,8,.1) 100%);}
.banner .tx{position:relative;padding:2.2rem 2.4rem;color:#fff;max-width:640px;}
.banner h1{color:#fff;margin:.3rem 0 0;font-size:clamp(34px,5vw,52px);} .banner .la{font-style:italic;color:#dbe6c6;}
.chip{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);border-radius:99px;padding:.15rem .7rem;font-size:.76rem;margin:.6rem .35rem .4rem 0;}
.banner p{color:#e6efdb;margin:.2rem 0 0;}
.info{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--mo);border-radius:14px;padding:.9rem 1rem;height:100%;font-size:.88rem;line-height:1.55;}
.info b{display:block;font-size:.68rem;letter-spacing:.12em;color:var(--fo);margin-bottom:.3rem;}
.sstep{display:flex;gap:.9rem;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem;margin:.5rem 0;font-size:.88rem;}
.sstep .n{font:700 1.7rem 'Bricolage Grotesque',sans-serif;color:var(--fo);line-height:1;flex:none;width:1.6rem;}
.band{background:linear-gradient(115deg,rgba(42,56,28,.94),rgba(95,111,51,.55));border:1px solid var(--line);border-radius:26px;padding:2.2rem 2rem;margin:2.2rem 0;}
.band .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;}
.band .k{border-left:2px solid var(--fo);padding-left:1rem;}
.band .k b{display:block;font:600 2.2rem 'Bricolage Grotesque',sans-serif;} .band .k span{font-size:.8rem;color:var(--mut);}
.cta{position:relative;border-radius:26px;overflow:hidden;background-size:cover;background-position:center;margin-top:2rem;border:1px solid var(--line);}
.cta .ov{position:absolute;inset:0;background:rgba(20,34,14,.78);} .cta .tx{position:relative;padding:2.4rem;}
.cta h2{margin:0 0 .3rem;font-size:2rem;} .cta p{margin:0;color:#d8e8c8;}
.kpi{border-top:3px solid var(--mo);padding-top:.6rem;} .kpi b{display:block;font:600 1.9rem 'Bricolage Grotesque',sans-serif;} .kpi span{font-size:.78rem;color:var(--mut);}
.warn{background:rgba(255,150,110,.14);border-left:3px solid var(--cl);border-radius:8px;padding:.6rem .9rem;font-size:.84rem;margin:.5rem 0;}
.dots{display:flex;gap:3px;margin:10px 0 4px;} .dots i{width:9px;height:9px;border-radius:50%;background:var(--mo);} .dots i.x{background:var(--cl);}
footer.ft{margin-top:2.4rem;padding-top:1.2rem;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:var(--mut);font-size:.85rem;}
@media (max-width:900px){.band .grid{grid-template-columns:1fr 1fr;}}
</style>
"""

# ----------------------------------------------------------------------
# ẢNH THẬT: tự tìm trong thư mục dự án (kể cả thư mục con như  crop/22 loại cây/)
# ----------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
ALIAS = {"maize": ["corn", "bap"], "kidneybeans": ["kidney", "daudo"], "pigeonpeas": ["pigeon", "dautrieu"],
         "mothbeans": ["moth", "daumo"], "mungbean": ["mung", "dauxanh"], "blackgram": ["black", "dauden"],
         "chickpea": ["dauga"], "lentil": ["daulang"], "grapes": ["grape"], "muskmelon": ["melon", "cantaloupe", "duale", "dualuoi"],
         "watermelon": ["duahau"], "pomegranate": ["luu"], "rice": ["gao", "paddy", "luanuoc"], "orange": ["cam"], "cotton": ["caybong", "bong"]}
HERO_NAMES = {"anhnen", "nen", "background", "hero", "bia"}


def norm(s):
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def find_images():
    files = {}
    for root, dirs, fns in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in (".venv", "venv", "__pycache__", ".git", "node_modules", "cache_models")]
        for fn in fns:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXT:
                files.setdefault(norm(stem), os.path.join(root, fn))
    hero = next((files[n] for n in HERO_NAMES if n in files), None)
    pool = {k: v for k, v in files.items() if k not in HERO_NAMES}
    out = {}
    for k, v in CROPS.items():
        cands = [norm(k), norm(v[0])] + [norm(a) for a in ALIAS.get(k, [])]
        hit = next((pool[c] for c in cands if c in pool), None)
        if not hit:
            hit = next((p for st_, p in pool.items() for c in cands if len(c) >= 4 and c in st_), None)
        if hit:
            out[k] = hit
    return out, hero


@st.cache_data(show_spinner=False)
def photo(path, w, h):
    from PIL import Image, ImageOps
    im = ImageOps.fit(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (w, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


IMGS, HERO_IMG = find_images()
FALLBACK = "background-image:linear-gradient(135deg,#355a3d,#86a566)"


def bg(k, w=640, h=480):
    p = IMGS.get(k)
    if p:
        try:
            return f"background-image:url('{photo(p, w, h)}')"
        except Exception:
            pass
    return FALLBACK


def bg_hero(w=1500, h=620):
    if HERO_IMG:
        try:
            return f"background-image:url('{photo(HERO_IMG, w, h)}')"
        except Exception:
            pass
    return bg(next(iter(IMGS), None), w, h) if IMGS else FALLBACK


def H(html):
    st.markdown(html, unsafe_allow_html=True)


def vn(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def pc(x, d=1):
    return vn(x * 100, d) + "%"


vi = lambda k: CROPS[k][0]
NAME2KEY = {v[0]: k for k, v in CROPS.items()}

# ----------------------------------------------------------------------
# DỮ LIỆU + MÔ HÌNH
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return M.doc_du_lieu()


@st.cache_resource(show_spinner=False)
def load_stats():
    tr, _ = M.doc_du_lieu()
    q = tr.groupby(TARGET)[FEATS].quantile([0.1, 0.5, 0.9])
    rng = {k: {f: (float(q.loc[(k, 0.1), f]), float(q.loc[(k, 0.5), f]), float(q.loc[(k, 0.9), f])) for f in FEATS} for k in tr[TARGET].unique()}
    ax = {f: (float(np.floor(tr[f].min())), float(np.ceil(tr[f].max()))) for f in FEATS}
    return {"rng": rng, "ax": ax, "med": tr[FEATS].median()}


@st.cache_resource(show_spinner="Đang chuẩn bị mô hình (lần đầu 20–60 giây; các lần sau nạp từ thư mục cache_models)...")
def get_model(mode, cols):
    tr, _ = load_data()
    return M.lay_mo_hinh(tr, M.MODES[mode], list(cols))


@st.cache_data(show_spinner=False)
def get_eval(mode, cols):
    _, te = load_data()
    return M.danh_gia(get_model(mode, cols), te[list(cols)], te[TARGET].to_numpy())


@st.cache_data(show_spinner="Đang đo độ bền trên test bị làm lệch...")
def get_robust(mode):
    _, te = load_data()
    return M.do_ben(get_model(mode, tuple(FEATS)), te)


@st.cache_data(show_spinner="Đang tính độ quan trọng (hoán vị)...")
def get_importance(mode):
    _, te = load_data()
    return M.quan_trong_hoan_vi(get_model(mode, tuple(FEATS)), te)


try:
    TR, TE = load_data()
    ST = load_stats()
except Exception as e:
    st.error(f"Không đọc được train.csv / test.csv cạnh app.py: {e}")
    st.stop()

S.setdefault("mode", M.DEFAULT_MODE)
S.setdefault("page", "Trang chủ")
S.setdefault("last_crop", "rice")
S.setdefault("vals", {f: float(round(ST["med"][f], 1)) for f in FEATS})
S.setdefault("use", {f: True for f in FEATS})
MODE = S.mode
FULL = tuple(FEATS)


# ----------------------------------------------------------------------
# THÀNH PHẦN GIAO DIỆN
# ----------------------------------------------------------------------
def go(page, crop=None):
    S.page = page
    if crop:
        S.crop = crop
        S.last_crop = crop


def crop_card(k, prefix, label="Xem cẩm nang →", page="Cẩm nang"):
    n, la, g, days, _ = CROPS[k]
    R = ST["rng"][k]
    H(f"<div class='pcard' style=\"{bg(k)}\"><div class='ov'></div><span class='tg'>{g}</span><div class='tx'>"
      f"<div class='nm'>{n}</div><div class='la'>{la}</div><div class='mini'><i>{days}</i><i>{R['temperature'][0]:.0f}–{R['temperature'][2]:.0f} °C</i>"
      f"<i>pH {vn(R['pH'][1])}</i></div></div></div>")
    st.button(label, key=f"{prefix}_{k}", on_click=go, args=(page, k))


def strip(f, k, v=None):
    lo, hi = ST["ax"][f]
    a, m, b = ST["rng"][k][f]
    p = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))
    tag, mark = "", ""
    if v is not None:
        out = v < a or v > b
        tag = "<span class='out'>(NGOÀI khoảng điển hình)</span>" if out else "<span class='ok'>(trong khoảng điển hình)</span>"
        mark = f"<i class='dot {'bad' if out else 'good'}' style='left:{p(v):.1f}%'></i>"
    return (f"<div class='strip'><div class='sl'>{LBL[f]} {tag}</div><div class='tr'>"
            f"<b class='rg' style='left:{p(a):.1f}%;width:{max(p(b) - p(a), 1):.1f}%'></b>"
            f"<b class='md' style='left:{p(m):.1f}%'></b>{mark}</div>"
            f"<div class='ax'><span>{lo:g}</span><span>{hi:g}</span></div></div>")


def compare(k, vals):
    feats = list(vals) if vals else FEATS
    H("".join(strip(f, k, vals.get(f) if vals else None) for f in feats))
    st.caption("Dải xanh: khoảng điển hình (10%–90% mẫu của cây này) · vạch trắng: trung vị" + (" · chấm tròn: giá trị của bạn." if vals else "."))
    if not vals:
        return
    st.markdown(f"##### Cần điều chỉnh gì để trồng {vi(k).lower()}?")
    n_bad = 0
    for f, v in vals.items():
        a, _, b = ST["rng"][k][f]
        if v < a or v > b:
            n_bad += 1
            H(f"<div class='adv'><b>{SHORT[f]} = {vn(v)}</b> (khoảng điển hình {vn(a)} – {vn(b)}). {ADV[f][0 if v < a else 1]}</div>")
    if not n_bad:
        H(f"<div class='adv okk'>Tất cả chỉ số bạn nhập đều nằm trong khoảng điển hình của {vi(k).lower()}.</div>")


def bars_html(items, hi=None):
    return "".join(f"<div class='pb'><span class='l' style='width:190px'>{t}</span><span class='t'><div class='f' style='width:{v * 100:.1f}%;"
                   f"{'' if i != hi else 'filter:brightness(1.25)'}'></div></span><span class='v'>{pc(v)}</span></div>" for i, (t, v) in enumerate(items))


def faq(items):
    for q, a in items:
        with st.expander(q):
            st.write(a)


def used_vals():
    return {f: S.vals[f] for f in FEATS if S.use.get(f)}


# ----------------------------------------------------------------------
# ĐIỀU HƯỚNG
# ----------------------------------------------------------------------
PAGES = ["Trang chủ", "Gợi ý", "Hỏi ngược", "Thư viện cây", "Phòng thí nghiệm", "Cẩm nang", "Về hệ thống"]
H(CSS)
c1, c2 = st.columns([1, 5.2])
c1.markdown("<div class='logo'><i></i>Đất &amp; Cây</div>", unsafe_allow_html=True)
c2.radio("Điều hướng", PAGES, key="page", horizontal=True, label_visibility="collapsed")
H("<div class='navline'></div>")
page = S.page
HERO_KEYS = [k for k in ["rice", "coffee", "banana", "mango", "orange", "cotton"] if k in IMGS] or list(IMGS)[:4]

# ======================================================================
# TRANG CHỦ
# ======================================================================
if page == "Trang chủ":
    EF = get_eval(MODE, FULL)
    n_all = f"{len(TR) + len(TE):,}".replace(",", ".")
    H(f"<div class='hero' style=\"{bg_hero()}\"><div class='ov'></div><div class='tx'><span class='eyebrow'>Khu vườn của bạn</span>"
      "<h1>Chọn đúng cây,<br><em>trúng mùa vàng.</em></h1><p>Nhập chỉ số đất (N, P, K, pH) và khí hậu (nhiệt độ, độ ẩm, lượng mưa). Mô hình Random Forest gợi ý "
      "loại cây hợp nhất và cho biết đất của bạn còn thiếu gì.</p><div class='hs'>"
      f"<div><b>{n_all}</b><span>mẫu dữ liệu</span></div><div><b>{len(CROPS)}</b><span>loại cây</span></div>"
      f"<div><b>{pc(EF['acc'])}</b><span>đúng gợi ý số 1 (test)</span></div></div></div></div>")
    b1, b2, _ = st.columns([1.2, 1.2, 5])
    b1.button("Bắt đầu gợi ý", type="primary", on_click=go, args=("Gợi ý",))
    b2.button("Xem thư viện cây", on_click=go, args=("Thư viện cây",))
    H("<div class='center'><span class='eyebrow'>Đơn giản</span><div class='sec'>Ba bước để biết nên trồng gì</div>"
      "<div class='sub'>Không cần chuyên môn, chỉ cần các chỉ số trong phiếu phân tích đất</div></div>")
    for col, (n, t, d) in zip(st.columns(3, gap="large"), [
            ("01", "Nhập chỉ số", "Kéo thanh trượt hoặc gõ số cho N, P, K, pH và khí hậu. Bỏ tick chỉ số nào bạn chưa đo."),
            ("02", "Nhận gợi ý", "100 cây quyết định bỏ phiếu, trả về 5 loại cây hợp nhất kèm xác suất và ảnh thật."),
            ("03", "Xem cách trồng", "Mỗi cây có cẩm nang: mùa vụ, quy trình, và đất của bạn cần chỉnh gì.")]):
        col.markdown(f"<div class='step'><div class='n'>{n}</div><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)
    H("<div class='center'><span class='eyebrow'>Thư viện cây</span><div class='sec'>Những cây được quan tâm</div><div class='sub'>Bấm vào cây để xem cẩm nang chi tiết</div></div>")
    for col, k in zip(st.columns(4), HERO_KEYS[:4]):
        with col:
            crop_card(k, "home")
    H("<div class='band'><div style='text-align:center'><span class='eyebrow'>Vì sao đáng tin</span></div>"
      f"<div class='sec'>Kiểm tra trên {len(TE)} mẫu chưa từng thấy</div><div class='sub'>Mô hình tự cài đặt từ công thức, tập test tách riêng</div><div class='grid'>"
      + "".join(f"<div class='k'><b>{v}</b><span>{t}</span></div>" for v, t in
                [(pc(EF["acc"]), "đúng ngay gợi ý số 1"), (pc(EF["top3"], 0), "đúng trong 3 gợi ý đầu"), (pc(NB_B10[7][1]), "nếu chỉ dùng chỉ số đất (notebook)"), ("100", "cây quyết định bỏ phiếu")])
      + "</div></div>")
    faq([("Tôi cần những chỉ số nào?", "N, P, K, pH của đất (phiếu phân tích đất) và nhiệt độ, độ ẩm, lượng mưa trung bình nơi trồng. Thiếu chỉ số vẫn dùng được nhưng kém chính xác hơn."),
         ("Vì sao có nhiều gợi ý chứ không một?", "Nhiều cây có điều kiện sống gần nhau. Xem top-3 giúp bạn có thêm lựa chọn khi cây số 1 chưa chắc."),
         ("Kết quả có thay thế kỹ sư nông nghiệp không?", "Không. Đây là công cụ tham khảo dựa trên dữ liệu mô phỏng; hãy kết hợp kinh nghiệm địa phương.")])
    H(f"<div class='cta' style=\"{bg(HERO_KEYS[-1] if HERO_KEYS else None, 1500, 420)}\"><div class='ov'></div><div class='tx'>"
      "<h2>Sẵn sàng chọn cây cho mảnh đất của bạn?</h2><p>Chỉ mất khoảng một phút.</p></div></div>")
    st.write("")
    st.button("Thử ngay", type="primary", on_click=go, args=("Gợi ý",), key="cta")

# ======================================================================
# GỢI Ý
# ======================================================================
elif page == "Gợi ý":
    for f in FEATS:
        lo, hi = ST["ax"][f]
        v = min(max(S.vals[f], lo), hi)
        S.setdefault("s_" + f, v)
        S.setdefault("n_" + f, v)
        S.setdefault("u_" + f, S.use[f])

    def from_slider(f):
        S["n_" + f] = S["s_" + f]

    def from_number(f):
        S["s_" + f] = S["n_" + f]

    def quick():
        c = S.quick
        if c in NAME2KEY:
            for f in FEATS:
                lo, hi = ST["ax"][f]
                val = min(max(float(round(ST["rng"][NAME2KEY[c]][f][1], 1)), lo), hi)
                S["s_" + f] = S["n_" + f] = val

    def set_mode():
        S.mode = S.mode_w

    H("<div class='ph'><h1 style='font-size:clamp(28px,4vw,42px)'>Gợi ý cây theo <em>đất và khí hậu</em></h1></div>")
    st.caption("Bỏ tick ở chỉ số bạn chưa có (cần ít nhất 2). Kết quả tự cập nhật khi bạn đổi giá trị.")
    left, right = st.columns([1, 1.45], gap="large")
    with left:
        st.markdown("#### Chỉ số của bạn")
        st.selectbox("Điền nhanh bằng đất điển hình của một cây (tùy chọn)", ["— Tự nhập —"] + sorted(NAME2KEY), key="quick", on_change=quick)
        for f in FEATS:
            lo, hi = ST["ax"][f]
            stp = LIMIT[f][2]
            with st.container(border=True):
                st.checkbox(LBL[f], key="u_" + f)
                sc, nc = st.columns([1.35, 1])
                sc.slider("Kéo để chọn", float(lo), float(hi), step=float(stp), key="s_" + f, on_change=from_slider, args=(f,), disabled=not S["u_" + f])
                nc.number_input("Nhập số", float(lo), float(hi), step=float(stp), key="n_" + f, format="%.1f", on_change=from_number, args=(f,), disabled=not S["u_" + f])
        S.vals = {f: float(S["n_" + f]) for f in FEATS}
        S.use = {f: bool(S["u_" + f]) for f in FEATS}
        with st.expander("Tùy chọn mô hình"):
            st.selectbox("Cách huấn luyện", list(M.MODES), index=list(M.MODES).index(MODE), key="mode_w", on_change=set_mode)
            st.caption("Học thêm nhiễu giúp mô hình bớt \"giòn\" khi số đo lệch, đổi lại độ chính xác trên số sạch giảm nhẹ (xem Phòng thí nghiệm). "
                       "Mô hình lần đầu được huấn luyện mất 20–60 giây rồi lưu vào thư mục cache_models.")
        H("<div class='note'>Dữ liệu không ghi đơn vị của N, P, K và mốc thời gian của lượng mưa, nên số nhập chỉ có nghĩa khi cùng thang với dữ liệu huấn luyện.</div>")

    cols = [f for f in FEATS if S.use[f]]
    with right:
        if len(cols) < 2:
            H("<div class='note w'>Hãy chọn ít nhất 2 chỉ số. Một chỉ số đơn lẻ cho kết quả rất kém.</div>")
            st.stop()
        model = get_model(MODE, tuple(cols))
        vals = used_vals()
        x = pd.DataFrame([[vals[f] for f in cols]], columns=cols)
        pr = model.predict_proba(x)[0]
        order = np.argsort(-pr)
        top = [(model.classes_[i], float(pr[i])) for i in order]
        k1, p1 = top[0]
        lvl = ("hi", "Độ tin cậy cao") if p1 >= 0.7 else ("mid", "Độ tin cậy trung bình") if p1 >= 0.4 else ("lo", "Độ tin cậy thấp")
        H(f'<div class="rescard" style="{bg(k1, 900, 600)}"><div class="ov"></div><span class="pill {lvl[0]}">{pc(p1, 0)} · {lvl[1]}</span>'
          f"<div class='tx'><div class='rk'>GỢI Ý SỐ 1</div><h2>{vi(k1)}</h2><div class='la'>{CROPS[k1][1]} · {CROPS[k1][2]}</div></div></div>")
        st.write("")
        bt1, bt2 = st.columns(2)
        bt1.button(f"Cẩm nang {vi(k1).lower()}", type="primary", on_click=go, args=("Cẩm nang", k1), key="r1")
        bt2.button("So với đất của bạn (Hỏi ngược)", on_click=go, args=("Hỏi ngược", k1), key="r1b")
        c2_, c3_ = st.columns(2)
        for col, (k, p), rk in zip((c2_, c3_), top[1:3], (2, 3)):
            with col:
                H(f'<div class="pcard sm" style="{bg(k, 480, 300)}"><div class="ov"></div><span class="pp">{pc(p, 0)}</span>'
                  f"<div class='tx'><div class='la'>HẠNG {rk}</div><div class='nm'>{vi(k)}</div></div></div>")
                st.button("Xem cẩm nang", on_click=go, args=("Cẩm nang", k), key=f"r{rk}")
        for f in cols:
            lo, hi = ST["ax"][f]
            if not lo <= vals[f] <= hi:
                H(f"<div class='warn'>{SHORT[f]} = {vn(vals[f])} nằm NGOÀI khoảng dữ liệu huấn luyện [{lo:g}, {hi:g}], kết quả có thể không đáng tin.</div>")
        if p1 < 0.5:
            H("<div class='warn'>Mô hình chưa chắc: nhiều loại cây có điều kiện tương tự. Hãy xem cả top-3.</div>")

        acc = ACC_SUBSET.get(frozenset(cols))
        miss = next((f for f in MEASURE_ORDER if f not in cols), None)
        h = (f"<div class='note'>Với bộ chỉ số này, mô hình đoán đúng gợi ý số 1 khoảng <b>{acc * 100:.0f}%</b> trên dữ liệu thử (notebook, số sạch).</div>"
             if acc else f"<div class='note'>Bạn nhập {len(cols)}/7 chỉ số. Càng ít chỉ số thì mô hình càng kém chắc (xem Phòng thí nghiệm).</div>")
        h += (f"<div class='note'><b>Nên đo thêm: {SHORT[miss]}.</b>" if miss else "<div class='note'><b>Bạn đã nhập đủ 7 chỉ số.</b>")
        if k1 in CONFUSED:
            h += f" <b>{vi(k1)}</b> hay bị nhầm với <b>{vi(CONFUSED[k1])}</b>, hãy xem cả hai."
        H(h + "</div>")
        rr = np.random.default_rng(7)
        W = pd.DataFrame([[vals[f] * rr.uniform(0.9, 1.1) for f in cols] for _ in range(20)], columns=cols)
        if "humidity" in cols:
            W["humidity"] = W["humidity"].clip(upper=100)
        keep = model.classes_[np.argmax(model.predict_proba(W), axis=1)] == k1
        H("<div class='dots'>" + "".join(f"<i class='{'' if ok else 'x'}'></i>" for ok in keep) + f"</div><small style='color:var(--mut)'>Làm lệch số đo ±10%: gợi ý số 1 giữ nguyên {int(keep.sum())}/20 lần.</small>")
    st.write("")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.markdown("##### Xác suất của 5 cây đứng đầu")
        H("".join(f'<div class="pb"><span class="th" style="{bg(k, 60, 60)}"></span><span class="l">{vi(k)}</span><span class="t"><div class="f" style="width:{p * 100:.1f}%"></div></span><span class="v">{pc(p, 0)}</span></div>' for k, p in top[:5]))
        st.caption("Đây là xác suất do mô hình ước lượng, không phải tỉ lệ thành công ngoài thực tế.")
    with b:
        st.markdown(f"##### Số liệu của bạn so với điều kiện điển hình của {vi(k1).lower()}")
        compare(k1, vals)

# ======================================================================
# HỎI NGƯỢC
# ======================================================================
elif page == "Hỏi ngược":
    H("<div class='ph'><h1 style='font-size:clamp(28px,4vw,42px)'>Tôi muốn trồng một cây, <em>đất thiếu gì?</em></h1></div>")
    st.caption("Dùng số bạn đã nhập ở trang Gợi ý (chỉ các chỉ số còn được tick). Dải xanh là khoảng cây ưa, chấm tròn là giá trị của bạn.")
    S.setdefault("crop", S.last_crop)
    k = st.selectbox("Chọn cây", list(CROPS), key="crop", format_func=vi)
    S.last_crop = k
    vals = used_vals()
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        H(f"<div class='pcard' style=\"{bg(k, 800, 520)}\"><div class='ov'></div><span class='tg'>{CROPS[k][2]}</span><div class='tx'><div class='nm'>{vi(k)}</div><div class='la'>{CROPS[k][1]}</div></div></div>")
        H(f"<div class='note'>{CROPS[k][4]} Bạn đang nhập {len(vals)}/7 chỉ số.</div>")
        st.button(f"Cẩm nang {vi(k).lower()}", type="primary", on_click=go, args=("Cẩm nang", k), key="hn_cn")
        st.button("Sửa số đo ở trang Gợi ý", on_click=go, args=("Gợi ý",), key="hn_gy")
    with right:
        compare(k, vals)

# ======================================================================
# THƯ VIỆN CÂY
# ======================================================================
elif page == "Thư viện cây":
    H(f"<div class='ph'><h1 style='font-size:clamp(28px,4vw,42px)'>{len(CROPS)} loại cây <em>trong dữ liệu</em></h1></div>")
    st.caption("Điều kiện điển hình (trung vị) của từng cây. Bấm vào cây để xem cẩm nang.")
    f1_, f2_ = st.columns([1, 3])
    q = f1_.text_input("Tìm theo tên", placeholder="Ví dụ: chuối, banana...")
    g = f2_.radio("Nhóm cây", ["Tất cả"] + GROUPS, horizontal=True)
    keys = sorted([k for k, v in CROPS.items() if (g == "Tất cả" or v[2] == g) and (not q or norm(q) in norm(v[0] + v[1] + k))], key=vi)
    st.write(f"Tìm thấy {len(keys)} cây.")
    for i in range(0, len(keys), 4):
        for col, k in zip(st.columns(4), keys[i:i + 4]):
            with col:
                crop_card(k, "lib")
    miss = [f"{k} ({vi(k)})" for k in CROPS if k not in IMGS]
    if miss:
        st.warning("Chưa tìm thấy ảnh cho: " + ", ".join(miss) + ". Đặt ảnh trong thư mục dự án (vd: crop/22 loại cây/) với tên tiếng Việt, ví dụ chuối.jfif, dưa hấu.jfif.")

# ======================================================================
# PHÒNG THÍ NGHIỆM
# ======================================================================
elif page == "Phòng thí nghiệm":
    H("<div class='ph'><h1 style='font-size:clamp(28px,4vw,42px)'>Phòng thí nghiệm <em>mô hình</em></h1></div>")
    st.caption(f"Số liệu đo trên {len(TE)} mẫu không dùng khi huấn luyện. Dữ liệu có dấu hiệu là dữ liệu mô phỏng. Nhiễu là giả lập của nhóm, không phải sai số đo thật.")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("##### Độ chính xác theo bộ chỉ số")
            st.caption("Accuracy trên tập test (notebook 04b, mục 12).")
            H(bars_html(NB_B10, hi=9))
    with b:
        with st.container(border=True):
            st.markdown("##### Nếu số đo bị lệch thì sao?")
            lvl = st.radio("Mức lệch", NB_NOISE, horizontal=True, key="lab_noise")
            i = NB_NOISE.index(lvl)
            H(bars_html([(t, v[i]) for t, v in NB_SUBSET], hi=3))
            st.caption("Mô hình học trên dữ liệu sạch. Ít chỉ số thì tụt nhanh hơn khi số đo lệch." if i else "Trên số sạch, càng nhiều chỉ số càng chính xác.")
    with st.container(border=True):
        st.markdown("##### Học thêm bản sao nhiễu có giúp không?")
        st.caption("Cả ba mô hình đều dùng 7 chỉ số, chỉ khác cách huấn luyện (notebook 15b.4). Học thêm nhiễu làm số sạch giảm nhẹ nhưng bớt giòn khi lệch: đó là đánh đổi.")
        st.dataframe(pd.DataFrame([[n] + [f"{x * 100:.1f}%" for x in v] for n, v in NB_TANG], columns=["Cách huấn luyện"] + NB_NOISE), hide_index=True, use_container_width=True)
    with st.container(border=True):
        st.markdown(f"##### Mô hình đang chạy trên web: {MODE}")
        rb = get_robust(MODE)
        st.dataframe(pd.DataFrame([[f"{rb[0.0][0] * 100:.1f}%"] + [f"{rb[m][0] * 100:.1f}% ± {rb[m][1] * 100:.1f}" for m in (0.05, 0.10, 0.20)]], columns=NB_NOISE, index=["Accuracy test"]), use_container_width=True)
        st.caption("Đo trực tiếp bây giờ bằng mô hình web (5 lần bốc nhiễu mỗi mức), nên có thể lệch vài phần nghìn so với bảng notebook (20 lần).")
    with st.container(border=True):
        st.markdown("##### Thêm cây có giúp không?")
        st.caption("OOB accuracy theo số cây (mô hình học dữ liệu sạch). Từ 50 cây trở đi gần như phẳng, mô hình cuối dùng 100 cây.")
        H(bars_html(NB_TREES, hi=5))

# ======================================================================
# CẨM NANG
# ======================================================================
elif page == "Cẩm nang":
    S.setdefault("crop", S.last_crop)
    k = st.selectbox("Chọn loại cây", list(CROPS), key="crop", format_func=vi)
    S.last_crop = k
    n, la, g, days, note = CROPS[k]
    R = ST["rng"][k]
    H(f"<div class='banner' style=\"{bg(k, 1500, 600)}\"><div class='ov'></div><div class='tx'><span class='eyebrow'>{g}</span>"
      f"<h1>{n}</h1><div class='la'>{la}</div><div><span class='chip'>{days}</span><span class='chip'>{R['temperature'][0]:.0f}–{R['temperature'][2]:.0f} °C</span></div><p>{note}</p></div></div>")
    i1, i2, i3 = st.columns(3)
    i1.markdown(f"<div class='info'><b>MÙA VỤ (THAM KHẢO)</b>{GROUP_SEASON[g]}</div>", unsafe_allow_html=True)
    i2.markdown(f"<div class='info'><b>NƯỚC</b>Lượng mưa điển hình {R['rainfall'][0]:.0f}–{R['rainfall'][2]:.0f} mm; độ ẩm không khí {R['humidity'][0]:.0f}–{R['humidity'][2]:.0f}%.</div>", unsafe_allow_html=True)
    i3.markdown(f"<div class='info'><b>ĐẤT NÊN</b>pH {vn(R['pH'][0])}–{vn(R['pH'][2])}; N {R['N'][0]:.0f}–{R['N'][2]:.0f}, P {R['P'][0]:.0f}–{R['P'][2]:.0f}, K {R['K'][0]:.0f}–{R['K'][2]:.0f}.</div>", unsafe_allow_html=True)
    st.markdown("### So với đất của bạn")
    st.caption("Khoảng điển hình = 10%–90% mẫu của cây trong dữ liệu. Chấm tròn là số bạn nhập ở trang Gợi ý.")
    compare(k, used_vals())
    st.markdown("### Quy trình trồng")
    for i, (t, d) in enumerate(STEPS_PERENNIAL if k in PERENNIAL else STEPS_ANNUAL, 1):
        H(f"<div class='sstep'><span class='n'>{i}</span><div><b>{t}</b><br>{d}</div></div>")
    st.caption("Quy trình mang tính tổng quát, không thay thế hướng dẫn kỹ thuật của địa phương.")

# ======================================================================
# VỀ HỆ THỐNG
# ======================================================================
else:
    EF = get_eval(MODE, FULL)
    imp = get_importance(MODE)
    H("<div class='ph'><h1 style='font-size:clamp(28px,4vw,42px)'>Đáng tin <em>đến đâu?</em></h1></div>")
    st.caption(f"Đo trên {len(TE)} mẫu không dùng khi huấn luyện (Random Forest 100 cây, 7 chỉ số, {MODE.lower()}). Dữ liệu có dấu hiệu là dữ liệu mô phỏng nên các số này chưa nói gì về ruộng thật.")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("##### Kết quả trên tập test")
            kp = [(pc(EF["acc"]), "đúng gợi ý số 1"), (pc(EF["top3"], 0), "đúng trong 3 gợi ý đầu"), (pc(EF["macro"]), "Macro F1"), (f"{EF['sai']} / {EF['n']}", "số mẫu sai")]
            k1_, k2_ = st.columns(2)
            for i, (v, t) in enumerate(kp):
                (k1_ if i % 2 == 0 else k2_).markdown(f"<div class='kpi'><b>{v}</b><span>{t}</span></div><br>", unsafe_allow_html=True)
            st.caption("Với mô hình học thêm nhiễu, không báo OOB/CV vì bản gốc và bản sao của cùng một dòng có thể nằm hai phía, con số sẽ cao giả. Chỉ báo kết quả trên test.")
    with b:
        with st.container(border=True):
            st.markdown("##### Chỉ số nào quan trọng?")
            st.caption("Độ giảm Accuracy khi xáo trộn một chỉ số trên test (độ quan trọng hoán vị).")
            ordered = sorted(imp.items(), key=lambda t: -t[1])
            mx = max(ordered[0][1], 1e-9)
            H("".join(f"<div class='pb'><span class='l' style='width:90px'>{SHORT[f]}</span><span class='t'><div class='f' style='width:{max(v, 0) / mx * 100:.0f}%'></div></span><span class='v'>{v * 100:.1f}</span></div>" for f, v in ordered))
            H("<div class='note'>Nhiệt độ và pH gần như không đóng góp cho dự đoán ở bộ dữ liệu này, nhưng pH vẫn có ý nghĩa nông học.</div>")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("##### Mô hình nói chắc X% thì có đúng X% không?")
            st.caption("Hiệu chỉnh xác suất trên test, ECE 0,022 (notebook 9b, mô hình học dữ liệu sạch).")
            st.dataframe(pd.DataFrame(NB_CALIB, columns=["Mức tự tin", "Số mẫu", "Tự tin TB", "Đúng thật"]), hide_index=True, use_container_width=True)
            H("<div class='note'>Mô hình hơi dè dặt: đúng nhiều hơn mức nó nói. Số này chỉ đúng với dữ liệu này.</div>")
    with b:
        with st.container(border=True):
            st.markdown("##### Giới hạn cần biết")
            st.markdown(f"- Chỉ chọn trong {len(CROPS)} loại cây có trong dữ liệu.\n- Số % không phải tỉ lệ thành công ngoài thực tế.\n"
                        "- Dữ liệu không ghi đơn vị của N, P, K và mốc thời gian của lượng mưa.\n"
                        "- Số đo lệch 10% làm Accuracy của mô hình học sạch tụt từ 99,3% xuống 94,4% (nhiễu giả lập); học thêm nhiễu giảm bớt nhưng không loại bỏ.\n"
                        "- Chỉ dùng chỉ số đất thì đúng khoảng 79%; có khí hậu mới lên trên 99%.")
    faq([("Vì sao chọn Random Forest?", "Nhiều cây quyết định bỏ phiếu cho kết quả ổn định hơn một cây đơn, xử lý tốt dữ liệu số và cho được tỉ lệ tin cậy."),
         ("Hệ thống có học thêm từ dữ liệu tôi nhập không?", "Không. Mô hình được huấn luyện sẵn từ train.csv; số bạn nhập chỉ dùng để dự đoán."),
         ("Bản sao nhiễu lấy từ đâu?", "Tạo trong bộ nhớ khi huấn luyện: mỗi chỉ số nhân hệ số ngẫu nhiên trong [1-a, 1+a], a = 10% hoặc 20%. File train.csv và test.csv không bị sửa."),
         ("Tôi nhập giá trị ngoài khoảng thì sao?", "Hệ thống vẫn dự đoán nhưng hiện cảnh báo vì mô hình chưa từng gặp giá trị như vậy.")])

H("<footer class='ft'><span>Đất &amp; Cây · Đề tài hệ thống hỗ trợ lựa chọn cây trồng dựa trên điều kiện đất</span><span>Đại học Kinh tế Đà Nẵng</span></footer>")