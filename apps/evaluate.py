"""BioSense AI evaluation UI. Run: python -m streamlit run apps/evaluate.py.

Training/test CSV files are supplied locally by an authorized user.
No competition data or fitted model is distributed with this source.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    fbeta_score, f1_score, roc_auc_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, precision_score, recall_score,
    average_precision_score, roc_curve, precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Opsiyonel ML kütüphaneleri ──
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
try:
    from catboost import CatBoostClassifier
    CB_OK = True
except ImportError:
    CB_OK = False

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ═══════════════════════════════════════════════════════════════════
#  SAYFA AYARLARI
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BioSense AI — Test Arayüzü",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
    .metrik-kart {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8c 100%);
        padding: 18px; border-radius: 12px; color: white; text-align: center;
        margin: 6px 0;
    }
    .metrik-deger { font-size: 30px; font-weight: 700; }
    .metrik-isim { font-size: 13px; opacity: 0.85; }
    .ci-text { font-size: 12px; opacity: 0.75; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 BioSense AI — Model Eğitim & Test Arayüzü")
st.caption("TEKNOFEST Değerlendirme Aracı · v7.9 entegrasyonu · "
           "Bağımsız test seti ile objektif performans ölçümü · "
           "JSON'dan otomatik parametre + SHAP özellik + ensemble üyesi")


# ═══════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR (ana pipeline ile tutarlı)
# ═══════════════════════════════════════════════════════════════════
def f2_binary(y_true, y_pred):
    """F2 skoru — binary, pozitif sınıf = 1 (patojenik)."""
    return fbeta_score(y_true, y_pred, beta=2, pos_label=1,
                       average="binary", zero_division=0)


def en_iyi_f2_esik(y_true, proba, adim=0.01, tolerans=0.005):
    """OOF/train üzerinde en iyi F2 eşiği — MCC tie-break ile.
    NOT: Bu eşik TRAIN/OOF'ta seçilir, TEST'e uygulanır (sızıntı yok)."""
    f2_dict = {}
    for e in np.arange(0.05, 0.95 + adim, adim):
        e = round(float(e), 4)
        pred = (proba >= e).astype(int)
        f2_dict[e] = f2_binary(y_true, pred)
    max_f2 = max(f2_dict.values())
    adaylar = [e for e, f in f2_dict.items() if f >= (max_f2 - tolerans)]
    if len(adaylar) == 1:
        return adaylar[0]
    # MCC tie-break
    en_iyi_mcc, en_iyi_esik = -2.0, adaylar[0]
    for e in adaylar:
        pred = (proba >= e).astype(int)
        try:
            mcc = matthews_corrcoef(y_true, pred)
        except Exception:
            mcc = -2.0
        if mcc > en_iyi_mcc:
            en_iyi_mcc, en_iyi_esik = mcc, e
    return en_iyi_esik


def stratified_bootstrap_ci(y_true, proba, threshold, metric="f2",
                            n_boot=2000, alpha=0.05, seed=43):
    """[v7.7] STRATIFIED bootstrap %95 CI — imbalanced data için.
    Her sınıftan orantılı resample → CI bozulmaz."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        # Tek sınıf — düz resample
        n = len(y_true)
        idx = rng.integers(0, n, (n_boot, n))
    else:
        boot_pos = rng.choice(pos_idx, (n_boot, n_pos), replace=True)
        boot_neg = rng.choice(neg_idx, (n_boot, n_neg), replace=True)
        idx = np.concatenate([boot_pos, boot_neg], axis=1)

    y_t = y_true[idx]
    y_p = (proba[idx] >= threshold).astype(int)

    TP = np.sum((y_t == 1) & (y_p == 1), axis=1).astype(np.float64)
    FP = np.sum((y_t == 0) & (y_p == 1), axis=1).astype(np.float64)
    FN = np.sum((y_t == 1) & (y_p == 0), axis=1).astype(np.float64)
    TN = np.sum((y_t == 0) & (y_p == 0), axis=1).astype(np.float64)

    if metric == "f2":
        numer = 5.0 * TP
        denom = 5.0 * TP + 4.0 * FN + FP
        skor = np.where(denom > 0, numer / denom, 0.0)
    elif metric == "f1":
        numer = 2.0 * TP
        denom = 2.0 * TP + FN + FP
        skor = np.where(denom > 0, numer / denom, 0.0)
    elif metric == "mcc":
        num = TP * TN - FP * FN
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        skor = np.where(den > 0, num / den, 0.0)
    else:
        skor = np.zeros(n_boot)

    return (float(np.mean(skor)),
            float(np.percentile(skor, 100 * alpha / 2)),
            float(np.percentile(skor, 100 * (1 - alpha / 2))))


def kategorik_isle(X_train, X_test):
    """[v7.7] Kategorik kolonları TRAIN→TEST aynı kategori seti ile encode et."""
    kat_kolonlar = []
    kat_kategoriler = {}
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in X_train.columns:
        if (X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
                or str(col).startswith("CAT_") or str(col).startswith("AA_")):
            X_train[col] = X_train[col].fillna("EKSİK").astype(str).astype("category")
            if "EKSİK" not in X_train[col].cat.categories:
                X_train[col] = X_train[col].cat.add_categories(["EKSİK"])
            kat_kategoriler[col] = X_train[col].cat.categories
            _test_col = X_test[col].fillna("EKSİK").astype(str).astype("category")
            _gecersiz = ~_test_col.isin(kat_kategoriler[col])
            if _gecersiz.any():
                _test_col = _test_col.astype(str)
                _test_col[_gecersiz.values] = "EKSİK"
            X_test[col] = pd.Categorical(_test_col, categories=kat_kategoriler[col])
            kat_kolonlar.append(col)
    return X_train, X_test, kat_kolonlar


def kat_idx(X, kat_kolonlar):
    """Kategorik kolon indekslerini döndür (CatBoost için)."""
    return [X.columns.get_loc(c) for c in kat_kolonlar]


def numerik_donustur(X, kat_kolonlar):
    """Tree-dışı modeller için kategorikleri sayısal kodla."""
    X = X.copy()
    for col in kat_kolonlar:
        if col in X.columns:
            X[col] = X[col].cat.codes
    return X


def model_olustur(algo, params, kat_kolonlar, X_train):
    """Verilen algoritma + parametre ile model nesnesi döndür."""
    p = dict(params)  # kopya
    if algo == "xgboost":
        p.setdefault("tree_method", "hist")
        p.setdefault("enable_categorical", True)
        p.setdefault("verbosity", 0)
        return xgb.XGBClassifier(**p)
    elif algo == "lightgbm":
        p.setdefault("verbosity", -1)
        return lgb.LGBMClassifier(**p)
    elif algo == "catboost":
        p.setdefault("verbose", False)
        p["cat_features"] = kat_idx(X_train, kat_kolonlar)
        return CatBoostClassifier(**p)
    elif algo == "random_forest":
        return RandomForestClassifier(**p)
    elif algo == "extra_trees":
        return ExtraTreesClassifier(**p)
    elif algo == "logistic":
        return LogisticRegression(**p)
    elif algo == "svm_rbf":
        p.setdefault("probability", True)
        return SVC(**p)
    elif algo == "knn":
        return KNeighborsClassifier(**p)
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo}")


def model_egit_tahmin(algo, params, X_train, y_train, X_test, kat_kolonlar):
    """Modeli eğit, test olasılıklarını döndür."""
    tree_based = algo in ("xgboost", "lightgbm", "catboost")
    if tree_based:
        Xtr, Xte = X_train, X_test
    else:
        # Tree-dışı: kategorikleri sayısala çevir
        Xtr = numerik_donustur(X_train, kat_kolonlar)
        Xte = numerik_donustur(X_test, kat_kolonlar)
        # [v7.10] Sayısal NaN doldurma — LogisticRegression/SVC/KNN NaN'ı KABUL ETMEZ
        # (örn. CFTR'de EK_3 gibi sayısal kolonlar NaN içerir → ensemble linear/kernel
        # üyeleri patlardı). Sızıntısız: impute değeri SADECE X_train'den (CV'de fold-train)
        # öğrenilir, X_test'e uygulanır. Tüm-NaN kolon için 0'a düşülür.
        if Xtr.isnull().values.any() or Xte.isnull().values.any():
            _med = Xtr.median(numeric_only=True)
            Xtr = Xtr.fillna(_med).fillna(0)
            Xte = Xte.fillna(_med).fillna(0)

    model = model_olustur(algo, params, kat_kolonlar, Xtr)

    if algo == "catboost":
        model.fit(Xtr, y_train, cat_features=kat_idx(Xtr, kat_kolonlar))
    else:
        model.fit(Xtr, y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)[:, 1]
    else:
        proba = model.decision_function(Xte)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    return proba


def tum_metrikler(y_true, proba, esik):
    """Bir olasılık vektörü + eşik için tüm TEKNOFEST metriklerini hesapla.
    Per-model ve ensemble satırlarını aynı fonksiyonla üretmek için."""
    y_true = np.asarray(y_true)
    pred = (proba >= esik).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        mcc = matthews_corrcoef(y_true, pred)
    except Exception:
        mcc = 0.0
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = 0.0
    try:
        prauc = average_precision_score(y_true, proba)
    except Exception:
        prauc = 0.0
    return {
        "F2": f2_binary(y_true, pred),
        "F1": f1_score(y_true, pred, pos_label=1, average="binary", zero_division=0),
        "MCC": mcc,
        "ROC-AUC": auc,
        "PR-AUC": prauc,
        "Recall": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "Precision": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "Specificity": spec,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════
#  KENAR ÇUBUĞU — VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════
st.sidebar.header("📂 Veri & Parametreler")

train_file = st.sidebar.file_uploader(
    "1️⃣ TRAIN CSV (model eğitimi)", type=["csv"])
test_file = st.sidebar.file_uploader(
    "2️⃣ TEST CSV (dokunulmamış %20)", type=["csv"])
json_file = st.sidebar.file_uploader(
    "3️⃣ Optimal Parametreler JSON (opsiyonel)", type=["json"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Ayarları")

hedef_kolon = st.sidebar.text_input("Hedef (label) kolonu", value="Label")
id_kolon = st.sidebar.text_input("ID kolonu (varsa)", value="Variant_ID")

mod = st.sidebar.radio(
    "Değerlendirme modu",
    ["Tek Model", "Ensemble (çoklu model + isotonic)"],
)

# Kütüphane durumu
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Kütüphane Durumu")
st.sidebar.write(f"{'✅' if XGB_OK else '❌'} XGBoost")
st.sidebar.write(f"{'✅' if LGB_OK else '❌'} LightGBM")
st.sidebar.write(f"{'✅' if CB_OK else '❌'} CatBoost")


# ═══════════════════════════════════════════════════════════════════
#  JSON PARAMETRELERİNİ OKU
# ═══════════════════════════════════════════════════════════════════
json_params = {}
if json_file is not None:
    try:
        json_params = json.load(json_file)
        st.sidebar.success("JSON parametreleri yüklendi.")
    except Exception as e:
        st.sidebar.error(f"JSON okuma hatası: {e}")


BILINEN_ALGOLAR = ["xgboost", "lightgbm", "catboost", "random_forest",
                   "extra_trees", "logistic", "svm_rbf", "knn"]


def json_den_algo_params(json_data):
    """JSON'dan algoritma → parametre sözlüğü çıkar.

    v7.9 yapısı (faz3_top_trials, pdr_v7_kopyala) ÖNCELİKLİ; eski/düz yapılar
    da geriye-uyumlu desteklenir. Bu fonksiyon olmadan v7.9 JSON'undaki
    tuned parametreler hiç yüklenmez, model default'a düşerdi.
    """
    sonuc = {}
    if not json_data:
        return sonuc
    # 1) v7.9: faz3_top_trials[algo] = [{cv_skor, params}, ...] → en iyi trial params
    ftt = json_data.get("faz3_top_trials")
    if isinstance(ftt, dict):
        for algo, trials in ftt.items():
            if isinstance(trials, list) and trials and isinstance(trials[0], dict):
                p = trials[0].get("params")
                if isinstance(p, dict):
                    sonuc[algo] = dict(p)
    # 2) v7.9: pdr_v7_kopyala = {algorithm, ...params, OPTIMAL_ESIK} → tek en iyi model
    pk = json_data.get("pdr_v7_kopyala")
    if isinstance(pk, dict) and "algorithm" in pk:
        algo = pk["algorithm"]
        p = {k: v for k, v in pk.items() if k not in ("algorithm", "OPTIMAL_ESIK")}
        sonuc.setdefault(algo, p)  # faz3_top_trials zaten verdiyse onu koru
    # 3) Eski/düz yapılar (geriye uyumluluk)
    for anahtar in ["en_iyi_parametreler", "best_params", "optimal_params",
                    "faz3_best_params", "parametreler", "final_params"]:
        if anahtar in json_data and isinstance(json_data[anahtar], dict):
            for algo, p in json_data[anahtar].items():
                if isinstance(p, dict):
                    sonuc.setdefault(algo, p)
    for algo in BILINEN_ALGOLAR:
        if algo in json_data and isinstance(json_data[algo], dict):
            sonuc.setdefault(algo, json_data[algo])
    return sonuc


def json_den_meta(json_data):
    """v7.9 JSON'dan meta bilgi çıkar: SHAP-seçili özellikler, ensemble üyeleri
    + ağırlıkları, v7.9'un optimal eşiği, pozitif sınıf etiketi."""
    meta = {"ozellikler": None, "kat_kolonlar": None,
            "ensemble_uyeleri": [], "ensemble_agirliklari": {},
            "optimal_esik": None, "pozitif_sinif": None}
    if not json_data:
        return meta
    veri = json_data.get("veri", {})
    if isinstance(veri, dict):
        if isinstance(veri.get("secilen_ozellikler"), list):
            meta["ozellikler"] = [str(c) for c in veri["secilen_ozellikler"]]
        if isinstance(veri.get("kategorik_kolonlar"), list):
            meta["kat_kolonlar"] = [str(c) for c in veri["kategorik_kolonlar"]]
        if veri.get("label_pozitif_sinif") is not None:
            meta["pozitif_sinif"] = str(veri["label_pozitif_sinif"])
    uyeler = json_data.get("ensemble_uyeleri")
    if isinstance(uyeler, list):
        for u in uyeler:
            if isinstance(u, dict) and "algoritma" in u:
                a = u["algoritma"]
                meta["ensemble_uyeleri"].append(a)
                if "agirlik" in u:
                    try:
                        meta["ensemble_agirliklari"][a] = float(u["agirlik"])
                    except Exception:
                        pass
    for k in ("ensemble_optimal_esik", "OPTIMAL_ESIK"):
        if k in json_data:
            try:
                meta["optimal_esik"] = float(json_data[k]); break
            except Exception:
                pass
    pk = json_data.get("pdr_v7_kopyala")
    if meta["optimal_esik"] is None and isinstance(pk, dict) and "OPTIMAL_ESIK" in pk:
        try:
            meta["optimal_esik"] = float(pk["OPTIMAL_ESIK"])
        except Exception:
            pass
    return meta


algo_params_json = json_den_algo_params(json_params)
json_meta = json_den_meta(json_params)


# ═══════════════════════════════════════════════════════════════════
#  ANA İŞLEM
# ═══════════════════════════════════════════════════════════════════
if train_file is not None and test_file is not None:
    # Veri yükle
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    st.subheader("📊 Veri Özeti")
    col1, col2, col3 = st.columns(3)
    col1.metric("Train satır", len(df_train))
    col2.metric("Test satır", len(df_test))
    col3.metric("Özellik sayısı",
                len([c for c in df_train.columns
                     if c not in [hedef_kolon, id_kolon]]))

    # ── Sızıntı kontrolü: ID çakışması ──
    if id_kolon in df_train.columns and id_kolon in df_test.columns:
        train_ids = set(df_train[id_kolon].astype(str))
        test_ids = set(df_test[id_kolon].astype(str))
        kesisim = train_ids & test_ids
        if kesisim:
            st.error(f"🚨 VERİ SIZINTISI: {len(kesisim)} ID train ve test'te ÇAKIŞIYOR! "
                     f"Örnek: {list(kesisim)[:5]}")
            st.stop()
        else:
            st.success(f"✅ Sızıntı kontrolü geçti: train/test ID çakışması yok "
                       f"({len(train_ids)} train, {len(test_ids)} test)")

    # ── Özellik hazırlığı ──
    ozellik = [c for c in df_train.columns if c not in [hedef_kolon, id_kolon]]
    # Sadece test'te de olan özellikler
    ozellik = [c for c in ozellik if c in df_test.columns]

    # [v7.9 UYUM] JSON'da SHAP-seçili özellik listesi varsa AYNI alt-kümeyle eğit.
    # Yoksa arayüz tüm özelliklerle eğitir; ama v7.9 parametreleri bu alt-küme
    # için ayarlandığından uyumsuz olur (overfitting/yanlış kapasite).
    if json_meta.get("ozellikler"):
        istenen = json_meta["ozellikler"]
        secili = [c for c in istenen if c in ozellik]
        eksik = [c for c in istenen if c not in ozellik]
        if secili:
            ozellik = secili
            _msg = (f"🧬 v7.9 JSON'dan **{len(secili)} SHAP-seçili özellik** "
                    f"uygulandı (pipeline ile aynı özellik seti).")
            if eksik:
                _msg += f" ⚠️ {len(eksik)} özellik bu veride yok, atlandı."
            st.success(_msg)
        else:
            st.warning("⚠️ JSON'daki seçili özellikler bu veri setinde bulunamadı — "
                       "tüm özelliklerle devam ediliyor.")

    X_train_ham = df_train[ozellik].copy()
    X_test_ham = df_test[ozellik].copy()

    # Label encode (train zorunlu; test'te Label OPSİYONEL → yoksa TAHMİN MODU)
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[hedef_kolon])
    # [v7.9 PARİTE] Etiket-kutbu guard'ı — pozitif sınıf (1) gerçekten patojenik mi?
    # LabelEncoder sınıfları ALFABETİK sıralar; pozitif yanlış tarafa düşerse TÜM
    # metrikler sessizce ters döner. (Ana pipeline biosense_ai_v7.9.py ile aynı mantık.)
    if len(le.classes_) != 2:
        st.error(f"🚨 Hedef '{hedef_kolon}' binary değil ({len(le.classes_)} sınıf): "
                 f"{list(le.classes_)}. İki sınıflı (patojenik/benign) Label gerekir.")
        st.stop()
    _c0, _c1 = str(le.classes_[0]).lower(), str(le.classes_[1]).lower()
    _poz = ("patho", "malign", "pos")
    if any(k in _c1 for k in _poz):
        pass  # pozitif sınıf (1) doğru tarafta
    elif any(k in _c0 for k in _poz):
        st.error(f"🚨 ETİKET KUTBU TERS: patojenik görünen sınıf '{le.classes_[0]}' "
                 f"class 0'a düşmüş — metrikler YANLIŞ sınıfı ölçer! CSV'de Label'ı "
                 f"düzeltin (patojenik = 1).")
        st.stop()
    elif _c0 in ("0", "1") and _c1 in ("0", "1"):
        st.warning("⚠️ Label sayısal ('0'/'1'). Konvansiyon: **1 = patojenik (pozitif)** "
                   "varsayıldı — lütfen doğrulayın; yanlışsa metrikler ters yönde olur.")
    else:
        st.warning(f"⚠️ Pozitif sınıf belirsiz: '{le.classes_[1]}' → 1 varsayıldı. "
                   f"Manuel doğrulama önerilir.")
    if hedef_kolon in df_test.columns:
        try:
            y_test = le.transform(df_test[hedef_kolon])
        except ValueError as _le_err:
            st.error(f"🚨 Test etiketleri train sınıflarıyla uyuşmuyor ({_le_err}). "
                     f"Train sınıfları: {list(le.classes_)}. Test'teki '{hedef_kolon}' "
                     f"değerlerini train ile eşitleyin.")
            st.stop()
        # Pozitif sınıf bilgisi
        st.info(f"**Label haritalama**: '{le.classes_[0]}' → 0 (negatif), "
                f"'{le.classes_[1]}' → 1 (POZİTİF, F2 hedefi). "
                f"Test pozitif oranı: {(y_test==1).mean():.1%}")
        # Naive baseline
        test_pos = (y_test == 1).mean()
        naive_f2 = 5.0 * test_pos / (4.0 * test_pos + 1.0)
        if test_pos > 0.70:
            st.warning(f"⚠️ Test seti %{test_pos*100:.0f} pozitif — F2 yanıltıcı olabilir. "
                       f"Naive 'her şey pozitif' baseline F2 = {naive_f2:.4f}. "
                       f"MCC ve Specificity'e dikkat edin.")
    else:
        y_test = None
        st.warning(f"🔮 Test setinde '{hedef_kolon}' kolonu YOK → **TAHMİN MODU**. "
                   f"Model senin train'inde eğitilip test için tahmin üretecek "
                   f"(gerçek etiket olmadığı için skor/metrik hesaplanamaz). "
                   f"Aşağıdaki butonla tahminleri CSV olarak indirebilirsin.")

    # Kategorik işle (v7.7 force)
    X_train, X_test, kat_kolonlar = kategorik_isle(X_train_ham, X_test_ham)

    # ── Algoritma seçimi ──
    mevcut_algolar = []
    if XGB_OK: mevcut_algolar.append("xgboost")
    if LGB_OK: mevcut_algolar.append("lightgbm")
    if CB_OK: mevcut_algolar.append("catboost")
    mevcut_algolar += ["random_forest", "extra_trees", "logistic", "svm_rbf", "knn"]

    st.markdown("---")
    st.subheader("🎯 Model Seçimi")

    if mod == "Tek Model":
        secili_algolar = [st.selectbox("Algoritma", mevcut_algolar)]
        ensemble_yontem = "Düz ortalama"
    else:
        if json_meta.get("ensemble_uyeleri"):
            varsayilan = [a for a in json_meta["ensemble_uyeleri"]
                          if a in mevcut_algolar]
            st.caption(f"ℹ️ v7.9'un önerdiği ensemble üyeleri ön-seçildi: "
                       f"`{', '.join(varsayilan)}`")
        else:
            varsayilan = [a for a in ["lightgbm", "catboost", "extra_trees"]
                          if a in mevcut_algolar][:3]
        secili_algolar = st.multiselect(
            "Ensemble üyeleri (2-5 algoritma seçin)",
            mevcut_algolar, default=varsayilan)
        _yontemler = ["Düz ortalama"]
        if json_meta.get("ensemble_agirliklari"):
            _yontemler.append("Ağırlıklı ortalama (v7.9 ağırlıkları)")
        _yontemler.append("Rank ortalama (sıralama-bazlı)")
        ensemble_yontem = st.radio("Ensemble birleştirme yöntemi", _yontemler,
                                   horizontal=True)

    # Karar eşiği kaynağı
    _esik_secenek = ["OOF'tan yeniden seç (F2-optimal + MCC tie-break)"]
    if json_meta.get("optimal_esik") is not None:
        _esik_secenek.append(
            f"v7.9 JSON eşiğini kullan ({json_meta['optimal_esik']:.3f})")
    _esik_secenek.append("Manuel eşik (slider ile)")
    esik_kaynak = st.radio("Karar eşiği kaynağı", _esik_secenek, horizontal=True)
    # Manuel eşik slider'ı — sadece manuel seçilince görünür ve test'e uygulanır
    manuel_esik = 0.50
    if esik_kaynak.startswith("Manuel"):
        manuel_esik = st.slider(
            "Manuel karar eşiği (test'e uygulanır)",
            min_value=0.05, max_value=0.95, value=0.50, step=0.01,
            help="Olasılık ≥ eşik → patojenik (1). Düşük eşik = daha çok pozitif "
                 "(recall↑, specificity↓); yüksek eşik = daha az yanlış alarm "
                 "(precision↑, recall↓). Klinikte FN'i (kaçırılan patojenik) "
                 "azaltmak için genelde düşük eşik tercih edilir.")

    # Her algoritma için parametre kaynağı
    st.markdown("**Hiperparametre Kaynağı:**")
    algo_params = {}
    for algo in secili_algolar:
        if algo in algo_params_json:
            algo_params[algo] = algo_params_json[algo]
            st.caption(f"  • `{algo}`: JSON'dan yüklendi "
                       f"({len(algo_params_json[algo])} parametre)")
        else:
            algo_params[algo] = {}  # default
            st.caption(f"  • `{algo}`: varsayılan parametreler "
                       f"(JSON'da bulunamadı)")

    # SPW hesabı (default için)
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    # ── EĞİT & TEST ──
    st.markdown("---")
    if st.button("🚀 Eğit ve Test Et", type="primary", use_container_width=True):
        if not secili_algolar:
            st.error("En az bir algoritma seçin.")
            st.stop()

        with st.spinner("Modeller eğitiliyor ve test ediliyor..."):
            # Her algoritma için OOF (train) + test olasılıkları
            from sklearn.model_selection import StratifiedKFold

            test_proba_list = []
            oof_proba_list = []
            basarili_algolar = []

            ilerleme = st.progress(0)
            for i, algo in enumerate(secili_algolar):
                try:
                    # Default parametre dolgu (SPW)
                    p = dict(algo_params.get(algo, {}))
                    if algo in ("xgboost", "lightgbm", "catboost"):
                        p.setdefault("scale_pos_weight", spw)
                        p.setdefault("random_state", 43)
                    elif algo in ("random_forest", "extra_trees"):
                        p.setdefault("class_weight", "balanced")
                        p.setdefault("random_state", 43)
                        p.setdefault("n_jobs", -1)
                    elif algo in ("logistic", "svm_rbf"):
                        p.setdefault("class_weight", "balanced")
                        p.setdefault("random_state", 43)
                        if algo == "logistic":
                            p.setdefault("max_iter", 3000)

                    # Test olasılığı (train full ile eğit)
                    test_proba = model_egit_tahmin(
                        algo, p, X_train, y_train, X_test, kat_kolonlar)
                    test_proba_list.append(test_proba)

                    # OOF olasılığı (eşik seçimi için, 5-fold)
                    oof = np.zeros(len(y_train))
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
                    for tr_idx, val_idx in skf.split(X_train, y_train):
                        Xtr_fold = X_train.iloc[tr_idx]
                        Xval_fold = X_train.iloc[val_idx]
                        fold_proba = model_egit_tahmin(
                            algo, p, Xtr_fold, y_train[tr_idx],
                            Xval_fold, kat_kolonlar)
                        oof[val_idx] = fold_proba
                    oof_proba_list.append(oof)

                    basarili_algolar.append(algo)
                except Exception as e:
                    st.warning(f"`{algo}` başarısız: {e}")
                ilerleme.progress((i + 1) / len(secili_algolar))

            if not test_proba_list:
                st.error("Hiçbir model eğitilemedi.")
                st.stop()

            # ── Ensemble birleştirme ──
            test_proba_arr = np.array(test_proba_list)  # [n_algo, n_test]
            oof_proba_arr = np.array(oof_proba_list)     # [n_algo, n_train]

            if len(basarili_algolar) == 1:
                ensemble_test = test_proba_arr[0]
                ensemble_oof = oof_proba_arr[0]
            else:
                if ensemble_yontem.startswith("Ağırlıklı"):
                    w = np.array([json_meta["ensemble_agirliklari"].get(a, 1.0)
                                  for a in basarili_algolar], dtype=float)
                    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
                    ensemble_test_raw = (test_proba_arr * w[:, None]).sum(axis=0)
                    ensemble_oof_raw = (oof_proba_arr * w[:, None]).sum(axis=0)
                elif ensemble_yontem.startswith("Rank"):
                    from scipy.stats import rankdata
                    ensemble_test_raw = np.mean(
                        [rankdata(p) / len(p) for p in test_proba_arr], axis=0)
                    ensemble_oof_raw = np.mean(
                        [rankdata(p) / len(p) for p in oof_proba_arr], axis=0)
                else:  # Düz ortalama
                    ensemble_test_raw = test_proba_arr.mean(axis=0)
                    ensemble_oof_raw = oof_proba_arr.mean(axis=0)
                # Isotonic kalibrasyon — OOF'ta fit, test'e uygula (sızıntı yok)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(ensemble_oof_raw, y_train)
                ensemble_oof = iso.predict(ensemble_oof_raw)
                ensemble_test = iso.predict(ensemble_test_raw)

            # ── Karar eşiği (kaynak: manuel / v7.9 JSON / OOF'tan yeniden seç) ──
            if esik_kaynak.startswith("Manuel"):
                optimal_esik = float(manuel_esik)
            elif (esik_kaynak.startswith("v7.9")
                    and json_meta.get("optimal_esik") is not None):
                optimal_esik = float(json_meta["optimal_esik"])
            else:
                optimal_esik = en_iyi_f2_esik(y_train, ensemble_oof)

            # ── TAHMİN MODU (Label'sız test): tahmin üret + indir + dur ──
            if y_test is None:
                _pred = (ensemble_test >= optimal_esik).astype(int)
                _cikti = df_test.copy()
                _cikti.insert(0, "tahmin_etiket",
                              [le.classes_[1] if t == 1 else le.classes_[0] for t in _pred])
                _cikti.insert(1, "tahmin", _pred)
                _cikti.insert(2, "patojenik_olasilik", np.round(ensemble_test, 4))
                st.success(
                    f"✅ Tahmin üretildi! Model(ler): {', '.join(basarili_algolar)} | "
                    f"eşik (train/OOF'tan, sızıntısız) = {optimal_esik:.3f} | "
                    f"{int(_pred.sum())}/{len(_pred)} örnek **PATOJENİK (1)** tahmin edildi.")
                st.dataframe(_cikti.head(25), use_container_width=True)
                st.download_button(
                    "📥 Tahminleri CSV indir", _cikti.to_csv(index=False).encode("utf-8"),
                    file_name="biosense_tahminler.csv", mime="text/csv",
                    type="primary", use_container_width=True)
                st.info("ℹ️ Skor metrikleri (F2/MCC) gösterilemiyor çünkü test setinde "
                        "gerçek etiket (Label) yok. Jüri etiketli test verirse o dosyayı "
                        "yükle → tam skor tablosu gelir.")
                st.stop()

            # ── Per-model bireysel test metrikleri (her model kendi OOF eşiğiyle) ──
            per_model_satir = []
            for i, a in enumerate(basarili_algolar):
                e_i = en_iyi_f2_esik(y_train, oof_proba_list[i])
                mm = tum_metrikler(y_test, test_proba_list[i], e_i)
                per_model_satir.append({"Model": a, "Eşik": round(e_i, 3),
                                        **{k: round(v, 4) for k, v in mm.items()}})

            # ── Test tahminleri ──
            test_pred = (ensemble_test >= optimal_esik).astype(int)

        # ═══════════════════════════════════════════════════════════
        #  SONUÇLAR
        # ═══════════════════════════════════════════════════════════
        st.success(f"✅ Tamamlandı! Model(ler): {', '.join(basarili_algolar)} | "
                   f"Optimal eşik (OOF'tan): {optimal_esik:.3f}")

        # Metrikler
        test_f2 = f2_binary(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, pos_label=1, average="binary", zero_division=0)
        test_mcc = matthews_corrcoef(y_test, test_pred)
        try:
            test_auc = roc_auc_score(y_test, ensemble_test)
        except ValueError:
            test_auc = 0.0
        try:
            test_prauc = average_precision_score(y_test, ensemble_test)
        except ValueError:
            test_prauc = 0.0
        test_acc = accuracy_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred, pos_label=1, zero_division=0)
        test_prec = precision_score(y_test, test_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Bootstrap CI (stratified)
        _, f2_lo, f2_hi = stratified_bootstrap_ci(y_test, ensemble_test, optimal_esik, "f2")
        _, f1_lo, f1_hi = stratified_bootstrap_ci(y_test, ensemble_test, optimal_esik, "f1")
        _, mcc_lo, mcc_hi = stratified_bootstrap_ci(y_test, ensemble_test, optimal_esik, "mcc")

        # ── Ana metrik kartları ──
        st.markdown("## 📈 TEKNOFEST Skorları")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metrik-kart">
                <div class="metrik-deger">{test_f2:.4f}</div>
                <div class="metrik-isim">F2 (Ana Metrik)</div>
                <div class="ci-text">95% CI: {f2_lo:.3f}–{f2_hi:.3f}</div>
                </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metrik-kart">
                <div class="metrik-deger">{test_f1:.4f}</div>
                <div class="metrik-isim">F1</div>
                <div class="ci-text">95% CI: {f1_lo:.3f}–{f1_hi:.3f}</div>
                </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metrik-kart">
                <div class="metrik-deger">{test_mcc:.4f}</div>
                <div class="metrik-isim">MCC</div>
                <div class="ci-text">95% CI: {mcc_lo:.3f}–{mcc_hi:.3f}</div>
                </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metrik-kart">
                <div class="metrik-deger">{test_auc:.4f}</div>
                <div class="metrik-isim">ROC-AUC</div>
                <div class="ci-text">PR-AUC: {test_prauc:.3f}</div>
                </div>""", unsafe_allow_html=True)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Accuracy", f"{test_acc:.4f}")
        c6.metric("Recall (Patojenik)", f"{test_recall:.4f}")
        c7.metric("Precision (Patojenik)", f"{test_prec:.4f}")
        c8.metric("Specificity (Benign)", f"{test_spec:.4f}")

        # ── Naive baseline karşılaştırması ──
        akillilik = (test_f2 - naive_f2) / max(1.0 - naive_f2, 1e-6)
        st.markdown("### 🎯 Naive Baseline Karşılaştırması")
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Naive F2 (her şey pozitif)", f"{naive_f2:.4f}")
        cb2.metric("Model F2", f"{test_f2:.4f}")
        cb3.metric("Akıllılık Skoru", f"{akillilik:+.4f}",
                   delta="İyi" if akillilik > 0.10 else "Düşük")
        if akillilik < 0:
            st.error("🚨 Model naive baseline'dan KÖTÜ — F2 yanıltıcı, MCC/AUC'ye bakın.")
        elif akillilik < 0.10:
            st.warning("⚠️ Marjinal model — naive'den çok az iyi.")
        else:
            st.success("✅ Model naive baseline'ı anlamlı şekilde geçti.")

        # ── Bireysel modeller vs Ensemble karşılaştırması ──
        if len(basarili_algolar) > 1:
            st.markdown("### 🔬 Bireysel Modeller vs Ensemble")
            st.caption("Her solo model kendi OOF-optimal eşiğiyle; ENSEMBLE seçili "
                       "birleştirme + isotonic ile. Hangi modelin/ensemble'ın daha "
                       "iyi olduğunu MCC ve PR-AUC üzerinden karşılaştırın.")
            ens_metrik = tum_metrikler(y_test, ensemble_test, optimal_esik)
            tum_satir = list(per_model_satir) + [
                {"Model": "🔗 ENSEMBLE", "Eşik": round(optimal_esik, 3),
                 **{k: round(v, 4) for k, v in ens_metrik.items()}}]
            df_karsi = pd.DataFrame(tum_satir)
            kolon_sira = ["Model", "Eşik", "F2", "F1", "MCC", "ROC-AUC", "PR-AUC",
                          "Recall", "Precision", "Specificity",
                          "TP", "FP", "FN", "TN"]
            df_karsi = df_karsi[[c for c in kolon_sira if c in df_karsi.columns]]
            st.dataframe(df_karsi, use_container_width=True, hide_index=True)

            _solo = df_karsi[df_karsi["Model"] != "🔗 ENSEMBLE"]
            if len(_solo):
                _en_iyi_solo = _solo.loc[_solo["MCC"].idxmax(), "Model"]
                _solo_mcc = float(_solo["MCC"].max())
                _ens_mcc = float(ens_metrik["MCC"])
                if _ens_mcc >= _solo_mcc:
                    st.success(f"✅ Ensemble MCC ({_ens_mcc:.3f}) ≥ en iyi solo "
                               f"**{_en_iyi_solo}** ({_solo_mcc:.3f}) — ensemble tercih edilebilir.")
                else:
                    st.info(f"ℹ️ En iyi solo **{_en_iyi_solo}** (MCC {_solo_mcc:.3f}) "
                            f"ensemble'dan (MCC {_ens_mcc:.3f}) iyi — tek model sunumu "
                            f"daha dürüst olabilir.")
            st.download_button("📥 Model karşılaştırma tablosu (CSV)",
                               data=df_karsi.to_csv(index=False),
                               file_name="model_karsilastirma.csv", mime="text/csv")

        # ── Klinik yorum ──
        st.markdown("### 🩺 Klinik Yorumlama")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""
            - **Doğru yakalanan patojenik (TP)**: {tp}
            - **KAÇIRILAN patojenik (FN)**: {fn}  ⚠️ klinik risk
            - **Yanlış alarm (FP)**: {fp}  (gereksiz takip)
            - **Doğru benign (TN)**: {tn}
            """)
        with cc2:
            if fn == 0:
                st.success(f"✅ Hiç patojenik kaçırılmadı (FN=0)")
            else:
                st.warning(f"⚠️ {fn} patojenik kaçırıldı — klinik açıdan kritik")
            if test_spec < 0.50:
                st.error(f"🚨 Specificity {test_spec:.2f} < 0.50 — "
                         f"çok fazla yanlış alarm")

        # ── Görseller ──
        st.markdown("### 📊 Görseller")
        g1, g2 = st.columns(2)

        with g1:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Patojenik"])
            ax.set_yticklabels(["Benign", "Patojenik"])
            ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
            ax.set_title("Confusion Matrix")
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                            color="white" if cm[ii, jj] > cm.max()/2 else "black",
                            fontsize=16, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with g2:
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(y_test, ensemble_test)
            ax.plot(fpr, tpr, label=f"ROC (AUC={test_auc:.3f})", lw=2)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            prec_c, rec_c, _ = precision_recall_curve(y_test, ensemble_test)
            ax.plot(rec_c, prec_c, label=f"PR (AUC={test_prauc:.3f})", lw=2)
            ax.set_xlabel("FPR / Recall"); ax.set_ylabel("TPR / Precision")
            ax.set_title("ROC & PR Eğrileri"); ax.legend()
            st.pyplot(fig)
            plt.close()

        # ── Karar eşiği tablosu ──
        st.markdown("### 📋 Farklı Karar Eşiklerinde Performans")
        esik_satirlar = []
        for e in np.arange(0.10, 0.75, 0.05):
            e = round(e, 2)
            pr = (ensemble_test >= e).astype(int)
            cm_e = confusion_matrix(y_test, pr)
            tn_e, fp_e, fn_e, tp_e = cm_e.ravel() if cm_e.shape == (2, 2) else (0, 0, 0, 0)
            esik_satirlar.append({
                "Eşik": e,
                "F2": round(f2_binary(y_test, pr), 4),
                "F1": round(f1_score(y_test, pr, pos_label=1, zero_division=0), 4),
                "MCC": round(matthews_corrcoef(y_test, pr), 4),
                "Recall": round(recall_score(y_test, pr, pos_label=1, zero_division=0), 4),
                "Precision": round(precision_score(y_test, pr, pos_label=1, zero_division=0), 4),
                "TP": tp_e, "FP": fp_e, "FN": fn_e, "TN": tn_e,
                "★": "★" if abs(e - optimal_esik) < 0.025 else "",
            })
        df_esik = pd.DataFrame(esik_satirlar)
        st.dataframe(df_esik, use_container_width=True, hide_index=True)

        # ── Sonuçları indir ──
        st.markdown("### 💾 Sonuçları İndir")
        sonuc_ozeti = {
            "modeller": basarili_algolar,
            "optimal_esik": float(optimal_esik),
            "test_f2": float(test_f2), "test_f2_ci": [float(f2_lo), float(f2_hi)],
            "test_f1": float(test_f1), "test_f1_ci": [float(f1_lo), float(f1_hi)],
            "test_mcc": float(test_mcc), "test_mcc_ci": [float(mcc_lo), float(mcc_hi)],
            "test_auc": float(test_auc), "test_pr_auc": float(test_prauc),
            "test_accuracy": float(test_acc),
            "patojenik_recall": float(test_recall),
            "patojenik_precision": float(test_prec),
            "specificity": float(test_spec),
            "naive_baseline_f2": float(naive_f2),
            "akillilik_skoru": float(akillilik),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp),
                                 "FN": int(fn), "TN": int(tn)},
        }
        st.download_button(
            "📥 JSON olarak indir",
            data=json.dumps(sonuc_ozeti, indent=2, ensure_ascii=False),
            file_name="test_sonuclari.json",
            mime="application/json",
        )
        st.download_button(
            "📥 Eşik tablosu (CSV)",
            data=df_esik.to_csv(index=False),
            file_name="karar_esigi_tablosu.csv",
            mime="text/csv",
        )

else:
    st.info("👈 Başlamak için kenar çubuğundan **Train CSV** ve **Test CSV** "
            "dosyalarını yükleyin.")
    st.markdown("""
    ### 📖 Nasıl Kullanılır?

    1. **Arkadaşların** orijinal veri setinin **%20'sini** ayırıp hiç dokunmadan
       ayrı bir `test.csv` dosyası yapar. Kalan %80 `train.csv` olur (v7.9 bununla çalışır).
    2. **Train CSV** ve **Test CSV** dosyalarını yükle.
    3. (Önerilir) v7.9'un ürettiği **`en_iyi_parametreler_pah_v4.json`** dosyasını yükle.
       Otomatik kullanılır: tuned hiperparametreler · **SHAP-seçili özellik listesi** ·
       **v7.9'un önerdiği ensemble üyeleri + ağırlıkları** · optimal eşik.
    4. **Tek Model** veya **Ensemble** modunu seç. Ensemble'da birden fazla model
       seçip birleştirme yöntemini (düz / ağırlıklı / rank) belirle.
    5. **Eğit ve Test Et** → her modelin **bireysel** metrikleri + **ensemble**
       karşılaştırması (MCC/PR-AUC ile en iyiyi seç).

    ### 🔬 Metodoloji (ana pipeline ile birebir)
    - JSON'dan **aynı SHAP-seçili özellik seti** uygulanır (parametre uyumu)
    - Kategorik kolonlar **TRAIN→TEST aynı kategori seti** ile encode edilir
    - **Stratified bootstrap** %95 CI (imbalanced data için)
    - **Binary F2** (pos_label=1, patojenik) + F1 + MCC + **PR-AUC** (rubrik zorunlu)
    - Optimal eşik **OOF/train'de** seçilir (veya v7.9 JSON eşiği), test'e uygulanır
    - **ID çakışma kontrolü** (veri sızıntısı tespiti)
    - **Naive baseline** karşılaştırması + akıllılık skoru

    ### ⚠️ Önemli
    Bu araç modeli **sıfırdan** train setiyle eğitir ve test setiyle **bir kez**
    değerlendirir. Test seti hiçbir eğitim/seçim/eşik adımında kullanılmaz (sızıntı yok).
    """)
