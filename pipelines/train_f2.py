"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BioSense AI v3.6 — 4-FAZ HİYERARŞİK ALGORİTMA SEÇİMİ                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FAZ 1: 10 algoritma geniş tarama (5×20=100 fold)                            ║
║         Hibrit eleme: F2_ort<0.70 + (F2_ort-0.5×std)<0.65 + Welch p<0.05     ║
║  FAZ 2: Hayatta kalanlar arasında korelasyon + çeşitlilik seçimi → top 3     ║
║         Greedy: en yüksek F2 → en düşük ort. korelasyona ekle                ║
║  FAZ 3: Seçilen 3 algoritmanın her birini Optuna ile derin tunela            ║
║         Her algo: ~1500 trial, 5×10=50 fold, TPE multivariate                ║
║  FAZ 4: Ensemble (top-5×3=15 üye) + isotonic kalibrasyon + test              ║
║                                                                              ║
║  v3.0 → v3.1 DÜZELTMELERİ:                                                   ║
║   ✓ FIX A: korelasyon_df NaN → 1.0 fillna (sabit-tahmin algos için)         ║
║   ✓ FIX B: FAZ 3 top-K kalite filtresi (CV<0.30 trial'lar elensin)          ║
║   ✓ FIX C: TRIAL_RENAME yukarı taşındı, FAZ 2 inline duplikasyon temizlendi  ║
║   ✓ İYİ:  "FİNAL 3 ALGORİTMA" gerçek sayıyı gösteriyor                       ║
║                                                                              ║
║  v3.1 → v3.2 DÜZELTMELERİ (ES israfı):                                       ║
║   ✓ FIX D: FAZ 1 objective — non-boosting algos %10 veri kaybı önlendi      ║
║   ✓ FIX E: FAZ 3 objective — aynı düzeltme derin tuning için                ║
║   ✓ FIX F: FAZ 4-B — non-boosting final modeller tam X_train kullanır       ║
║      Net etki: SVM, LR, MLP, RF, ET, kNN, NB algolarına +22-25 örnek geri    ║
║      kazandırıldı (388 örnekte fark eden bir kazanım).                       ║
║                                                                              ║
║  v3.2 → v3.3 EKLEMELER (Jüri etkisi):                                        ║
║   ✓ FIX G: LabelEncoder sınıf haritalaması açıkça yazdırılıyor,             ║
║      patojenik class 0'da kalırsa otomatik uyarı                            ║
║   ✓ EK 1: Bootstrap 95% CI (Test F2, F1, AUC için 2000 iterasyon)           ║
║      "Test F2 = 0.87 [95% CI: 0.82-0.91]" formatında jüri sunumu            ║
║   ✓ EK 2: Per-class klinik metrikler (Patojenik recall/precision/F2/F1)     ║
║      KAÇIRILAN patojenik sayısı (false negative = klinik risk)              ║
║      Yanlış alarm sayısı (false positive = gereksiz takip)                  ║
║                                                                              ║
║  v3.3 → v3.4 DÜZELTMELERİ (Run çıktısı ortaya çıkardı):                     ║
║   ✓ FIX H: CatBoost duplicate verbose=False kaldırıldı (TypeError önlendi)  ║
║      + bootstrap_type="Bernoulli" → subsample desteği sağlandı              ║
║   ✓ FIX I: SimpleImputer non-tree pipeline'lara eklendi                     ║
║      LR/SVM/kNN/NB/MLP artık NaN içeren veriyle çalışabiliyor               ║
║   ✓ FIX J: SHAP budama top-K mantığına geçti (yüzdelik karşılaştırma değil) ║
║      Çok sayıda sıfır SHAP değeri olunca da garanti %20 budama              ║
║                                                                              ║
║  v3.4 → v3.5 METRİK DÜZELTMESİ (Klinik Uyum):                               ║
║   ✓ weighted F2/F1 → binary F2/F1 (pos_label=1, sadece patojenik sınıfı)    ║
║      Klinik amaç (patojenik yakalama) ile metrik birebir uyumlu             ║
║      Optuna objective, eşik optimizasyonu, test raporlama hepsi binary'de   ║
║   ✓ FAZ 1 tablosuna F1_ort sütunu eklendi (sadece raporlama amaçlı)         ║
║      Eleme ve seçim hâlâ F2 ile, F1 jüri kıyaslaması için durur             ║
║                                                                              ║
║  v3.5 → v3.6 DÜZELTMELERİ (gerçek-veri run'larından çıkanlar):              ║
║   ✓ FIX M: Kategorik kolonların NaN'ları "EKSİK" string'i ile dolduruluyor  ║
║      CatBoost float NaN'ı cat_features'da kabul etmiyor → v3.4'te crash    ║
║      oluyordu. Bu fix sonrası CatBoost FAZ 1'de en iyi F2=0.9615 verdi.    ║
║   ✓ FIX N: FAZ 3 trial bütçesi algoritmaya göre dict (1500 sabit yerine)   ║
║      sklearn algos (RF/ET/LR/SVM/kNN/NB/MLP) GIL + joblib overhead'ten      ║
║      dolayı CPU %10'a düşüyor, 6 saatlik tahminler veriyor. Algo-bazlı     ║
║      bütçe → FAZ 3 toplam süre ~18 saat'ten ~4 saat'e indi.                ║
║                                                                              ║
║  v2.8'in tüm FIX #1-#20 düzeltmeleri korundu.                                ║
║  10 algoritma: 3 boosting + 2 bagging + 1 linear + 1 kernel + 1 instance-    ║
║  based + 1 probabilistic + 1 neural → 7 paradigma kapsanır                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, time, warnings, gc

# [FIX K] Windows CMD/PowerShell default encoding cp1254 (Türkçe) bazı
# Unicode karakterleri (→, ★, ✓, █, üç noktalı tireler) yazdıramaz.
# Output redirect (> log.txt) durumunda UnicodeEncodeError fırlar.
# Bu satırlar stdout/stderr'i UTF-8'e zorlar — kod platformdan bağımsız çalışır.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    fbeta_score, f1_score, roc_auc_score,
    accuracy_score, matthews_corrcoef, confusion_matrix,
    precision_score, recall_score,
)
from scipy.stats import rankdata, ttest_ind

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False
    print("[UYARI] LightGBM yok → pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CB_OK = True
except ImportError:
    CB_OK = False
    print("[UYARI] CatBoost yok → pip install catboost")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("[UYARI] SHAP yok → pip install shap")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  KONTROL PANELİ
# ══════════════════════════════════════════════════════════════════
VERI_DOSYASI   = "YARISMA_TRAIN_KANSER.csv"
CIKTI_JSON     = "en_iyi_parametreler_v3.json"
CIKTI_FAZ1_CSV = "faz1_algoritma_kiyaslama.csv"
CIKTI_KORELASYON_CSV = "faz2_korelasyon_matrisi.csv"

PARALEL_ISCI   = 24

# ── FAZ 1: Geniş tarama ───────────────────────────────────────────
FAZ1_FOLDS     = 5
FAZ1_TEKRAR    = 20         # 5×20 = 100 fold (kararlılık için)
# Algoritma başına Optuna trial sayısı (karmaşıklığa göre değişiyor)
FAZ1_TRIAL_PER_ALGO = {
    "xgboost":       80,
    "lightgbm":      80,
    "catboost":      60,
    "random_forest": 50,
    "extra_trees":   50,
    "logistic":      40,
    "svm_rbf":       60,
    "knn":           20,
    "naive_bayes":    5,    # çok az hiperparametre
    "mlp":           80,
}
FAZ1_DB_PREFIX = "sqlite:///biosense_faz1_v3"

# FAZ 1 eleme kriterleri (3'ü AYNI ANDA sağlanırsa elenir)
FAZ1_F2_ORT_ESIK    = 0.70   # F2 ortalama bu altındaysa
FAZ1_F2_LCB_ESIK    = 0.65   # (F2_ort - 0.5×std) bu altındaysa
FAZ1_TTEST_P_ESIK   = 0.05   # ve top'a göre t-test p<0.05 ise

# ── FAZ 2: Çeşitlilik Seçimi ──────────────────────────────────────
FAZ2_FOLDS_OOF     = 5
FAZ2_TEKRAR_OOF    = 10
FAZ2_SECILEN_SAYI  = 3       # final ensemble paradigma sayısı
FAZ2_OOF_ES_SPLIT  = 0.12

# ── FAZ 3: Derin Tuning ───────────────────────────────────────────
# [FIX N] Algoritma bazlı trial bütçesi. Önceki tek-sayı yaklaşımı
# (1500 her algoritma için) sklearn-tabanlı algos için verimsizdi:
# - XGB/LGB/CB: C++ tabanlı, paralelizm verimli → 1000-1500 trial mantıklı
# - sklearn (RF, ET, LR, SVM, kNN, NB, MLP): Python GIL + joblib spawn
#   overhead → CPU %10'a düşüyor. 500 trial sklearn için zaten doyurur,
#   hyperparameter yüzeyleri daha düz.
# Net etki: FAZ 3 toplam süresi ~18 saat → ~4 saat.
FAZ3_TRIAL_PER_ALGO = {
    # C++ tabanlı boosting — paralelizm verimli
    "xgboost":       1500,
    "lightgbm":      1500,
    "catboost":      1000,
    # sklearn tree — paralelizm verimsiz, az trial yeterli
    "random_forest":  500,
    "extra_trees":    500,
    # sklearn non-tree
    "logistic":       600,
    "svm_rbf":        400,   # probability=True yavaş
    "knn":            200,   # 3 hyperparam, hızla doyurur
    "naive_bayes":    100,   # 1 hyperparam
    "mlp":            800,
}
FAZ3_FOLDS          = 5
FAZ3_TEKRAR         = 10     # 50 fold
FAZ3_DB_PREFIX      = "sqlite:///biosense_faz3_v3"
FAZ3_TOP_K          = 5      # her algo'dan ensemble'a ekle

# ── FAZ 4: Ensemble + Test ────────────────────────────────────────
OOF_FOLDS      = 5
OOF_TEKRAR     = 10
OOF_ES_SPLIT   = 0.12

EARLY_STOP_ROUNDS      = 50
ES_SPLIT_OBJECTIVE     = 0.10
MIN_ES_ORNEKLER        = 8

SHAP_PERCENTILE    = 20
W_F2               = 0.80
W_AUC              = 0.20
SOFTMAX_SICAKLIK   = 10.0
MIN_MESAFE_ICIN_ENSEMBLE = 0.05

TEST_ORANI     = 0.20
RASTGELE_TOHUM = 43

# Tüm algoritma havuzu (faz 1'de taranır)
TUM_ALGORITMALAR = ["xgboost"]
if LGBM_OK: TUM_ALGORITMALAR.append("lightgbm")
if CB_OK:   TUM_ALGORITMALAR.append("catboost")
TUM_ALGORITMALAR.extend([
    "random_forest", "extra_trees",
    "logistic", "svm_rbf", "knn", "naive_bayes", "mlp",
])

# Algoritma → paradigma haritası (sunum tablosu için)
ALGO_PARADIGMA = {
    "xgboost":       "Boosting Ağaç (level-wise)",
    "lightgbm":      "Boosting Ağaç (leaf-wise)",
    "catboost":      "Boosting Ağaç (ordered)",
    "random_forest": "Bagging Ağaç",
    "extra_trees":   "Bagging Ağaç (randomize)",
    "logistic":      "Doğrusal (olasılıksal)",
    "svm_rbf":       "Kernel (geometrik)",
    "knn":           "Örnek-temelli (geometrik)",
    "naive_bayes":   "Üretimsel (olasılıksal)",
    "mlp":           "Sinir Ağı",
}

# Algoritma → preprocessing gereksinimi
# Tree-native (kategorik desteği): XGB, LGB, CB
# Tree-numeric (kategorik integer kodu yeterli): RF, ET
# Non-tree (kategorik integer + StandardScaler şart): LR, SVM, kNN, NB, MLP
PREPROCESSING_GEREK = {
    "xgboost":       "tree_native",
    "lightgbm":      "tree_native",
    "catboost":      "tree_native",
    "random_forest": "tree_numeric",
    "extra_trees":   "tree_numeric",
    "logistic":      "non_tree",
    "svm_rbf":       "non_tree",
    "knn":           "non_tree",
    "naive_bayes":   "non_tree",
    "mlp":           "non_tree",
}

print("=" * 72)
print("  BioSense AI v3.6 — 4-FAZ HİYERARŞİK (binary F2 + algo-bazlı bütçe)")
print(f"  Toplam algoritma havuzu: {len(TUM_ALGORITMALAR)}")
print(f"  FAZ 1: {sum(FAZ1_TRIAL_PER_ALGO.get(a, 30) for a in TUM_ALGORITMALAR)} "
      f"trial × {FAZ1_FOLDS*FAZ1_TEKRAR} fold")
print(f"  FAZ 3: {FAZ2_SECILEN_SAYI} algo × (algo-bazlı trial) × "
      f"{FAZ3_FOLDS*FAZ3_TEKRAR} fold")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
#  1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════
print("\n[1/9] Veri yükleniyor...")

if not os.path.exists(VERI_DOSYASI):
    print(f"  HATA: '{VERI_DOSYASI}' bulunamadı!"); sys.exit(1)

df      = pd.read_csv(VERI_DOSYASI)
HEDEF   = "Label"
OZELLIK = [c for c in df.columns if c not in ["Variant_ID", HEDEF]]

tek_tip = [c for c in OZELLIK if df[c].nunique() <= 1]
if tek_tip:
    df.drop(columns=tek_tip, inplace=True)
    OZELLIK = [c for c in df.columns if c not in ["Variant_ID", HEDEF]]
    print(f"  {len(tek_tip)} sıfır-varyanslı kolon silindi.")

X  = df[OZELLIK].copy()
le = LabelEncoder()
y  = le.fit_transform(df[HEDEF])

# [FIX G] Pozitif sınıf hangi etikete karşılık geliyor? Yanlış yorumlama
# riskine karşı açıkça yazdır. F2 metriği class 1'i pozitif kabul eder.
print(f"  Label haritalama: '{le.classes_[0]}' → 0 (negatif), "
      f"'{le.classes_[1]}' → 1 (POZİTİF, F2 hedefi)")
if "patho" in str(le.classes_[1]).lower() or "malign" in str(le.classes_[1]).lower():
    print(f"  ✓ Pozitif sınıf doğru (Pathogenic/Malignant tarafı)")
elif "patho" in str(le.classes_[0]).lower() or "malign" in str(le.classes_[0]).lower():
    print(f"  ⚠ UYARI: Pozitif sınıf (1) muhtemelen YANLIŞ tarafta! "
          f"Pathogenic class 0'a düşmüş. Etiketleri kontrol edin veya "
          f"y = 1 - y ile ters çevirin.")

KAT_KOLONLAR = []
for col in X.columns:
    if (X[col].dtype == "object" or X[col].dtype.name == "category"
            or str(col).startswith("CAT_") or str(col).startswith("AA_")):
        # [FIX M] Kategorik kolonlardaki float NaN'lar CatBoost tarafından
        # okunamıyordu (CatBoost cat_features parametresinde float NaN'ı
        # kabul etmez). Onları "EKSİK" string'i ile dolduralım — CatBoost
        # bunu ayrı bir kategori olarak ele alır, tree-numeric / non-tree
        # algos da numerik kodlama sırasında bu kategoriyi doğru işler.
        # v3.4 run'da CatBoost crash'inin asıl sebebi buydu.
        X[col] = X[col].fillna("EKSİK").astype(str).astype("category")
        KAT_KOLONLAR.append(col)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_ORANI, random_state=RASTGELE_TOHUM, stratify=y
)
y_train = np.array(y_train)
y_test  = np.array(y_test)

SPW     = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
N_TRAIN = len(X_train)
N_TEST  = len(X_test)

print(f"  Train: {N_TRAIN} | Test (kilitli): {N_TEST} | SPW: {SPW:.3f}")
print(f"  Özellik: {len(OZELLIK)} ({len(KAT_KOLONLAR)} kategorik)")
print(f"  Sınıf dağılımı: {dict(zip(*np.unique(y, return_counts=True)))}")

BUDANAN_OZELLIKLER = []

# ══════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR (v2.8'den korunan)
# ══════════════════════════════════════════════════════════════════

def get_algo(t):
    return t.user_attrs.get("algo", t.params.get("algorithm", "xgboost"))

def en_iyi_f2_esik_benzersiz(y_gercek, problar):
    """v2.8 FIX #17: filtre boşsa orijinal prob dağılımına geri dön.
    v3.5: average='binary' (pos_label=1) — klinik amaçla uyumlu."""
    adaylar = np.unique(problar)
    adaylar_dar = adaylar[(adaylar > 0.01) & (adaylar < 0.99)]
    adaylar = adaylar_dar if len(adaylar_dar) > 0 else adaylar
    if len(adaylar) == 0:
        return 0.50, 0.0
    en_iyi_f2  = 0.0
    esik_sonuc = 0.50
    for e in adaylar:
        f2 = fbeta_score(y_gercek, (problar >= e).astype(int),
                         beta=2, pos_label=1, average="binary",
                         zero_division=0)
        if f2 > en_iyi_f2:
            en_iyi_f2  = f2
            esik_sonuc = float(e)
    return round(esik_sonuc, 5), round(en_iyi_f2, 6)

def en_iyi_f2_esik_grid(y_gercek, problar, adim=0.005):
    """v3.5: average='binary' (pos_label=1)."""
    en_iyi_f2  = 0.0
    esik_sonuc = 0.50
    for e in np.arange(0.05, 0.95 + adim, adim):
        f2 = fbeta_score(y_gercek, (problar >= e).astype(int),
                         beta=2, pos_label=1, average="binary",
                         zero_division=0)
        if f2 > en_iyi_f2:
            en_iyi_f2  = f2
            esik_sonuc = round(float(e), 4)
    return esik_sonuc, en_iyi_f2

def kat_idx(X_df):
    return [X_df.columns.get_loc(c) for c in KAT_KOLONLAR if c in X_df.columns]

def softmax_agirliklar(f2_dizi, sicaklik=10.0):
    z = np.array(f2_dizi, dtype=float) * sicaklik
    z -= z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

def bootstrap_metric_ci(y_true, y_proba_or_pred, metric_func, n_boot=2000,
                         alpha=0.05, threshold=None, random_state=43):
    """
    Bootstrap yüzdelik yöntemiyle herhangi bir metriğin güven aralığı.
    TEKNOFEST jüri sunumunda "Test F2 = 0.87 [95% CI: 0.82-0.91]" gibi
    istatistiksel rigor sağlar.

    Args:
        y_true            : gerçek etiketler (np.array)
        y_proba_or_pred   : model olasılıkları veya sınıf tahminleri
        metric_func       : metric(y_true, y_pred) → float
        n_boot            : bootstrap iterasyon (2000 → kararlı CI)
        alpha             : 0.05 → 95% CI
        threshold         : None ise y_proba_or_pred zaten tahmin;
                            sayı ise olasılığı eşikle dijitalize et
        random_state      : tekrar üretilebilirlik

    Returns:
        (mean, lower_95ci, upper_95ci) tuple
    """
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_p = y_proba_or_pred[idx]
        if threshold is not None:
            y_p = (y_p >= threshold).astype(int)
        try:
            s = float(metric_func(y_t, y_p))
        except Exception:
            s = 0.0
        scores.append(s)
    return (float(np.mean(scores)),
            float(np.percentile(scores, 100 * alpha / 2)),
            float(np.percentile(scores, 100 * (1 - alpha / 2))))

def rank_ensemble(proba_listesi):
    ranked = [rankdata(p) / len(p) for p in proba_listesi]
    return np.mean(ranked, axis=0)

# v2.8 FIX #1+#5: Merkezi ES ayrım fonksiyonu
def fold_es_ayir(X_fold_tr, y_fold_tr):
    if len(X_fold_tr) < 30:
        return X_fold_tr, y_fold_tr, None, None
    n_es = max(MIN_ES_ORNEKLER, int(len(X_fold_tr) * ES_SPLIT_OBJECTIVE))
    try:
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_fold_tr, y_fold_tr, test_size=n_es,
            random_state=RASTGELE_TOHUM, stratify=y_fold_tr,
        )
        return X_tr, y_tr, X_es, y_es
    except ValueError:
        pass
    try:
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_fold_tr, y_fold_tr, test_size=n_es,
            random_state=RASTGELE_TOHUM,
        )
        return X_tr, y_tr, X_es, y_es
    except Exception:
        return X_fold_tr, y_fold_tr, None, None

# ══════════════════════════════════════════════════════════════════
#  BİRLEŞİK MODEL ARAYÜZÜ — 10 ALGORİTMA İÇİN
# ══════════════════════════════════════════════════════════════════

def numerik_donustur(X_df):
    """
    Kategorik kolonları integer kodlara çevirir (tree-numeric ve non-tree için).
    X_train ve X_test aynı kategori haritasını paylaşır (pandas .cat.codes).
    """
    X_num = X_df.copy()
    for col in X_num.columns:
        if X_num[col].dtype.name == "category":
            X_num[col] = X_num[col].cat.codes.astype(float)
    return X_num

def model_olustur(algo, params, X_tr_sample=None):
    """
    Bir algoritma için Pipeline veya doğrudan model nesnesi döndürür.
    Non-tree modeller için StandardScaler içerir.
    """
    p = dict(params)

    if algo == "xgboost":
        p.setdefault("enable_categorical", True)
        p.setdefault("tree_method", "hist")
        p.setdefault("device", "cpu")
        p.setdefault("verbosity", 0)
        p.setdefault("random_state", RASTGELE_TOHUM)
        return xgb.XGBClassifier(**p)

    elif algo == "lightgbm":
        p.setdefault("verbosity", -1)
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("subsample_freq", 1)
        return lgb.LGBMClassifier(**p)

    elif algo == "catboost":
        p.setdefault("random_seed", RASTGELE_TOHUM)
        p.setdefault("verbose", False)
        p.setdefault("bootstrap_type", "Bernoulli")
        return CatBoostClassifier(**p)

    elif algo == "random_forest":
        p.setdefault("random_state", RASTGELE_TOHUM)
        return RandomForestClassifier(**p)

    elif algo == "extra_trees":
        p.setdefault("random_state", RASTGELE_TOHUM)
        return ExtraTreesClassifier(**p)

    elif algo == "logistic":
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("max_iter", 2000)
        # [FIX I] SimpleImputer eklendi — sklearn LR NaN'ı kaldıramaz.
        # Genomik verilerde sıkça eksik değer olur; median ile dolduruyoruz.
        # Fold içinde fit ediliyor → veri sızıntısı yok.
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", LogisticRegression(**p))])

    elif algo == "svm_rbf":
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("probability", True)
        p.setdefault("kernel", "rbf")
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", SVC(**p))])

    elif algo == "knn":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(**p))])

    elif algo == "naive_bayes":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", GaussianNB(**p))])

    elif algo == "mlp":
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("max_iter", 500)
        p.setdefault("early_stopping", True)
        p.setdefault("validation_fraction", 0.15)
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", MLPClassifier(**p))])
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo}")

def model_fit_universal(algo, params, X_tr, y_tr, X_val=None, y_val=None,
                        n_jobs_override=None):
    """
    Tüm 10 algoritma için tek arayüz.
    - Tree-native (XGB/LGB/CB): pandas categorical doğrudan
    - Tree-numeric (RF/ET): numerik X kullan
    - Non-tree: numerik X + Pipeline (StandardScaler içeride)
    Early stopping: yalnız XGB/LGB/CB destekler.
    """
    use_es = (X_val is not None) and (y_val is not None) and (len(X_val) >= 5)
    prep = PREPROCESSING_GEREK[algo]

    # Tree-native — kategorik native destekli
    if algo == "xgboost":
        p = dict(params)
        # Defensif defaults — FAZ 2'den eksik params gelirse de güvenli
        p.setdefault("enable_categorical", True)
        p.setdefault("tree_method", "hist")
        p.setdefault("device", "cpu")
        p.setdefault("verbosity", 0)
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("n_jobs", 1)
        if n_jobs_override: p["n_jobs"] = n_jobs_override
        if use_es:
            m = xgb.XGBClassifier(**p, early_stopping_rounds=EARLY_STOP_ROUNDS,
                                  eval_metric="auc")
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr)
        return m

    if algo == "lightgbm":
        p = dict(params)
        p.setdefault("verbosity", -1)
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("n_jobs", 1)
        p.setdefault("subsample_freq", 1)
        if n_jobs_override: p["n_jobs"] = n_jobs_override
        cat_i = kat_idx(X_tr)
        cat_i_safe = cat_i if cat_i else "auto"
        m = lgb.LGBMClassifier(**p)
        if use_es:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  categorical_feature=cat_i_safe,
                  callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                             lgb.log_evaluation(-1)])
        else:
            m.fit(X_tr, y_tr, categorical_feature=cat_i_safe)
        return m

    if algo == "catboost":
        p = dict(params)
        p.setdefault("random_seed", RASTGELE_TOHUM)
        p.setdefault("verbose", False)
        p.setdefault("thread_count", 1)
        # [FIX H] Yeni CatBoost: default bootstrap_type=Bayesian subsample'ı kabul
        # etmez. Bernoulli klasik subsample desteklemeli.
        p.setdefault("bootstrap_type", "Bernoulli")
        if n_jobs_override: p["thread_count"] = n_jobs_override
        cat_i = kat_idx(X_tr)
        cat_i_safe = cat_i if cat_i else None
        if use_es:
            # [FIX H] verbose=False zaten p içinde → constructor'a tekrar
            # eklemek "got multiple values for 'verbose'" TypeError üretiyordu.
            m = CatBoostClassifier(**p, early_stopping_rounds=EARLY_STOP_ROUNDS,
                                   cat_features=cat_i_safe)
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        else:
            m = CatBoostClassifier(**p, cat_features=cat_i_safe)
            m.fit(X_tr, y_tr)
        return m

    # Tree-numeric ve Non-tree: numerik kategoriler şart
    X_tr_in = numerik_donustur(X_tr) if prep != "tree_native" else X_tr

    p = dict(params)
    # RF / ET için n_jobs override
    if algo in ("random_forest", "extra_trees") and n_jobs_override:
        p["n_jobs"] = n_jobs_override
    # k-NN n_jobs
    if algo == "knn" and n_jobs_override:
        p["n_jobs"] = n_jobs_override

    m = model_olustur(algo, p)
    m.fit(X_tr_in, y_tr)
    return m

def predict_proba_universal(model, algo, X_in):
    """
    Modelden olasılık tahmini al. Non-tree için numerik dönüşüm yapar.
    """
    prep = PREPROCESSING_GEREK[algo]
    if prep == "tree_native":
        return model.predict_proba(X_in)[:, 1]
    else:
        X_num = numerik_donustur(X_in)
        return model.predict_proba(X_num)[:, 1]

# ══════════════════════════════════════════════════════════════════
#  OOF FONKSİYONU (10 ALGORİTMA İÇİN BİRLEŞİK)
# ══════════════════════════════════════════════════════════════════

def oof_tahmin_universal(algo, params, X_tr, y_tr, n_folds, n_tekrar,
                          n_jobs_override=None, es_split=0.0):
    """
    10 algoritmanın hepsiyle uyumlu OOF tahmini.
    v2.8 FIX #10+#14: try/except + skip örnekler prior alır.
    """
    rskf   = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_tekrar,
                                     random_state=RASTGELE_TOHUM)
    toplam = np.zeros(len(X_tr))
    sayac  = np.zeros(len(X_tr))

    for tr_idx, val_idx in rskf.split(X_tr, y_tr):
        X_ft = X_tr.iloc[tr_idx]
        y_ft = y_tr[tr_idx]
        X_fv = X_tr.iloc[val_idx]

        X_es = y_es = None
        # Early stopping yalnız boosting algos için anlamlı
        if (algo in ("xgboost", "lightgbm", "catboost")
                and es_split > 0 and len(tr_idx) >= 20):
            n_es = max(MIN_ES_ORNEKLER, int(len(tr_idx) * es_split))
            try:
                X_ft, X_es, y_ft, y_es = train_test_split(
                    X_ft, y_ft, test_size=n_es,
                    random_state=RASTGELE_TOHUM, stratify=y_ft,
                )
            except ValueError:
                try:
                    X_ft, X_es, y_ft, y_es = train_test_split(
                        X_ft, y_ft, test_size=n_es, random_state=RASTGELE_TOHUM,
                    )
                except Exception:
                    X_es = y_es = None

        try:
            m = model_fit_universal(algo, params, X_ft, y_ft, X_es, y_es,
                                    n_jobs_override=n_jobs_override)
            fold_proba = predict_proba_universal(m, algo, X_fv)
            toplam[val_idx] += fold_proba
            sayac[val_idx]  += 1
        except Exception:
            pass

    prior = float(y_tr.mean())
    sonuc = toplam / np.maximum(sayac, 1)
    sonuc[sayac == 0] = prior
    return sonuc

# ══════════════════════════════════════════════════════════════════
#  10 ALGORİTMA İÇİN OPTUNA SUGGEST FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════

def suggest_params(trial, algo, geniş=True):
    """
    Bir algoritmanın hiperparametre aralıklarını öner.
    `geniş=True` → FAZ 1/FAZ 3 için tam aralık
    """
    if algo == "xgboost":
        return {
            "n_estimators":      trial.suggest_int("xgb_n_est", 100, 1000),
            "max_depth":         trial.suggest_int("xgb_depth", 2, 8),
            "min_child_weight":  trial.suggest_int("xgb_mcw", 1, 30),
            "max_delta_step":    trial.suggest_int("xgb_mds", 0, 10),
            "learning_rate":     trial.suggest_float("xgb_lr", 0.003, 0.30, log=True),
            "subsample":         trial.suggest_float("xgb_sub", 0.35, 1.00),
            "colsample_bytree":  trial.suggest_float("xgb_cbt", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("xgb_cbl", 0.40, 1.00),
            "colsample_bynode":  trial.suggest_float("xgb_cbn", 0.40, 1.00),
            "gamma":             trial.suggest_float("xgb_gamma", 0.0, 8.0),
            "reg_alpha":         trial.suggest_float("xgb_alpha", 0.0, 15.0),
            "reg_lambda":        trial.suggest_float("xgb_lambda", 0.1, 30.0),
            "scale_pos_weight":  trial.suggest_float("xgb_spw", SPW*0.2, SPW*6.0),
            "enable_categorical": True, "tree_method": "hist", "device": "cpu",
            "verbosity": 0, "random_state": RASTGELE_TOHUM, "n_jobs": 1,
        }

    if algo == "lightgbm":
        return {
            "n_estimators":      trial.suggest_int("lgb_n_est", 100, 1000),
            "num_leaves":        trial.suggest_int("lgb_leaves", 20, 300),
            "max_depth":         trial.suggest_int("lgb_depth", -1, 12),
            "min_child_samples": trial.suggest_int("lgb_mcs", 5, 100),
            "learning_rate":     trial.suggest_float("lgb_lr", 0.003, 0.30, log=True),
            "subsample":         trial.suggest_float("lgb_sub", 0.35, 1.00),
            "colsample_bytree":  trial.suggest_float("lgb_cbt", 0.35, 1.00),
            "reg_alpha":         trial.suggest_float("lgb_alpha", 0.0, 15.0),
            "reg_lambda":        trial.suggest_float("lgb_lambda", 0.1, 30.0),
            "min_split_gain":    trial.suggest_float("lgb_msg", 0.0, 5.0),
            "scale_pos_weight":  trial.suggest_float("lgb_spw", SPW*0.2, SPW*6.0),
            "verbosity": -1, "random_state": RASTGELE_TOHUM, "n_jobs": 1,
            "subsample_freq": 1,
        }

    if algo == "catboost":
        return {
            "iterations":        trial.suggest_int("cb_iter", 100, 800),
            "depth":             trial.suggest_int("cb_depth", 2, 8),
            "learning_rate":     trial.suggest_float("cb_lr", 0.003, 0.30, log=True),
            "l2_leaf_reg":       trial.suggest_float("cb_l2", 0.1, 30.0),
            "min_data_in_leaf":  trial.suggest_int("cb_mdl", 1, 50),
            "subsample":         trial.suggest_float("cb_sub", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("cb_cbl", 0.40, 1.00),
            "scale_pos_weight":  trial.suggest_float("cb_spw", SPW*0.2, SPW*6.0),
            "random_seed": RASTGELE_TOHUM, "thread_count": 1, "verbose": False,
        }

    if algo == "random_forest":
        return {
            "n_estimators":     trial.suggest_int("rf_n_est", 100, 1000),
            "max_depth":        trial.suggest_int("rf_depth", 3, 25),
            "min_samples_split":trial.suggest_int("rf_mss", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_msl", 1, 20),
            "max_features":     trial.suggest_float("rf_mf", 0.2, 1.0),
            "class_weight":     trial.suggest_categorical("rf_cw",
                                                          ["balanced", "balanced_subsample", None]),
            "n_jobs": 1, "random_state": RASTGELE_TOHUM,
        }

    if algo == "extra_trees":
        return {
            "n_estimators":     trial.suggest_int("et_n_est", 100, 1000),
            "max_depth":        trial.suggest_int("et_depth", 3, 25),
            "min_samples_split":trial.suggest_int("et_mss", 2, 20),
            "min_samples_leaf": trial.suggest_int("et_msl", 1, 20),
            "max_features":     trial.suggest_float("et_mf", 0.2, 1.0),
            "class_weight":     trial.suggest_categorical("et_cw",
                                                          ["balanced", "balanced_subsample", None]),
            "n_jobs": 1, "random_state": RASTGELE_TOHUM,
        }

    if algo == "logistic":
        # ElasticNet → liblinear desteklemez, saga gerekir
        penalty = trial.suggest_categorical("lr_pen", ["l1", "l2", "elasticnet"])
        params = {
            "C":         trial.suggest_float("lr_C", 0.001, 100.0, log=True),
            "penalty":   penalty,
            "solver":    "saga",
            "class_weight": trial.suggest_categorical("lr_cw", ["balanced", None]),
            "max_iter":  3000,
            "random_state": RASTGELE_TOHUM,
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("lr_l1r", 0.0, 1.0)
        return params

    if algo == "svm_rbf":
        return {
            "C":            trial.suggest_float("svm_C", 0.01, 100.0, log=True),
            "gamma":        trial.suggest_float("svm_gamma", 0.0001, 1.0, log=True),
            "class_weight": trial.suggest_categorical("svm_cw", ["balanced", None]),
            "kernel":       "rbf",
            "probability":  True,
            "random_state": RASTGELE_TOHUM,
        }

    if algo == "knn":
        return {
            "n_neighbors": trial.suggest_int("knn_k", 3, 30),
            "weights":     trial.suggest_categorical("knn_w", ["uniform", "distance"]),
            "p":           trial.suggest_categorical("knn_p", [1, 2]),
            "n_jobs":      1,
        }

    if algo == "naive_bayes":
        return {
            "var_smoothing": trial.suggest_float("nb_vs", 1e-12, 1e-3, log=True),
        }

    if algo == "mlp":
        n_layers = trial.suggest_int("mlp_layers", 1, 3)
        layer_size = trial.suggest_int("mlp_units", 16, 128)
        hidden = tuple([layer_size] * n_layers)
        return {
            "hidden_layer_sizes": hidden,
            "activation":         trial.suggest_categorical("mlp_act", ["relu", "tanh"]),
            "alpha":              trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("mlp_lri", 1e-4, 1e-1, log=True),
            "batch_size":         trial.suggest_categorical("mlp_bs", [32, 64, "auto"]),
            "max_iter":           500,
            "early_stopping":     True,
            "validation_fraction": 0.15,
            "random_state":       RASTGELE_TOHUM,
        }

    raise ValueError(f"Bilinmeyen algoritma: {algo}")

# ══════════════════════════════════════════════════════════════════
#  TRIAL → MODEL PARAMS (10 algo için merkezi rename + defaults)
#  v3.0'da bu fonksiyon FAZ 2, FAZ 4-A, FAZ 4-B ve JSON çıktısında kullanılır.
# ══════════════════════════════════════════════════════════════════

TRIAL_RENAME = {
    "xgboost": {
        "xgb_n_est": "n_estimators", "xgb_depth": "max_depth",
        "xgb_mcw": "min_child_weight", "xgb_mds": "max_delta_step",
        "xgb_lr": "learning_rate", "xgb_sub": "subsample",
        "xgb_cbt": "colsample_bytree", "xgb_cbl": "colsample_bylevel",
        "xgb_cbn": "colsample_bynode", "xgb_gamma": "gamma",
        "xgb_alpha": "reg_alpha", "xgb_lambda": "reg_lambda",
        "xgb_spw": "scale_pos_weight",
    },
    "lightgbm": {
        "lgb_n_est": "n_estimators", "lgb_leaves": "num_leaves",
        "lgb_depth": "max_depth", "lgb_mcs": "min_child_samples",
        "lgb_lr": "learning_rate", "lgb_sub": "subsample",
        "lgb_cbt": "colsample_bytree", "lgb_alpha": "reg_alpha",
        "lgb_lambda": "reg_lambda", "lgb_msg": "min_split_gain",
        "lgb_spw": "scale_pos_weight",
    },
    "catboost": {
        "cb_iter": "iterations", "cb_depth": "depth",
        "cb_lr": "learning_rate", "cb_l2": "l2_leaf_reg",
        "cb_mdl": "min_data_in_leaf", "cb_sub": "subsample",
        "cb_cbl": "colsample_bylevel", "cb_spw": "scale_pos_weight",
    },
    "random_forest": {
        "rf_n_est": "n_estimators", "rf_depth": "max_depth",
        "rf_mss": "min_samples_split", "rf_msl": "min_samples_leaf",
        "rf_mf": "max_features", "rf_cw": "class_weight",
    },
    "extra_trees": {
        "et_n_est": "n_estimators", "et_depth": "max_depth",
        "et_mss": "min_samples_split", "et_msl": "min_samples_leaf",
        "et_mf": "max_features", "et_cw": "class_weight",
    },
    "logistic": {
        "lr_C": "C", "lr_pen": "penalty", "lr_l1r": "l1_ratio",
        "lr_cw": "class_weight",
    },
    "svm_rbf": {
        "svm_C": "C", "svm_gamma": "gamma", "svm_cw": "class_weight",
    },
    "knn": {
        "knn_k": "n_neighbors", "knn_w": "weights", "knn_p": "p",
    },
    "naive_bayes": {
        "nb_vs": "var_smoothing",
    },
}

def trial_to_model_params(algo, trial_params, n_jobs_override=None):
    """
    Optuna ham params (trial.params) → model_fit_universal'ın beklediği dict.
    Tüm gerekli defaults (random_state, enable_categorical, vb.) eklenir.
    """
    if algo == "mlp":
        n_layers   = trial_params.get("mlp_layers", 1)
        layer_size = trial_params.get("mlp_units", 64)
        return {
            "hidden_layer_sizes":  tuple([layer_size] * n_layers),
            "activation":          trial_params.get("mlp_act", "relu"),
            "alpha":               trial_params.get("mlp_alpha", 1e-4),
            "learning_rate_init":  trial_params.get("mlp_lri", 1e-3),
            "batch_size":          trial_params.get("mlp_bs", "auto"),
            "max_iter":            500,
            "early_stopping":      True,
            "validation_fraction": 0.15,
            "random_state":        RASTGELE_TOHUM,
        }

    rd = TRIAL_RENAME.get(algo, {})
    p = {rd[k]: v for k, v in trial_params.items() if k in rd}

    if algo == "xgboost":
        p.update({"enable_categorical": True, "tree_method": "hist", "device": "cpu",
                  "verbosity": 0, "random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})
    elif algo == "lightgbm":
        p.update({"verbosity": -1, "random_state": RASTGELE_TOHUM, "subsample_freq": 1,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})
    elif algo == "catboost":
        p.update({"random_seed": RASTGELE_TOHUM, "verbose": False,
                  "thread_count": n_jobs_override if n_jobs_override else 1,
                  "bootstrap_type": "Bernoulli"})
    elif algo == "random_forest":
        p.update({"random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})
    elif algo == "extra_trees":
        p.update({"random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})
    elif algo == "logistic":
        p.update({"solver": "saga", "max_iter": 3000, "random_state": RASTGELE_TOHUM})
    elif algo == "svm_rbf":
        p.update({"kernel": "rbf", "probability": True, "random_state": RASTGELE_TOHUM})
    elif algo == "knn":
        p.update({"n_jobs": n_jobs_override if n_jobs_override else 1})

    return p

# ══════════════════════════════════════════════════════════════════
#  2. SHAP ÖZELLİK BUDAMA (v2.8 FIX #2 + #13 + #18)
# ══════════════════════════════════════════════════════════════════
print(f"\n[2/9] SHAP özellik budama (alt %{SHAP_PERCENTILE} atılıyor)...")
if SHAP_OK and SHAP_PERCENTILE > 0:
    try:
        SHAP_XGB_PARAMS = {
            "enable_categorical": True, "tree_method": "hist", "device": "cpu",
            "verbosity": 0, "n_jobs": -1, "random_state": RASTGELE_TOHUM,
            "n_estimators": 200, "max_depth": 6,
            "learning_rate": 0.05, "subsample": 0.8, "scale_pos_weight": SPW,
        }
        print("  SHAP referans modeli (XGBoost) eğitiliyor...")
        shap_model = xgb.XGBClassifier(**SHAP_XGB_PARAMS)
        shap_model.fit(X_train, y_train)

        explainer   = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_train)
        # v2.8 FIX #13: yeni SHAP API 3D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        all_cols      = list(X_train.columns)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        print("\n  Top 10 SHAP Özelliği:")
        onem_sirasi = np.argsort(mean_abs_shap)[::-1]
        shap_max = max(float(mean_abs_shap.max()), 1e-9)   # FIX #18
        for rk in onem_sirasi[:10]:
            bar = "█" * int(mean_abs_shap[rk] / shap_max * 35)
            print(f"    {all_cols[rk]:<30} {bar}  {mean_abs_shap[rk]:.5f}")

        # [FIX J] Yüzdelik temelli ">=" karşılaştırması: çok sayıda özellik
        # aynı SHAP değerine sahipse (örn. çoğu 0.0) yüzdelik 0.0'a düşer ve
        # tüm özellikler tutulur. Top-K ile garanti %20 budama.
        n_kalan = max(1, int(len(mean_abs_shap) * (1 - SHAP_PERCENTILE / 100)))
        top_idx = np.argsort(mean_abs_shap)[::-1][:n_kalan]
        kalan_mask = np.zeros(len(mean_abs_shap), dtype=bool)
        kalan_mask[top_idx] = True
        shap_esik  = float(mean_abs_shap[top_idx[-1]])  # gerçek alt eşik
        kalan_cols = [c for c, keep in zip(all_cols, kalan_mask) if keep]
        budanan_cnt = len(all_cols) - len(kalan_cols)

        X_train       = X_train[kalan_cols]
        X_test        = X_test[kalan_cols]
        KAT_KOLONLAR  = [c for c in KAT_KOLONLAR if c in kalan_cols]
        BUDANAN_OZELLIKLER = [c for c in all_cols if c not in kalan_cols]

        print(f"\n  {len(kalan_cols)} özellik kaldı ({budanan_cnt} budandı)")
        del shap_values, shap_model, explainer
        gc.collect()
    except Exception as e:
        print(f"  [UYARI] SHAP budama başarısız: {e}")
        BUDANAN_OZELLIKLER = []
else:
    print("  SHAP budama devre dışı.")

# ══════════════════════════════════════════════════════════════════
#  3. FAZ 1 — 10 ALGORİTMA GENİŞ TARAMA
#     Her algoritma: ~30-80 trial × 100 fold OOF
#     Çıktı: her algonun mean F2, std F2, top-N fold skor listesi
# ══════════════════════════════════════════════════════════════════
print(f"\n[3/9] FAZ 1: {len(TUM_ALGORITMALAR)} algoritma geniş tarama")
print("-" * 72)

def faz1_objective_factory(algo):
    """Bir algoritma için Optuna objective fonksiyonu üretir."""
    def objective(trial):
        params = suggest_params(trial, algo)
        rskf = RepeatedStratifiedKFold(n_splits=FAZ1_FOLDS, n_repeats=FAZ1_TEKRAR,
                                       random_state=RASTGELE_TOHUM)
        f2_listesi  = []
        f1_listesi  = []   # v3.5: sadece tabloda raporlamak için (eleme/seçim DEĞİL)
        auc_listesi = []
        # Pruner için ara rapor adımları
        toplam_fold = FAZ1_FOLDS * FAZ1_TEKRAR
        RAPOR_ADIMI = {toplam_fold // 4, toplam_fold // 2, 3 * toplam_fold // 4}

        for fold_no, (tr_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
            # [FIX] ES ayrımı yalnız boosting algos için anlamlı; diğerleri için
            # %10 veri kaybı önlenir. 388 örnekte her örnek değerli.
            if algo in ("xgboost", "lightgbm", "catboost"):
                X_fold_tr, y_fold_tr, X_fold_es, y_fold_es = fold_es_ayir(
                    X_train.iloc[tr_idx], y_train[tr_idx]
                )
            else:
                X_fold_tr = X_train.iloc[tr_idx]
                y_fold_tr = y_train[tr_idx]
                X_fold_es = y_fold_es = None
            try:
                m    = model_fit_universal(algo, params, X_fold_tr, y_fold_tr,
                                            X_fold_es, y_fold_es)
                prob = predict_proba_universal(m, algo, X_train.iloc[val_idx])
                esik, f2 = en_iyi_f2_esik_grid(y_train[val_idx], prob, adim=0.01)
                # Aynı eşikte F1'i de hesapla (sadece tablo için)
                pred = (prob >= esik).astype(int)
                f1 = f1_score(y_train[val_idx], pred, pos_label=1,
                              average="binary", zero_division=0)
                auc   = roc_auc_score(y_train[val_idx], prob)
            except Exception:
                return 0.0

            f2_listesi.append(f2)
            f1_listesi.append(f1)
            auc_listesi.append(auc)

            if fold_no in RAPOR_ADIMI:
                birlesik = float(np.mean([W_F2*f + W_AUC*a
                                          for f, a in zip(f2_listesi, auc_listesi)]))
                try:
                    trial.report(birlesik, step=fold_no)
                    if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                except optuna.exceptions.TrialPruned: raise
                except Exception: pass

        # Trial'a fold-bazlı F2 + F1 skorlarını user_attr olarak kaydet
        trial.set_user_attr("f2_scores", f2_listesi)
        trial.set_user_attr("f1_scores", f1_listesi)   # v3.5: tablo için
        return float(np.mean([W_F2*f + W_AUC*a
                              for f, a in zip(f2_listesi, auc_listesi)]))
    return objective

# Her algoritma için ayrı Optuna study çalıştır
faz1_sonuclari = {}   # {algo: {"mean": ..., "std": ..., "scores": [...], "best_params": ...}}

for algo in TUM_ALGORITMALAR:
    n_trial = FAZ1_TRIAL_PER_ALGO.get(algo, 30)
    db_url = f"{FAZ1_DB_PREFIX}_{algo}.db"

    print(f"\n  ── {algo.upper():<14} ─── {n_trial} trial × {FAZ1_FOLDS*FAZ1_TEKRAR} fold")

    study = optuna.create_study(
        study_name=f"faz1_{algo}",
        direction="maximize",
        storage=optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"connect_args": {"timeout": 360,
                                            "check_same_thread": False}},
        ),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RASTGELE_TOHUM,
                                            n_startup_trials=min(20, n_trial//3),
                                            multivariate=True,
                                            warn_independent_sampling=False),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                            n_warmup_steps=3, interval_steps=3),
    )

    mevcut = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    kalan  = max(0, n_trial - mevcut)

    if kalan > 0:
        t0 = time.perf_counter()
        # ŞEBNUS ÖLÜMSÜZ MODU
        while kalan > 0:
            try:
                study.optimize(faz1_objective_factory(algo), n_trials=kalan,
                                n_jobs=PARALEL_ISCI, show_progress_bar=True,
                                catch=(Exception,))
                break
            except Exception as e:
                if "Cannot tell" in str(e) or "database is locked" in str(e):
                    print(f"\n    [ŞEBNUS] {e}")
                    time.sleep(1)
                else:
                    raise e
            mevcut = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            kalan  = max(0, n_trial - mevcut)
        print(f"    Bitti: {time.perf_counter()-t0:.1f}s")

    # En iyi trial → fold-bazlı F2 dağılımı
    try:
        best_t = study.best_trial
    except Exception:
        print(f"    [UYARI] {algo} için tamamlanan trial yok, atlanıyor.")
        continue

    f2_scores = best_t.user_attrs.get("f2_scores", [])
    f1_scores = best_t.user_attrs.get("f1_scores", [])  # v3.5: tablo için
    if not f2_scores:
        # user_attr kayıp olabilir; baz veri ile doldur
        f2_scores = [best_t.value]
    if not f1_scores:
        f1_scores = [0.0]

    faz1_sonuclari[algo] = {
        "mean_f2":     float(np.mean(f2_scores)),
        "std_f2":      float(np.std(f2_scores)),
        "lcb_f2":      float(np.mean(f2_scores) - 0.5 * np.std(f2_scores)),
        "mean_f1":     float(np.mean(f1_scores)),   # v3.5: kıyas için
        "std_f1":      float(np.std(f1_scores)),
        "best_score":  float(best_t.value),
        "scores":      f2_scores,
        "best_params": dict(best_t.params),
        "n_trials":    mevcut + (kalan if kalan == 0 else 0),
    }

    print(f"    Sonuç: mean F2={np.mean(f2_scores):.4f}  std={np.std(f2_scores):.4f}"
          f"  LCB={np.mean(f2_scores)-0.5*np.std(f2_scores):.4f}"
          f"  | F1={np.mean(f1_scores):.4f}")

# ── FAZ 1 ELEME: Hibrit Üçlü Kriter ───────────────────────────────
print("\n  ── FAZ 1 ELEME (Hibrit Üçlü Kriter) ────────────────────────")

if not faz1_sonuclari:
    sys.exit("  [HATA] FAZ 1'de hiçbir algoritma tamamlanmadı.")

# Top performer (mean F2'ye göre)
top_algo = max(faz1_sonuclari, key=lambda a: faz1_sonuclari[a]["mean_f2"])
top_scores = faz1_sonuclari[top_algo]["scores"]
print(f"  Top performer: {top_algo.upper()} (mean F2={faz1_sonuclari[top_algo]['mean_f2']:.4f})")

hayatta_kalan = []
eleme_raporu  = []

for algo, sonuc in faz1_sonuclari.items():
    # Üçlü eleme kriteri
    mean_f2 = sonuc["mean_f2"]
    lcb_f2  = sonuc["lcb_f2"]

    # Welch's t-test
    if algo == top_algo:
        t_pval = 1.0  # kendisi
    else:
        try:
            _, t_pval = ttest_ind(top_scores, sonuc["scores"], equal_var=False)
        except Exception:
            t_pval = 1.0

    kriter1 = mean_f2 < FAZ1_F2_ORT_ESIK
    kriter2 = lcb_f2  < FAZ1_F2_LCB_ESIK
    # [FIX L] NaN p_value = sabit tahmin / crash. Python'da nan < 0.05 = False
    # döner, bu nedenle kriter3 yanlışlıkla False çıkar ve crash olan algo
    # "GEÇTİ" gösterilir. NaN'ı "anlamlı şekilde farklı" sayalım.
    if np.isnan(t_pval):
        kriter3 = True
    else:
        kriter3 = t_pval < FAZ1_TTEST_P_ESIK

    elendi = kriter1 and kriter2 and kriter3
    durum  = "ELENDİ" if elendi else "GEÇTİ"

    sonuc["t_pval"]  = float(t_pval)
    sonuc["elendi"]  = bool(elendi)

    eleme_raporu.append({
        "algo": algo, "mean_f2": mean_f2, "std_f2": sonuc["std_f2"],
        "lcb_f2": lcb_f2, "mean_f1": sonuc.get("mean_f1", 0.0),
        "t_pval": t_pval, "durum": durum,
    })

    if not elendi:
        hayatta_kalan.append(algo)

# Eleme tablosunu yazdır (F1 sadece kıyaslama için, eleme F2 ile)
print(f"\n  {'Algoritma':<14}  {'F2_ort':>8}  {'F2_std':>8}  "
      f"{'LCB':>8}  {'F1_ort':>8}  {'p_value':>8}  {'Durum':>8}")
print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for r in sorted(eleme_raporu, key=lambda x: x["mean_f2"], reverse=True):
    print(f"  {r['algo']:<14}  {r['mean_f2']:>8.4f}  {r['std_f2']:>8.4f}  "
          f"{r['lcb_f2']:>8.4f}  {r['mean_f1']:>8.4f}  "
          f"{r['t_pval']:>8.4f}  {r['durum']:>8}")

print(f"\n  Hayatta kalan: {len(hayatta_kalan)} algoritma → {hayatta_kalan}")

# FAZ 1 CSV çıktısı
faz1_df = pd.DataFrame(eleme_raporu)
faz1_df["paradigma"] = faz1_df["algo"].map(ALGO_PARADIGMA)
faz1_df = faz1_df.sort_values("mean_f2", ascending=False)
faz1_df.to_csv(CIKTI_FAZ1_CSV, index=False, encoding="utf-8")
print(f"  FAZ 1 tablosu '{CIKTI_FAZ1_CSV}' kaydedildi.")

if len(hayatta_kalan) < FAZ2_SECILEN_SAYI:
    print(f"\n  [UYARI] Yalnız {len(hayatta_kalan)} algoritma geçti, "
          f"FAZ 2 için {FAZ2_SECILEN_SAYI} gerekiyor. Top {FAZ2_SECILEN_SAYI} alınıyor.")
    hayatta_kalan = sorted(faz1_sonuclari, key=lambda a: faz1_sonuclari[a]["mean_f2"],
                           reverse=True)[:FAZ2_SECILEN_SAYI]

# ══════════════════════════════════════════════════════════════════
#  4. FAZ 2 — KORELASYON + ÇEŞİTLİLİK SEÇİMİ → TOP 3 ALGORİTMA
# ══════════════════════════════════════════════════════════════════
print(f"\n[4/9] FAZ 2: Korelasyon + Çeşitlilik Seçimi")
print("-" * 72)
print(f"  {len(hayatta_kalan)} hayatta kalan algoritma için OOF tahminleri "
      f"({FAZ2_FOLDS_OOF*FAZ2_TEKRAR_OOF} fold)...")

# Her hayatta kalan için OOF tahmini (best_params ile)
oof_proba_per_algo = {}
oof_f2_per_algo    = {}

for algo in hayatta_kalan:
    best_params = faz1_sonuclari[algo]["best_params"]

    # [FIX] Merkezi trial_to_model_params fonksiyonu kullanılıyor.
    # v3.0'ın ilk halinde burada inline rename_dict vardı — duplikasyon
    # ve tutarsızlık riski yarattı. Artık tek bir kaynak (TRIAL_RENAME).
    model_params = trial_to_model_params(algo, best_params, n_jobs_override=-1)

    print(f"  OOF [{algo:<14}] ...", end="", flush=True)
    t0 = time.perf_counter()

    proba = oof_tahmin_universal(
        algo, model_params, X_train, y_train,
        FAZ2_FOLDS_OOF, FAZ2_TEKRAR_OOF,
        n_jobs_override=-1,
        es_split=(FAZ2_OOF_ES_SPLIT if algo in ("xgboost", "lightgbm", "catboost") else 0.0),
    )
    _, f2 = en_iyi_f2_esik_benzersiz(y_train, proba)

    oof_proba_per_algo[algo] = proba
    oof_f2_per_algo[algo]    = f2

    print(f" OOF_F2={f2:.4f}  {time.perf_counter()-t0:.1f}s")
    gc.collect()

# ── Korelasyon Matrisi ────────────────────────────────────────────
print("\n  Korelasyon Matrisi:")
korelasyon_df = pd.DataFrame(
    {algo: oof_proba_per_algo[algo] for algo in hayatta_kalan}
).corr()
# [FIX] Sabit-tahmin (prior fallback) durumunda corr() NaN üretir.
# Greedy seçimde NaN → max() davranışı tanımsız → yanlış algo seçilebilir.
# NaN'leri 1.0 ile doldur: "tam korele" gibi davran → çeşitlilik için seçilmez.
nan_count = korelasyon_df.isna().sum().sum()
if nan_count > 0:
    print(f"  [UYARI] {nan_count} NaN korelasyon değeri var (sabit-tahmin algos), 1.0 ile dolduruluyor.")
    korelasyon_df = korelasyon_df.fillna(1.0)
print(korelasyon_df.round(3).to_string())
korelasyon_df.to_csv(CIKTI_KORELASYON_CSV, encoding="utf-8")
print(f"\n  Korelasyon matrisi '{CIKTI_KORELASYON_CSV}' kaydedildi.")

# ── Greedy Çeşitlilik Seçimi ──────────────────────────────────────
# 1) En yüksek F2'li algoritmayı seç
# 2) Sonra: kalan adaylar arasında, seçilenlerle ORTALAMA korelasyonu en düşük olanı ekle
# 3) FAZ2_SECILEN_SAYI tamamlanana kadar tekrarla
print("\n  Greedy Çeşitlilik Seçimi:")

secilen_algoritmalar = []
en_iyi_f2_algo = max(oof_f2_per_algo, key=oof_f2_per_algo.get)
secilen_algoritmalar.append(en_iyi_f2_algo)
print(f"    1. {en_iyi_f2_algo.upper()} (en yüksek F2: {oof_f2_per_algo[en_iyi_f2_algo]:.4f})")

while len(secilen_algoritmalar) < FAZ2_SECILEN_SAYI:
    adaylar = [a for a in hayatta_kalan if a not in secilen_algoritmalar]
    if not adaylar:
        break

    # Her aday için seçilenlerle ortalama korelasyonu hesapla
    ort_kor = {}
    for a in adaylar:
        kor_listesi = [abs(korelasyon_df.loc[a, s]) for s in secilen_algoritmalar]
        ort_kor[a] = float(np.mean(kor_listesi))

    # F2 × (1 - ort_korelasyon) bileşik skoru — hem skor hem çeşitlilik ister
    bilesik = {a: oof_f2_per_algo[a] * (1 - ort_kor[a]) for a in adaylar}
    en_iyi = max(bilesik, key=bilesik.get)

    print(f"    {len(secilen_algoritmalar)+1}. {en_iyi.upper()}  "
          f"(F2={oof_f2_per_algo[en_iyi]:.4f}, "
          f"ort_kor={ort_kor[en_iyi]:.3f}, "
          f"bileşik={bilesik[en_iyi]:.4f})")
    secilen_algoritmalar.append(en_iyi)

print(f"\n  FİNAL {len(secilen_algoritmalar)} ALGORİTMA: {secilen_algoritmalar}")
print(f"  Paradigmalar: "
      f"{[ALGO_PARADIGMA[a] for a in secilen_algoritmalar]}")

# ══════════════════════════════════════════════════════════════════
#  5. FAZ 3 — SEÇİLEN 3 ALGORİTMANIN DERİN TUNING'İ
#     Her algoritma için bağımsız Optuna study (TPE multivariate)
#     ~1500 trial × 50 fold = derin arama
# ══════════════════════════════════════════════════════════════════
print(f"\n[5/9] FAZ 3: {len(secilen_algoritmalar)} algoritmanın derin tuning'i")
print("-" * 72)

def faz3_objective_factory(algo):
    """FAZ 3 için tek-algoritma Optuna objective."""
    def objective(trial):
        params = suggest_params(trial, algo)
        rskf = RepeatedStratifiedKFold(n_splits=FAZ3_FOLDS, n_repeats=FAZ3_TEKRAR,
                                       random_state=RASTGELE_TOHUM)
        f2_listesi  = []
        auc_listesi = []
        toplam_fold = FAZ3_FOLDS * FAZ3_TEKRAR
        RAPOR_ADIMI = {toplam_fold // 4, toplam_fold // 2}

        for fold_no, (tr_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
            # [FIX] ES ayrımı yalnız boosting algos için
            if algo in ("xgboost", "lightgbm", "catboost"):
                X_fold_tr, y_fold_tr, X_fold_es, y_fold_es = fold_es_ayir(
                    X_train.iloc[tr_idx], y_train[tr_idx]
                )
            else:
                X_fold_tr = X_train.iloc[tr_idx]
                y_fold_tr = y_train[tr_idx]
                X_fold_es = y_fold_es = None
            try:
                m    = model_fit_universal(algo, params, X_fold_tr, y_fold_tr,
                                            X_fold_es, y_fold_es)
                prob = predict_proba_universal(m, algo, X_train.iloc[val_idx])
                _, f2 = en_iyi_f2_esik_grid(y_train[val_idx], prob, adim=0.005)
                auc   = roc_auc_score(y_train[val_idx], prob)
            except Exception:
                return 0.0

            f2_listesi.append(f2)
            auc_listesi.append(auc)

            if fold_no in RAPOR_ADIMI:
                birlesik = float(np.mean([W_F2*f + W_AUC*a
                                          for f, a in zip(f2_listesi, auc_listesi)]))
                try:
                    trial.report(birlesik, step=fold_no)
                    if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                except optuna.exceptions.TrialPruned: raise
                except Exception: pass

        return float(np.mean([W_F2*f + W_AUC*a
                              for f, a in zip(f2_listesi, auc_listesi)]))
    return objective

# Her algoritma için ayrı study
faz3_studies = {}

for algo in secilen_algoritmalar:
    db_url = f"{FAZ3_DB_PREFIX}_{algo}.db"
    # [FIX N] Algo-bazlı bütçe (sklearn'lerden tasarruf)
    algo_trial_budget = FAZ3_TRIAL_PER_ALGO.get(algo, 500)
    print(f"\n  ── {algo.upper():<14} ─── {algo_trial_budget} trial × {FAZ3_FOLDS*FAZ3_TEKRAR} fold")

    study = optuna.create_study(
        study_name=f"faz3_{algo}",
        direction="maximize",
        storage=optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"connect_args": {"timeout": 360,
                                            "check_same_thread": False}},
        ),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RASTGELE_TOHUM + 100,
                                            n_startup_trials=60,
                                            multivariate=True,
                                            warn_independent_sampling=False),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=40,
                                            n_warmup_steps=5, interval_steps=5),
    )

    # FAZ 1'in en iyi parametrelerini seed olarak kuyruğa ekle
    if algo in faz1_sonuclari:
        try:
            study.enqueue_trial(faz1_sonuclari[algo]["best_params"])
            print(f"    FAZ 1 best params seed olarak eklendi.")
        except Exception as e:
            print(f"    [UYARI] Seed eklenemedi: {e}")

    mevcut = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    kalan  = max(0, algo_trial_budget - mevcut)

    if kalan > 0:
        t0 = time.perf_counter()
        while kalan > 0:
            try:
                study.optimize(faz3_objective_factory(algo), n_trials=kalan,
                                n_jobs=PARALEL_ISCI, show_progress_bar=True,
                                catch=(Exception,))
                break
            except Exception as e:
                if "Cannot tell" in str(e) or "database is locked" in str(e):
                    print(f"\n    [ŞEBNUS] {e}")
                    time.sleep(1)
                else:
                    raise e
            mevcut = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            kalan  = max(0, algo_trial_budget - mevcut)
        print(f"    Bitti: {time.perf_counter()-t0:.1f}s")

    faz3_studies[algo] = study
    try:
        print(f"    En iyi CV skor: {study.best_value:.4f}")
    except Exception:
        pass

# ── Her algoritmadan TOP-K çeşitli trial seç ──────────────────────
print(f"\n  Her algoritmadan top-{FAZ3_TOP_K} çeşitli trial seçiliyor...")

def parametre_mesafe_basit(t1, t2):
    """Aynı algonun iki trial'ı arasındaki normalize edilmiş L2 mesafesi."""
    farklar = []
    keys = set(t1.params.keys()) | set(t2.params.keys())
    for k in keys:
        v1 = t1.params.get(k, None)
        v2 = t2.params.get(k, None)
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, str) or isinstance(v2, str):
            farklar.append(0.0 if v1 == v2 else 1.0)
        else:
            try:
                a, b = float(v1), float(v2)
                denom = max(abs(a), abs(b), 1e-9)
                farklar.append(((a - b) / denom) ** 2)
            except (TypeError, ValueError):
                farklar.append(0.0 if v1 == v2 else 1.0)
    return float(np.sqrt(np.mean(farklar))) if farklar else 0.0

MIN_MESAFE_ICIN_FAZ3 = 0.03
FAZ3_KALITE_ESIK     = 0.30   # bu skorun altındaki trial'lar ensemble'a alınmaz
faz3_top_trials_per_algo = {}

for algo, study in faz3_studies.items():
    tamamlanan = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not tamamlanan:
        print(f"  [UYARI] {algo} için tamamlanan trial yok.")
        faz3_top_trials_per_algo[algo] = []
        continue

    # [FIX] Kalite filtresi: 0.0 dönen (tüm foldlar hata) trial'lar elensin
    kaliteli = [t for t in tamamlanan if t.value >= FAZ3_KALITE_ESIK]
    if not kaliteli:
        print(f"  [UYARI] {algo} için kaliteli trial (≥{FAZ3_KALITE_ESIK}) yok, "
              f"en iyi {min(FAZ3_TOP_K, len(tamamlanan))} kullanılıyor.")
        kaliteli = tamamlanan
    elif len(kaliteli) < len(tamamlanan):
        print(f"  {algo}: {len(tamamlanan)-len(kaliteli)} düşük-kalite trial filtrelendi.")

    sirali = sorted(kaliteli, key=lambda t: t.value, reverse=True)

    secilen = [sirali[0]]
    for tc in sirali[1:]:
        if len(secilen) >= FAZ3_TOP_K: break
        if all(parametre_mesafe_basit(tc, s) >= MIN_MESAFE_ICIN_FAZ3 for s in secilen):
            secilen.append(tc)
    # Yetersiz çeşitlilik → kalanları doldur
    for tc in sirali:
        if len(secilen) >= FAZ3_TOP_K: break
        if tc not in secilen:
            secilen.append(tc)

    faz3_top_trials_per_algo[algo] = secilen
    print(f"  {algo:<14}: {len(secilen)} trial seçildi, "
          f"CV skorları: {[round(t.value, 4) for t in secilen]}")

# Tüm ensemble üyeleri: 3 algo × 5 trial = 15
secilen_trials = []
secilen_trials_algo = []
for algo in secilen_algoritmalar:
    for t in faz3_top_trials_per_algo[algo]:
        secilen_trials.append(t)
        secilen_trials_algo.append(algo)

print(f"\n  TOPLAM ENSEMBLE: {len(secilen_trials)} üye "
      f"({len(secilen_algoritmalar)} algo × ortalama "
      f"{len(secilen_trials)//len(secilen_algoritmalar)} trial)")

if not secilen_trials:
    sys.exit("  [HATA] FAZ 3'te hiç ensemble üyesi seçilemedi.")

# TRIAL_RENAME ve trial_to_model_params yukarıda (suggest_params'tan sonra) tanımlı.

# ══════════════════════════════════════════════════════════════════
#  6. FAZ 4-A — ENSEMBLE ÜYELERİ İÇİN OOF TAHMİNİ
# ══════════════════════════════════════════════════════════════════
print(f"\n[6/9] FAZ 4-A: {len(secilen_trials)} ensemble üyesi için OOF")
print(f"  {OOF_FOLDS*OOF_TEKRAR} fold (ES split: %{int(OOF_ES_SPLIT*100)})...")

model_oof_proba_listesi = []
model_oof_f2_listesi    = []

for i, (t, algo) in enumerate(zip(secilen_trials, secilen_trials_algo)):
    model_params = trial_to_model_params(algo, t.params, n_jobs_override=-1)
    es_use = OOF_ES_SPLIT if algo in ("xgboost", "lightgbm", "catboost") else 0.0

    print(f"  OOF [{i+1:2d}/{len(secilen_trials)}] {algo:<14} CV={t.value:.4f} ...",
          end="", flush=True)
    t0 = time.perf_counter()

    proba_i = oof_tahmin_universal(algo, model_params, X_train, y_train,
                                    OOF_FOLDS, OOF_TEKRAR,
                                    n_jobs_override=-1, es_split=es_use)
    _, f2_i = en_iyi_f2_esik_benzersiz(y_train, proba_i)

    model_oof_proba_listesi.append(proba_i)
    model_oof_f2_listesi.append(f2_i)
    print(f" OOF_F2={f2_i:.4f}  {time.perf_counter()-t0:.1f}s")
    gc.collect()

# ── Softmax Ağırlıkları + Rank/Weighted Ensemble ──────────────────
agirliklar = softmax_agirliklar(model_oof_f2_listesi, SOFTMAX_SICAKLIK)
print(f"\n  Softmax Ensemble Ağırlıkları (sıcaklık={SOFTMAX_SICAKLIK}):")
for i, (algo, w, f2) in enumerate(zip(secilen_trials_algo, agirliklar,
                                       model_oof_f2_listesi)):
    bar = "█" * max(1, int(w * 55))
    print(f"    [{i+1:2d}] {algo:<14}  F2={f2:.4f}  w={w:.4f}  {bar}")

oof_rank_proba     = rank_ensemble(model_oof_proba_listesi)
oof_weighted_proba = sum(w * p for w, p in zip(agirliklar, model_oof_proba_listesi))

_, f2_rank     = en_iyi_f2_esik_benzersiz(y_train, oof_rank_proba)
_, f2_weighted = en_iyi_f2_esik_benzersiz(y_train, oof_weighted_proba)

ensemble_oof_proba = oof_rank_proba if f2_rank >= f2_weighted else oof_weighted_proba
SECILEN_YONTEM     = "rank_averaging" if f2_rank >= f2_weighted else "softmax_weighted"
print(f"\n  Seçilen ensemble yöntemi: {SECILEN_YONTEM}  "
      f"(rank={f2_rank:.4f}  weighted={f2_weighted:.4f})")

# ── Isotonic Kalibrasyon + Eşik ───────────────────────────────────
print("  Isotonic Regression kalibrasyonu...")
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(ensemble_oof_proba, y_train)
kalibre_oof_proba    = iso_reg.predict(ensemble_oof_proba)
OPTIMAL_ESIK, oof_f2 = en_iyi_f2_esik_benzersiz(y_train, kalibre_oof_proba)

oof_tahminler = (kalibre_oof_proba >= OPTIMAL_ESIK).astype(int)
oof_f1        = f1_score(y_train, oof_tahminler, pos_label=1,
                          average="binary", zero_division=0)
try:
    oof_auc = roc_auc_score(y_train, kalibre_oof_proba)
except ValueError:
    oof_auc = 0.0
oof_mcc       = matthews_corrcoef(y_train, oof_tahminler)

print(f"\n  {'─'*56}")
print(f"  OOF F2  (kalibre, leakage-free) : {oof_f2:.4f}")
print(f"  OOF F1                          : {oof_f1:.4f}")
print(f"  OOF AUC                         : {oof_auc:.4f}")
print(f"  OOF MCC                         : {oof_mcc:.4f}")
print(f"  Optimal Eşik                    : {OPTIMAL_ESIK:.5f}")
print(f"  {'─'*56}")

# ══════════════════════════════════════════════════════════════════
#  7. FAZ 4-B — FİNAL MODELLER + TEST DEĞERLENDİRMESİ
# ══════════════════════════════════════════════════════════════════
print(f"\n[7/9] FAZ 4-B: {len(secilen_trials)} final model + test")

final_test_proba_listesi = []

for i, (t, algo) in enumerate(zip(secilen_trials, secilen_trials_algo)):
    model_params = trial_to_model_params(algo, t.params, n_jobs_override=-1)

    # [FIX] Non-boosting için %8 ES split israfı önlendi; tam X_train kullanılıyor.
    # Boosting için stratify-korumalı split, ES'le early stopping yapılır.
    if algo in ("xgboost", "lightgbm", "catboost"):
        es_size = max(5, int(N_TRAIN * 0.08))
        try:
            X_fin_tr, X_fin_es, y_fin_tr, y_fin_es = train_test_split(
                X_train, y_train, test_size=es_size,
                random_state=RASTGELE_TOHUM + i, stratify=y_train,
            )
        except ValueError:
            X_fin_tr, X_fin_es, y_fin_tr, y_fin_es = train_test_split(
                X_train, y_train, test_size=es_size, random_state=RASTGELE_TOHUM + i,
            )
    else:
        # Non-boosting → full data, no ES
        X_fin_tr = X_train
        y_fin_tr = y_train
        X_fin_es = y_fin_es = None

    print(f"  Final [{i+1:2d}/{len(secilen_trials)}] {algo:<14} ...",
          end="", flush=True)
    t0 = time.perf_counter()

    # FIX #15: try/except + prior fallback (liste uyumu korunuyor)
    try:
        fm = model_fit_universal(algo, model_params, X_fin_tr, y_fin_tr,
                                  X_fin_es, y_fin_es)
        test_proba = predict_proba_universal(fm, algo, X_test)
    except Exception as e:
        print(f"\n    [UYARI] Final model başarısız ({e}), prior kullanılıyor.")
        test_proba = np.full(N_TEST, float(y_train.mean()))

    final_test_proba_listesi.append(test_proba)
    print(f" {time.perf_counter()-t0:.2f}s")
    gc.collect()

# Ensemble birleştirme (OOF'da seçilen yöntem)
if SECILEN_YONTEM == "rank_averaging":
    final_raw = rank_ensemble(final_test_proba_listesi)
else:
    final_raw = sum(w * p for w, p in zip(agirliklar, final_test_proba_listesi))

final_test_proba = iso_reg.predict(final_raw)

# Test metrikleri (v3.5: binary, pos_label=1)
test_tahminler = (final_test_proba >= OPTIMAL_ESIK).astype(int)
test_f2        = fbeta_score(y_test, test_tahminler, beta=2,
                              pos_label=1, average="binary", zero_division=0)
test_f1        = f1_score(y_test, test_tahminler, pos_label=1,
                           average="binary", zero_division=0)
try:
    test_auc = roc_auc_score(y_test, final_test_proba)
except ValueError:
    test_auc = 0.0
test_mcc       = matthews_corrcoef(y_test, test_tahminler)
test_acc       = accuracy_score(y_test, test_tahminler)
cm             = confusion_matrix(y_test, test_tahminler)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    test_recall    = tp / max(tp + fn, 1)
    test_precision = tp / max(tp + fp, 1)
    test_spec      = tn / max(tn + fp, 1)
else:
    test_recall = test_precision = test_spec = 0.0
    tn = fp = fn = tp = 0

# ─── BOOTSTRAP 95% GÜVEN ARALIKLARI (JÜRİ İÇİN İSTATİSTİKSEL RİGOR) ─────
print("\n  Bootstrap 95% Güven Aralıkları (2000 iterasyon)...")
_, test_f2_lo, test_f2_hi = bootstrap_metric_ci(
    y_test, final_test_proba,
    lambda yt, yp: fbeta_score(yt, yp, beta=2, pos_label=1,
                                average="binary", zero_division=0),
    threshold=OPTIMAL_ESIK,
)
_, test_f1_lo, test_f1_hi = bootstrap_metric_ci(
    y_test, final_test_proba,
    lambda yt, yp: f1_score(yt, yp, pos_label=1, average="binary",
                             zero_division=0),
    threshold=OPTIMAL_ESIK,
)
try:
    _, test_auc_lo, test_auc_hi = bootstrap_metric_ci(
        y_test, final_test_proba, roc_auc_score,
    )
except Exception:
    test_auc_lo = test_auc_hi = test_auc

# ─── PER-CLASS KLİNİK METRİKLER (POZİTİF SINIF = PATOJENİK) ─────────────
test_recall_path    = float(recall_score(y_test, test_tahminler,
                                          pos_label=1, zero_division=0))
test_precision_path = float(precision_score(y_test, test_tahminler,
                                             pos_label=1, zero_division=0))
test_f2_path        = float(fbeta_score(y_test, test_tahminler, beta=2,
                                         pos_label=1, average="binary",
                                         zero_division=0))
test_f1_path        = float(f1_score(y_test, test_tahminler,
                                      pos_label=1, average="binary",
                                      zero_division=0))

# Klinik yorumlama: kaçan patojenikler ve gereksiz alarmlar
fn_count = int(fn)   # KAÇIRILAN patojenik → klinik risk
fp_count = int(fp)   # YANLIŞ alarm → gereksiz takip/biyopsi
tp_count = int(tp)
tn_count = int(tn)
toplam_patojenik = int((y_test == 1).sum())
toplam_benign    = int((y_test == 0).sum())

print("\n" + "=" * 72)
print("  GERÇEK TEST SONUÇLARI  (X_test ilk ve tek kez)")
print("=" * 72)
print(f"  F2 (binary, pos=patojenik) : {test_f2:.4f}  [95% CI: {test_f2_lo:.4f}-{test_f2_hi:.4f}]  ←─ Ana Metrik")
print(f"  F1 (binary, pos=patojenik) : {test_f1:.4f}  [95% CI: {test_f1_lo:.4f}-{test_f1_hi:.4f}]")
print(f"  AUC                        : {test_auc:.4f}  [95% CI: {test_auc_lo:.4f}-{test_auc_hi:.4f}]")
print(f"  MCC                        : {test_mcc:.4f}")
print(f"  Doğruluk                   : %{test_acc*100:.2f}")
print(f"  Eşik                       : {OPTIMAL_ESIK:.5f}")
print("─" * 72)
print(f"  KLİNİK YORUMLAMA (Pozitif sınıf = Patojenik):")
print(f"    Patojenik Recall    : {test_recall_path:.4f}  "
      f"({tp_count}/{toplam_patojenik} doğru yakalandı)")
print(f"    Patojenik Precision : {test_precision_path:.4f}")
print(f"    Patojenik F2        : {test_f2_path:.4f}  (kaçırma riskini en aza indiren metrik)")
print(f"    Patojenik F1        : {test_f1_path:.4f}")
print(f"    KAÇIRILAN patojenik : {fn_count}  ←─ KLİNİK RİSK (false negative)")
print(f"    Yanlış alarm        : {fp_count}  (gereksiz takip, false positive)")
print(f"    Specificity (Benign): {test_spec:.4f}  ({tn_count}/{toplam_benign})")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
#  8. JÜRİ TABLOSU — Her Algoritmanın Karşılaştırması
# ══════════════════════════════════════════════════════════════════
print(f"\n[8/9] TEKNOFEST Jüri Karşılaştırma Tablosu")
print("=" * 72)

juri_tablosu = []
for algo in TUM_ALGORITMALAR:
    if algo not in faz1_sonuclari:
        continue
    s = faz1_sonuclari[algo]
    durum = "ELENDİ" if s.get("elendi", False) else "GEÇTİ"
    secildi = "★ FİNAL" if algo in secilen_algoritmalar else ("✓" if durum == "GEÇTİ" else "✗")
    # FAZ 3 sonuçları
    faz3_skor = None
    if algo in faz3_studies:
        try:
            faz3_skor = round(faz3_studies[algo].best_value, 4)
        except Exception:
            pass

    juri_tablosu.append({
        "algoritma":     algo,
        "paradigma":     ALGO_PARADIGMA[algo],
        "faz1_f2_ort":   round(s["mean_f2"], 4),
        "faz1_f2_std":   round(s["std_f2"], 4),
        "faz1_lcb":      round(s["lcb_f2"], 4),
        "faz1_f1_ort":   round(s.get("mean_f1", 0.0), 4),   # v3.5: kıyas için
        "faz1_p_value":  round(s.get("t_pval", 1.0), 4),
        "faz1_durum":    durum,
        "faz3_cv_skor":  faz3_skor,
        "ensemble":      secildi,
    })

# Sıralı yazdır
juri_tablosu_sirali = sorted(juri_tablosu, key=lambda x: x["faz1_f2_ort"], reverse=True)

print(f"\n  {'Algoritma':<14} {'Paradigma':<28} "
      f"{'F2_ort':>7} {'F2_std':>7} {'LCB':>7} {'F1_ort':>7} {'p_val':>6} "
      f"{'Faz3':>7} {'Sonuç':>10}")
print(f"  {'-'*14} {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*10}")
for r in juri_tablosu_sirali:
    faz3_str = f"{r['faz3_cv_skor']:.4f}" if r['faz3_cv_skor'] is not None else "  -  "
    print(f"  {r['algoritma']:<14} {r['paradigma']:<28} "
          f"{r['faz1_f2_ort']:>7.4f} {r['faz1_f2_std']:>7.4f} "
          f"{r['faz1_lcb']:>7.4f} {r['faz1_f1_ort']:>7.4f} "
          f"{r['faz1_p_value']:>6.4f} "
          f"{faz3_str:>7} {r['ensemble']:>10}")

# Korelasyon matrisi yazdır
print(f"\n  Hayatta Kalanlar Arası Korelasyon Matrisi:")
print(f"  {korelasyon_df.round(3).to_string()}")

print(f"\n  ENSEMBLE: {SECILEN_YONTEM}  |  Kalibrasyon: Isotonic")
print(f"  Seçilen 3 Algoritma: {secilen_algoritmalar}")
print(f"  Test F2={test_f2:.4f}  OOF F2={oof_f2:.4f}  Fark={test_f2-oof_f2:+.4f}")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
#  9. JSON ÇIKTI
# ══════════════════════════════════════════════════════════════════
print(f"\n[9/9] JSON çıktı kaydediliyor...")

# En iyi tek model (jüri "tek model" sorarsa)
best_ensemble_idx = int(np.argmax(model_oof_f2_listesi))
best_trial_obj    = secilen_trials[best_ensemble_idx]
best_algo         = secilen_trials_algo[best_ensemble_idx]
best_model_params = trial_to_model_params(best_algo, best_trial_obj.params)
# Internal/runtime parametreleri kaldır
for k in ["n_jobs", "thread_count", "verbosity", "verbose",
          "random_state", "random_seed", "enable_categorical",
          "tree_method", "device", "max_iter", "early_stopping",
          "validation_fraction", "probability"]:
    best_model_params.pop(k, None)

sonuc = {
    "_aciklama": {
        "surum"          : "BioSense AI v3.6 — 4-Faz + Binary F2/F1 + Algo-bazlı bütçe",
        "pipeline"       : [
            "Faz 1: 10 algo geniş tarama + hibrit istatistiksel eleme",
            "Faz 2: Korelasyon + greedy çeşitlilik seçimi → top 3",
            "Faz 3: Seçilen 3 algonun derin Optuna tuning'i",
            "Faz 4: Ensemble + isotonic kalibrasyon + test",
        ],
        "faz1_eleme_kriteri": (
            f"F2_ort < {FAZ1_F2_ORT_ESIK} VE "
            f"(F2_ort - 0.5×std) < {FAZ1_F2_LCB_ESIK} VE "
            f"Welch's t-test p < {FAZ1_TTEST_P_ESIK}"
        ),
        "secilen_algoritmalar" : secilen_algoritmalar,
        "secilen_paradigmalar" : [ALGO_PARADIGMA[a] for a in secilen_algoritmalar],
        "ensemble_uye_sayisi"  : len(secilen_trials),
        "ensemble_yontem"      : SECILEN_YONTEM,
        "kalibrasyon"          : "IsotonicRegression",
        "shap_budanan"         : len(BUDANAN_OZELLIKLER),
    },
    "faz1_tum_algoritmalar": juri_tablosu_sirali,
    "faz2_korelasyon_matrisi": {
        algo: {a2: round(float(korelasyon_df.loc[algo, a2]), 4)
               for a2 in korelasyon_df.columns}
        for algo in korelasyon_df.index
    },
    "faz2_secim_rationale": {
        "1_en_yuksek_f2": {
            "algo": secilen_algoritmalar[0],
            "f2":   round(oof_f2_per_algo[secilen_algoritmalar[0]], 4),
        },
        "2_3_cesitlilik": [
            {
                "algo": a,
                "f2": round(oof_f2_per_algo[a], 4),
                "ort_korelasyon": round(
                    float(np.mean([abs(korelasyon_df.loc[a, s])
                                   for s in secilen_algoritmalar[:i]])), 3
                ),
            }
            for i, a in enumerate(secilen_algoritmalar)
            if i > 0
        ],
    },
    "faz3_top_trials": {
        algo: [
            {
                "cv_skor":  round(t.value, 4),
                "params":   {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in trial_to_model_params(algo, t.params).items()
                    if k not in ("n_jobs", "thread_count", "verbosity", "verbose",
                                 "random_state", "random_seed", "enable_categorical",
                                 "tree_method", "device", "max_iter",
                                 "early_stopping", "validation_fraction",
                                 "probability")
                },
            }
            for t in faz3_top_trials_per_algo.get(algo, [])
        ]
        for algo in secilen_algoritmalar
    },
    "pdr_v7_kopyala": {
        "algorithm"    : best_algo,
        **{k: (round(v, 6) if isinstance(v, float) else v)
           for k, v in best_model_params.items()},
        "OPTIMAL_ESIK" : float(OPTIMAL_ESIK),
    },
    "ensemble_uyeleri": [
        {
            "sira":      i + 1,
            "algoritma": secilen_trials_algo[i],
            "cv_skor":   round(t.value, 4),
            "oof_f2":    round(model_oof_f2_listesi[i], 4),
            "agirlik":   round(float(agirliklar[i]), 6),
        }
        for i, t in enumerate(secilen_trials)
    ],
    "ensemble_optimal_esik": float(OPTIMAL_ESIK),
    "skorlar": {
        "test_f2"        : round(test_f2, 4),
        "test_f2_ci95"   : [round(test_f2_lo, 4), round(test_f2_hi, 4)],
        "test_f1"        : round(test_f1, 4),
        "test_f1_ci95"   : [round(test_f1_lo, 4), round(test_f1_hi, 4)],
        "test_auc"       : round(test_auc, 4),
        "test_auc_ci95"  : [round(test_auc_lo, 4), round(test_auc_hi, 4)],
        "test_mcc"       : round(test_mcc, 4),
        "test_acc_pct"   : round(test_acc * 100, 4),
        "test_recall"    : round(test_recall, 4),
        "test_precision" : round(test_precision, 4),
        "test_spec"      : round(test_spec, 4),
        "oof_f2"         : round(oof_f2, 4),
        "oof_f1"         : round(oof_f1, 4),
        "oof_auc"        : round(oof_auc, 4),
        "oof_mcc"        : round(oof_mcc, 4),
    },
    "klinik_metrikler": {
        "_aciklama": "Patojenik (class 1) için özel hesaplanan klinik metrikler. "
                     "False Negative = kaçırılan patojenik → klinik risk. "
                     "False Positive = gereksiz takip/biyopsi.",
        "patojenik_recall"       : round(test_recall_path, 4),
        "patojenik_precision"    : round(test_precision_path, 4),
        "patojenik_f2"           : round(test_f2_path, 4),
        "patojenik_f1"           : round(test_f1_path, 4),
        "true_positives"         : tp_count,
        "true_negatives"         : tn_count,
        "false_negatives"        : fn_count,
        "false_positives"        : fp_count,
        "toplam_patojenik_test"  : toplam_patojenik,
        "toplam_benign_test"     : toplam_benign,
        "yakalama_orani_pct"     : round(100.0 * tp_count / max(toplam_patojenik, 1), 2),
        "kacirilan_orani_pct"    : round(100.0 * fn_count / max(toplam_patojenik, 1), 2),
    },
    "veri": {
        "n_train"            : int(N_TRAIN),
        "n_test"             : int(N_TEST),
        "spw_baz"            : round(SPW, 3),
        "toplam_ozellik"     : int(len(OZELLIK)),
        "kalan_ozellik"      : int(len(X_train.columns)),
        "budanan_ozellik"    : int(len(BUDANAN_OZELLIKLER)),
    },
}

with open(CIKTI_JSON, "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=False, indent=2)

print(f"  '{CIKTI_JSON}' kaydedildi.")
print(f"  '{CIKTI_FAZ1_CSV}' (FAZ 1 tablosu)")
print(f"  '{CIKTI_KORELASYON_CSV}' (FAZ 2 korelasyon matrisi)")
print("\n" + "=" * 72)
print("  BioSense AI v3.6 — TAMAMLANDI")
print("=" * 72)