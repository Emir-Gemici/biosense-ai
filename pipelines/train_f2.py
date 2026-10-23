"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BioSense AI v4.1 — TEKNOFEST RAPORU İÇİN TAM PIPELINE                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v4.0 → v4.1 EKLEMESİ — TEKNOFEST ŞARTNAMESİ TAM UYUMU:                     ║
║   ✓ PR-AUC (average_precision_score) eklendi — ZORUNLU metrik               ║
║      Hem OOF hem test için bootstrap %95 CI ile                             ║
║   ✓ Karar eşiği tablosu — 13 farklı eşikte (0.10-0.70) tüm metrikler        ║
║      "Farklı karar eşiklerinin test sonuçları" şartı karşılandı             ║
║   ✓ 8 adet PNG görsel otomatik üretilir (grafikler/ klasörü):              ║
║      - SHAP top 15 özellik (açıklanabilirlik)                              ║
║      - FAZ 1 algoritma karşılaştırma bar chart                             ║
║      - FAZ 2 korelasyon heatmap                                             ║
║      - Confusion matrix (ensemble + top 3 solo, klinik etiketli)           ║
║      - Solo vs Ensemble F2 bar (CI'lı)                                      ║
║      - ROC + Precision-Recall eğri karşılaştırması                          ║
║                                                                              ║
║  ── 4 FAZ PIPELINE (değişmedi) ────────────────────────────────────────────  ║
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
    average_precision_score, precision_recall_curve, roc_curve,
)

# v4.1: Görsel üretimi için matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")  # headless mode (Windows server / SSH)
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[UYARI] matplotlib yok → pip install matplotlib (görseller atlanır)")
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
print("  BioSense AI v4.1 — TEKNOFEST RAPORU İÇİN TAM PIPELINE")
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

# ══════════════════════════════════════════════════════════════════
#  v4.1: GÖRSEL ÜRETİM FONKSİYONLARI (TEKNOFEST raporu için)
# ══════════════════════════════════════════════════════════════════

GRAFIK_KLASOR = "grafikler"
if MATPLOTLIB_OK and not os.path.exists(GRAFIK_KLASOR):
    os.makedirs(GRAFIK_KLASOR)

def confusion_matrix_gorseli(y_true, y_pred, ad, dosya_adi):
    """
    Confusion matrix'i PNG olarak kaydet (klinik anlamla).
    TEKNOFEST raporunda her model için bu görsel zorunlu.
    """
    if not MATPLOTLIB_OK:
        return
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hücre değerleri ve klinik etiketler
    etiketler = [
        [f"{tn}\nGerçek Negatif\n(Doğru Benign)",
         f"{fp}\nYanlış Pozitif\n(Gereksiz takip)"],
        [f"{fn}\nYanlış Negatif\n(KAÇIRILAN PATOJENİK)",
         f"{tp}\nGerçek Pozitif\n(Doğru Patojenik)"],
    ]
    for i in range(2):
        for j in range(2):
            renk = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, etiketler[i][j], ha="center", va="center",
                    color=renk, fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tahmin: Benign (0)", "Tahmin: Patojenik (1)"])
    ax.set_yticklabels(["Gerçek: Benign (0)", "Gerçek: Patojenik (1)"])
    ax.set_title(f"{ad}\nKarmaşıklık Matrisi", fontsize=12, fontweight="bold")
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def faz1_karsilastirma_bar(faz1_data, dosya_adi="faz1_algoritma_karsilastirma.png"):
    """
    FAZ 1'in 10 algoritmasının F2/F1/MCC karşılaştırma bar chart'ı.
    """
    if not MATPLOTLIB_OK:
        return
    sirali = sorted(faz1_data, key=lambda x: x["mean_f2"], reverse=True)
    algos = [r["algo"] for r in sirali]
    f2s   = [r["mean_f2"] for r in sirali]
    f1s   = [r.get("mean_f1", 0) for r in sirali]
    stds  = [r["std_f2"] for r in sirali]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos))
    genislik = 0.35

    bar1 = ax.bar(x - genislik/2, f2s, genislik, label="F2 (binary)",
                   yerr=stds, capsize=4, color="#1f77b4", edgecolor="black")
    bar2 = ax.bar(x + genislik/2, f1s, genislik, label="F1 (binary)",
                   color="#ff7f0e", edgecolor="black")

    ax.set_xlabel("Algoritma", fontsize=11, fontweight="bold")
    ax.set_ylabel("Skor", fontsize=11, fontweight="bold")
    ax.set_title("FAZ 1 — 10 Algoritma Karşılaştırması (Binary F2 ve F1, ±std)",
                  fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=35, ha="right")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim([0, 1.05])

    # Eleme çizgisi
    ax.axhline(y=0.70, color="red", linestyle=":", linewidth=1.5,
                label="Eleme eşiği (F2=0.70)")
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def korelasyon_heatmap(korelasyon_df, dosya_adi="faz2_korelasyon_heatmap.png"):
    """FAZ 2 korelasyon matrisi heatmap'i."""
    if not MATPLOTLIB_OK:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(korelasyon_df.values, cmap="RdYlGn_r", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Korelasyon")

    cols = list(korelasyon_df.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    # Sayıları yaz
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = korelasyon_df.iloc[i, j]
            renk = "white" if val < 0.7 or val > 0.95 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=renk, fontsize=9)

    ax.set_title("FAZ 2 — Algoritma OOF Tahminleri Arası Korelasyon\n"
                  "(Düşük korelasyon = Yüksek Çeşitlilik)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def roc_pr_curve_karsilastirma(y_true, proba_dict, dosya_adi="roc_pr_curves.png"):
    """
    ROC + Precision-Recall Curve karşılaştırması (çoklu model).
    proba_dict: {model_adi: olasilik_dizisi}
    """
    if not MATPLOTLIB_OK:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    renkler = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (ad, proba) in enumerate(proba_dict.items()):
        c = renkler[i % len(renkler)]
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_v = roc_auc_score(y_true, proba)
            ax1.plot(fpr, tpr, color=c, lw=2,
                      label=f"{ad} (AUC={auc_v:.3f})")
        except Exception:
            pass
        # PR
        try:
            prec, rec, _ = precision_recall_curve(y_true, proba)
            ap = average_precision_score(y_true, proba)
            ax2.plot(rec, prec, color=c, lw=2,
                      label=f"{ad} (PR-AUC={ap:.3f})")
        except Exception:
            pass

    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Eğrisi Karşılaştırması", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(linestyle="--", alpha=0.6)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])

    baseline = float((y_true == 1).mean())
    ax2.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                 label=f"Baz (sınıf %{baseline*100:.0f})")
    ax2.set_xlabel("Recall (Duyarlılık)", fontsize=11)
    ax2.set_ylabel("Precision (Kesinlik)", fontsize=11)
    ax2.set_title("Precision-Recall Eğrisi Karşılaştırması",
                   fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(linestyle="--", alpha=0.6)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def solo_vs_ensemble_bar(karsilastirma_listesi,
                          dosya_adi="solo_vs_ensemble.png"):
    """Solo modeller + Ensemble F2 + CI bar chart'ı."""
    if not MATPLOTLIB_OK:
        return
    sirali = sorted(karsilastirma_listesi,
                     key=lambda x: x.get("test_f2", 0), reverse=True)
    isimler = [r["model"] for r in sirali]
    f2s     = [r["test_f2"] for r in sirali]
    lo      = [r.get("f2_ci_low", r["test_f2"]) for r in sirali]
    hi      = [r.get("f2_ci_high", r["test_f2"]) for r in sirali]
    err_lo  = [f2s[i] - lo[i] for i in range(len(f2s))]
    err_hi  = [hi[i] - f2s[i] for i in range(len(f2s))]

    fig, ax = plt.subplots(figsize=(11, 6))
    renkler = ["#2ca02c" if "ENSEMBLE" in n else "#1f77b4" for n in isimler]
    bars = ax.bar(range(len(isimler)), f2s,
                   yerr=[err_lo, err_hi], capsize=8,
                   color=renkler, edgecolor="black", linewidth=1.2)

    for i, (b, v) in enumerate(zip(bars, f2s)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(isimler)))
    ax.set_xticklabels(isimler, rotation=20, ha="right")
    ax.set_ylabel("Test F2 (Binary, pos=Patojenik)", fontsize=11, fontweight="bold")
    ax.set_title("Solo Model vs Ensemble — Test Seti F2 Skoru (95% Bootstrap CI)",
                  fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim([0.8, 1.0])
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def shap_bar_grafigi(top_features, top_values,
                      dosya_adi="shap_top_ozellikler.png"):
    """SHAP top özellikler bar grafiği (açıklanabilirlik için)."""
    if not MATPLOTLIB_OK:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_values, color="#1f77b4", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Ortalama |SHAP değeri|", fontsize=11, fontweight="bold")
    ax.set_title("SHAP — Top 15 En Önemli Özellik\n(Açıklanabilirlik Analizi)",
                  fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    for i, v in enumerate(top_values):
        ax.text(v + max(top_values)*0.01, i, f"{v:.4f}",
                 va="center", fontsize=9)
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

def karar_esigi_tablosu(y_true, y_proba, esikler=None):
    """
    Farklı karar eşiklerinde metrikleri hesapla.
    TEKNOFEST'in 'Farklı karar eşiklerinin test sonuçları' zorunluluğu.
    """
    if esikler is None:
        esikler = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    sonuclar = []
    for e in esikler:
        pred = (y_proba >= e).astype(int)
        cm = confusion_matrix(y_true, pred)
        if cm.shape == (2, 2):
            tn, fp, fn_, tp = cm.ravel()
        else:
            tn = fp = fn_ = tp = 0
        f2 = fbeta_score(y_true, pred, beta=2, pos_label=1,
                          average="binary", zero_division=0)
        f1 = f1_score(y_true, pred, pos_label=1, average="binary", zero_division=0)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, pred, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_true, pred) if cm.shape == (2, 2) else 0
        sonuclar.append({
            "esik":      round(e, 2),
            "F2":        round(f2, 4),
            "F1":        round(f1, 4),
            "Recall":    round(rec, 4),
            "Precision": round(prec, 4),
            "MCC":       round(mcc, 4),
            "TP":        int(tp), "FP": int(fp),
            "FN":        int(fn_), "TN": int(tn),
        })
    return sonuclar

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

        # v4.1: SHAP top 15 özellik grafiği (rapor için açıklanabilirlik)
        top15_idx = onem_sirasi[:15]
        top15_features = [all_cols[i] for i in top15_idx]
        top15_values   = [float(mean_abs_shap[i]) for i in top15_idx]
        shap_bar_grafigi(top15_features, top15_values)
        print(f"  → grafikler/shap_top_ozellikler.png kaydedildi.")
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

# v4.1: FAZ 1 algoritma karşılaştırma bar chart
faz1_karsilastirma_bar(eleme_raporu)
print(f"  → grafikler/faz1_algoritma_karsilastirma.png kaydedildi.")

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

# v4.1: Korelasyon heatmap
korelasyon_heatmap(korelasyon_df)
print(f"  → grafikler/faz2_korelasyon_heatmap.png kaydedildi.")

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

# v4.1: PR-AUC (kesinlik-duyarlılık eğrisi altında kalan alan) — ZORUNLU metrik
try:
    test_pr_auc = float(average_precision_score(y_test, final_test_proba))
    _, test_pr_auc_lo, test_pr_auc_hi = bootstrap_metric_ci(
        y_test, final_test_proba, average_precision_score,
    )
except Exception:
    test_pr_auc = 0.0
    test_pr_auc_lo = test_pr_auc_hi = 0.0

# OOF PR-AUC (overfitting karşılaştırması için)
try:
    oof_pr_auc = float(average_precision_score(y_train, kalibre_oof_proba))
except Exception:
    oof_pr_auc = 0.0

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
print(f"  PR-AUC (Avg. Precision)    : {test_pr_auc:.4f}  [95% CI: {test_pr_auc_lo:.4f}-{test_pr_auc_hi:.4f}]")
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
#  8.5. FAZ — SOLO MODEL KARŞILAŞTIRMASI (v4.0 yeni)
#       En iyi 5 modelin tek başına test performansını ölç, ensemble
#       ile karşılaştır. Bootstrap CI çakışma analizi yap.
#       Bu, "ensemble vs solo" jüri sorusuna istatistiksel cevap verir.
# ══════════════════════════════════════════════════════════════════
print(f"\n[8.5/9] SOLO MODEL KARŞILAŞTIRMASI — En iyi 5 model tek başına test...")
print("=" * 72)

def en_iyi_f2_esik_train(y_train, probs):
    """Train üzerinde optimal F2 eşiği."""
    en_iyi = 0.0
    esik = 0.5
    for e in np.arange(0.05, 0.95, 0.01):
        f2 = fbeta_score(y_train, (probs >= e).astype(int),
                          beta=2, pos_label=1, average="binary", zero_division=0)
        if f2 > en_iyi:
            en_iyi = f2
            esik = round(float(e), 4)
    return esik

def _solo_evaluate(model_adi, fit_fn, predict_fn):
    """Tek bir modeli eğit, test et, metrikleri ve CI'ları dön."""
    t0 = time.perf_counter()
    try:
        model = fit_fn()
        train_probs = predict_fn(model, X_train)
        esik = en_iyi_f2_esik_train(y_train, train_probs)
        test_probs = predict_fn(model, X_test)
        test_pred  = (test_probs >= esik).astype(int)

        f2 = fbeta_score(y_test, test_pred, beta=2, pos_label=1,
                          average="binary", zero_division=0)
        f1 = f1_score(y_test, test_pred, pos_label=1,
                       average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = 0.0
        acc = accuracy_score(y_test, test_pred)
        mcc = matthews_corrcoef(y_test, test_pred)

        _, f2_lo, f2_hi = bootstrap_metric_ci(
            y_test, test_probs,
            lambda yt, yp: fbeta_score(yt, yp, beta=2, pos_label=1,
                                        average="binary", zero_division=0),
            threshold=esik,
        )

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn_, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0))
        recall_p = recall_score(y_test, test_pred, pos_label=1, zero_division=0)
        precision_p = precision_score(y_test, test_pred, pos_label=1, zero_division=0)

        dt = time.perf_counter() - t0
        print(f"  ── {model_adi:<22} F2={f2:.4f} [{f2_lo:.4f}-{f2_hi:.4f}]  "
              f"FN={fn_}  ({dt:.1f}s)")

        del model; gc.collect()

        return {
            "model": model_adi,
            "test_f2": round(f2, 4),
            "f2_ci_low": round(f2_lo, 4),
            "f2_ci_high": round(f2_hi, 4),
            "test_f1": round(f1, 4),
            "test_auc": round(auc, 4),
            "test_mcc": round(mcc, 4),
            "test_acc_pct": round(acc * 100, 2),
            "patojenik_recall": round(recall_p, 4),
            "patojenik_precision": round(precision_p, 4),
            "false_negatives": int(fn_),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "optimal_esik": float(esik),
            "egitim_suresi_s": round(dt, 1),
        }
    except Exception as e:
        print(f"  ── {model_adi:<22} HATA: {type(e).__name__}: {e}")
        return {"model": model_adi, "test_f2": 0.0, "hata": str(e)}

# Solo model setleri (FAZ 1 + FAZ 3 sonuçlarından makul defaults)
solo_sonuclari = []

# 1. CatBoost
def _solo_fit_cb():
    m = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        min_data_in_leaf=10, subsample=0.8, colsample_bylevel=0.8,
        scale_pos_weight=SPW, bootstrap_type="Bernoulli",
        random_seed=RASTGELE_TOHUM, verbose=False, thread_count=-1,
    )
    m.fit(X_train, y_train, cat_features=kat_idx(X_train))
    return m
def _solo_pred_cb(m, X): return m.predict_proba(X)[:, 1]
if CB_OK:
    solo_sonuclari.append(_solo_evaluate("CatBoost", _solo_fit_cb, _solo_pred_cb))

# 2. ExtraTrees (FAZ 3 derin tuning kullanılır)
def _solo_fit_et():
    if "extra_trees" in faz3_top_trials_per_algo and faz3_top_trials_per_algo["extra_trees"]:
        params = trial_to_model_params("extra_trees",
                                        faz3_top_trials_per_algo["extra_trees"][0].params,
                                        n_jobs_override=-1)
    else:
        params = {"n_estimators": 500, "max_depth": 15, "min_samples_split": 5,
                  "min_samples_leaf": 2, "max_features": 0.5,
                  "class_weight": "balanced", "random_state": RASTGELE_TOHUM,
                  "n_jobs": -1}
    m = ExtraTreesClassifier(**params)
    m.fit(numerik_donustur(X_train), y_train)
    return m
def _solo_pred_et(m, X): return m.predict_proba(numerik_donustur(X))[:, 1]
solo_sonuclari.append(_solo_evaluate("ExtraTrees (FAZ3)", _solo_fit_et, _solo_pred_et))

# 3. LightGBM
def _solo_fit_lgb():
    m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, max_depth=-1,
        min_child_samples=20, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=SPW, random_state=RASTGELE_TOHUM,
        n_jobs=-1, verbosity=-1, subsample_freq=1,
    )
    m.fit(X_train, y_train, categorical_feature=kat_idx(X_train))
    return m
def _solo_pred_lgb(m, X): return m.predict_proba(X)[:, 1]
if LGBM_OK:
    solo_sonuclari.append(_solo_evaluate("LightGBM", _solo_fit_lgb, _solo_pred_lgb))

# 4. Random Forest
def _solo_fit_rf():
    m = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features=0.5, class_weight="balanced",
        random_state=RASTGELE_TOHUM, n_jobs=-1,
    )
    m.fit(numerik_donustur(X_train), y_train)
    return m
def _solo_pred_rf(m, X): return m.predict_proba(numerik_donustur(X))[:, 1]
solo_sonuclari.append(_solo_evaluate("Random Forest", _solo_fit_rf, _solo_pred_rf))

# 5. XGBoost
def _solo_fit_xgb():
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
        enable_categorical=True, tree_method="hist", device="cpu",
        verbosity=0, random_state=RASTGELE_TOHUM, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m
def _solo_pred_xgb(m, X): return m.predict_proba(X)[:, 1]
solo_sonuclari.append(_solo_evaluate("XGBoost", _solo_fit_xgb, _solo_pred_xgb))

# Ensemble sonucunu da listeye ekle (sıralama için)
ensemble_satiri = {
    "model": "ENSEMBLE (4-faz pipeline)",
    "test_f2": round(test_f2, 4),
    "f2_ci_low": round(test_f2_lo, 4),
    "f2_ci_high": round(test_f2_hi, 4),
    "test_f1": round(test_f1, 4),
    "test_auc": round(test_auc, 4),
    "test_mcc": round(test_mcc, 4),
    "test_acc_pct": round(test_acc * 100, 2),
    "patojenik_recall": round(test_recall_path, 4),
    "patojenik_precision": round(test_precision_path, 4),
    "false_negatives": fn_count,
    "false_positives": fp_count,
    "true_positives": tp_count,
    "true_negatives": tn_count,
    "optimal_esik": float(OPTIMAL_ESIK),
    "egitim_suresi_s": "N/A",
}
karsilastirma_listesi = solo_sonuclari + [ensemble_satiri]

# CSV kaydet
df_karsi = pd.DataFrame(karsilastirma_listesi)
df_karsi_sirali = df_karsi.sort_values("test_f2", ascending=False).reset_index(drop=True)
df_karsi.to_csv("solo_vs_ensemble.csv", index=False, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────
# v4.1: GÖRSELLER — TEKNOFEST raporu için zorunlu
# ─────────────────────────────────────────────────────────────────
print(f"\n  Görseller üretiliyor (grafikler/ klasörüne)...")

# 1. Ensemble confusion matrix
confusion_matrix_gorseli(y_test, test_tahminler,
                          "ENSEMBLE (4-faz pipeline)",
                          "cm_ensemble.png")

# 2. Solo modeller için CM (sadece eğitilmiş olanlar — yeniden eğitmek
# pahalı, en iyi 2 solo için yapalım)
en_iyi_3_solo = sorted(solo_sonuclari, key=lambda x: x.get("test_f2", 0),
                        reverse=True)[:3]
for i, s in enumerate(en_iyi_3_solo):
    ad = s["model"]
    print(f"  → {ad} CM hesaplanıyor (yeniden tahmin)...")
    # Modeli yeniden eğit ve tahmin yap (probs gerekiyor, depolamadık)
    try:
        if "CatBoost" in ad and CB_OK:
            m = _solo_fit_cb()
            probs = _solo_pred_cb(m, X_test)
        elif "ExtraTrees" in ad:
            m = _solo_fit_et()
            probs = _solo_pred_et(m, X_test)
        elif "LightGBM" in ad and LGBM_OK:
            m = _solo_fit_lgb()
            probs = _solo_pred_lgb(m, X_test)
        elif "Random Forest" in ad:
            m = _solo_fit_rf()
            probs = _solo_pred_rf(m, X_test)
        elif "XGBoost" in ad:
            m = _solo_fit_xgb()
            probs = _solo_pred_xgb(m, X_test)
        else:
            continue
        esik = s["optimal_esik"]
        pred = (probs >= esik).astype(int)
        # Confusion matrix
        guvenli_ad = ad.replace(" ", "_").replace("(", "").replace(")", "")
        confusion_matrix_gorseli(y_test, pred, ad,
                                  f"cm_solo_{i+1}_{guvenli_ad}.png")
        # ROC/PR için sakla
        if i == 0:  # en iyi solo
            en_iyi_solo_probs = probs
            en_iyi_solo_ad    = ad
        del m
        gc.collect()
    except Exception as e:
        print(f"    [UYARI] {ad} CM atlandı: {e}")

# 3. Solo vs Ensemble bar chart
solo_vs_ensemble_bar(karsilastirma_listesi)
print(f"  → grafikler/solo_vs_ensemble.png")

# 4. ROC + PR curve karşılaştırma (Ensemble vs en iyi solo)
try:
    proba_dict = {"Ensemble (4-faz)": final_test_proba}
    if 'en_iyi_solo_probs' in dir():
        proba_dict[en_iyi_solo_ad] = en_iyi_solo_probs
    roc_pr_curve_karsilastirma(y_test, proba_dict)
    print(f"  → grafikler/roc_pr_curves.png")
except Exception as e:
    print(f"  [UYARI] ROC/PR grafiği atlandı: {e}")

# 5. Karar eşiği tablosu (TEKNOFEST zorunlu)
print(f"\n  Farklı Karar Eşiklerinde Performans (Ensemble Test):")
esik_sonuc = karar_esigi_tablosu(y_test, final_test_proba)
print(f"\n    {'Eşik':>6} {'F2':>7} {'F1':>7} {'Recall':>7} "
      f"{'Precision':>10} {'MCC':>7} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
print(f"    {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*7} "
      f"{'-'*3} {'-'*3} {'-'*3} {'-'*3}")
for r in esik_sonuc:
    is_optimal = abs(r["esik"] - OPTIMAL_ESIK) < 0.025
    marker = " ★" if is_optimal else "  "
    print(f"   {marker}{r['esik']:>6.2f} {r['F2']:>7.4f} {r['F1']:>7.4f} "
          f"{r['Recall']:>7.4f} {r['Precision']:>10.4f} {r['MCC']:>7.4f} "
          f"{r['TP']:>3} {r['FP']:>3} {r['FN']:>3} {r['TN']:>3}")
# Karar eşiği tablosunu CSV olarak da kaydet
pd.DataFrame(esik_sonuc).to_csv("karar_esigi_tablosu.csv",
                                   index=False, encoding="utf-8")
print(f"\n  → karar_esigi_tablosu.csv kaydedildi (rapor için).")

# ══════════════════════════════════════════════════════════════════
#  TEKNOFEST JÜRİ SUNUM BLOKU — Hibrit Karşılaştırma
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 88)
print("  TEKNOFEST JÜRİ SUNUM BLOKU — Hibrit Karşılaştırma (v4.0)")
print("═" * 88)

print(f"\n  PERFORMANS KARŞILAŞTIRMASI (Test seti: {N_TEST} örnek)")
print(f"\n  {'Model':<26} {'F2':>7} {'95% CI':<18} {'F1':>7} "
      f"{'Recall':>7} {'FN':>3} {'FP':>3}")
print(f"  {'-'*26} {'-'*7} {'-'*18} {'-'*7} {'-'*7} {'-'*3} {'-'*3}")

for _, r in df_karsi_sirali.iterrows():
    ci_str = f"[{r['f2_ci_low']:.3f}-{r['f2_ci_high']:.3f}]"
    marker = " ★" if "ENSEMBLE" in r['model'] else "  "
    print(f"  {r['model']:<26}{marker}{r['test_f2']:>7.4f} {ci_str:<18} "
          f"{r['test_f1']:>7.4f} {r['patojenik_recall']:>7.4f} "
          f"{r['false_negatives']:>3} {r['false_positives']:>3}")

# CI çakışma analizi
ensemble_lo = ensemble_satiri["f2_ci_low"]
ensemble_hi = ensemble_satiri["f2_ci_high"]
en_iyi_solo = max(solo_sonuclari, key=lambda x: x.get("test_f2", 0))
solo_lo = en_iyi_solo["f2_ci_low"]
solo_hi = en_iyi_solo["f2_ci_high"]
ci_cakisma = (solo_lo <= ensemble_hi) and (ensemble_lo <= solo_hi)

print(f"\n  İSTATİSTİKSEL ANALİZ (Bootstrap %95 CI çakışma):")
print(f"  En iyi solo model      : {en_iyi_solo['model']:<22} "
      f"[{solo_lo:.3f}, {solo_hi:.3f}]")
print(f"  Ensemble (4-faz)       : [{ensemble_lo:.3f}, {ensemble_hi:.3f}]")

if ci_cakisma:
    print(f"  ★ Güven aralıkları ÇAKIŞIYOR → İstatistiksel olarak ANLAMLI FARK YOK")
else:
    print(f"  ⚠ Güven aralıkları çakışmıyor → İstatistiksel fark var")

print(f"\n  METODOLOJİ TERCİHİ: Ensemble (4-faz hiyerarşik)")
print(f"  Gerekçe:")
print(f"    1. Tek modele bağımlılık yok (robustness)")
print(f"    2. 3 farklı algoritmik paradigma birleşimi (genelleme garantisi)")
print(f"    3. Isotonic kalibrasyon (klinik açıdan anlamlı olasılıklar)")
if ci_cakisma:
    print(f"    4. İstatistiksel olarak en iyi solo modele EŞİT performans")
print("═" * 88)

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
        "surum"          : "BioSense AI v4.1 — TEKNOFEST Tam Pipeline (PR-AUC, görseller, eşik tablosu)",
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
        "test_pr_auc"    : round(test_pr_auc, 4),                                # v4.1 yeni
        "test_pr_auc_ci95": [round(test_pr_auc_lo, 4), round(test_pr_auc_hi, 4)],
        "test_mcc"       : round(test_mcc, 4),
        "test_acc_pct"   : round(test_acc * 100, 4),
        "test_recall"    : round(test_recall, 4),
        "test_precision" : round(test_precision, 4),
        "test_spec"      : round(test_spec, 4),
        "oof_f2"         : round(oof_f2, 4),
        "oof_f1"         : round(oof_f1, 4),
        "oof_auc"        : round(oof_auc, 4),
        "oof_pr_auc"     : round(oof_pr_auc, 4),                                 # v4.1 yeni
        "oof_mcc"        : round(oof_mcc, 4),
    },
    "karar_esigi_tablosu": esik_sonuc,    # v4.1: TEKNOFEST zorunlu
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
    "solo_vs_ensemble": {
        "_aciklama": "v4.0: TEKNOFEST jüri sunum için solo modellerin tek başına "
                     "test performansları. Bootstrap %95 CI çakışması istatistiksel "
                     "olarak ensemble'ın seçimini destekler.",
        "ci_cakisiyor"          : bool(ci_cakisma),
        "yorum"                 : ("İstatistiksel olarak anlamlı fark yok"
                                    if ci_cakisma else "İstatistiksel fark var"),
        "en_iyi_solo_model"     : en_iyi_solo["model"],
        "en_iyi_solo_f2"        : en_iyi_solo["test_f2"],
        "en_iyi_solo_ci95"      : [en_iyi_solo["f2_ci_low"],
                                    en_iyi_solo["f2_ci_high"]],
        "ensemble_f2"           : ensemble_satiri["test_f2"],
        "ensemble_ci95"         : [ensemble_satiri["f2_ci_low"],
                                    ensemble_satiri["f2_ci_high"]],
        "fark"                  : round(en_iyi_solo["test_f2"]
                                         - ensemble_satiri["test_f2"], 4),
        "tum_modeller"          : df_karsi_sirali.to_dict(orient="records"),
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
print(f"  'solo_vs_ensemble.csv' (v4.0 jüri karşılaştırması)")
print(f"  'karar_esigi_tablosu.csv' (v4.1 TEKNOFEST zorunlu)")
print(f"  'grafikler/' klasöründe 8 PNG görsel:")
print(f"     - shap_top_ozellikler.png")
print(f"     - faz1_algoritma_karsilastirma.png")
print(f"     - faz2_korelasyon_heatmap.png")
print(f"     - cm_ensemble.png + cm_solo_*.png (4 adet)")
print(f"     - solo_vs_ensemble.png")
print(f"     - roc_pr_curves.png")
print("\n" + "=" * 72)
print("  BioSense AI v4.1 — TAMAMLANDI")
print("=" * 72)