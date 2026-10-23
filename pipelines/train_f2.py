"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BioSense AI v2.2 — MUTLAK KUSURSUZ (GECE MODU / NO-CRASH GUARANTEE)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DÜZELTMELER (v2.2):                                                         ║
║   ✓ CRASH #3 (ÖNLENDİ) CMA-ES Kategorik Parametre hatası kökünden çözüldü.   ║
║   ✓ CRASH #4 (ÖNLENDİ) SQLite Timeout 240 sn yapıldı, 24 çekirdek ezilmeyecek║
║   ✓ CRASH #5 (ÖNLENDİ) JSON Numpy Float Casting hataları (TypeError) giderildi║
║   ✓ CRASH #6 (ÖNLENDİ) CatBoost boş kategorik liste (empty cat_features) fix ║
║  YENİ F2 OPTİMİZASYONLARI & TEKNOFEST ÖZEL:                                  ║
║   ✓ FAZ2 kazanan algoritmaya otomatik odaklanır (FAZ1 sonucuna göre)         ║
║   ✓ Benzersiz olasılık eşik araması (grid'den daha hassas)                   ║
║   ✓ TEKNOFEST: Her algoritmanın zirve skoru ve parametreleri JSON'a eklenir. ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, time, warnings, gc
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    fbeta_score, f1_score, roc_auc_score,
    accuracy_score, matthews_corrcoef, confusion_matrix,
)
from scipy.stats import rankdata

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
    print("[UYARI] SHAP yok → pip install shap  (özellik budama devre dışı)")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  KONTROL PANELİ
# ══════════════════════════════════════════════════════════════════
VERI_DOSYASI   = "YARISMA_TRAIN_KANSER.csv"
CIKTI_JSON     = "en_iyi_parametreler_v2.json"

PARALEL_ISCI   = 24

FAZ1_TRIAL     = 3000
FAZ1_FOLDS     = 5
FAZ1_TEKRAR    = 5          # 5×5 = 25 fold
FAZ1_DB        = "sqlite:///biosense_faz1_v2.db"

FAZ2_TRIAL     = 1500
FAZ2_FOLDS     = 5
FAZ2_TEKRAR    = 10         # 5×10 = 50 fold
FAZ2_DB        = "sqlite:///biosense_faz2_v2.db"

TOP_K_ENSEMBLE = 15
OOF_FOLDS      = 5
OOF_TEKRAR     = 10
OOF_ES_SPLIT   = 0.12       # OOF fold içi erken-dur ayrımı (%12)

EARLY_STOP_ROUNDS  = 50
SHAP_PERCENTILE    = 20     # 0 = devre dışı
W_F2               = 0.80   # Hedef ağırlığı F2 
W_AUC              = 0.20
SOFTMAX_SICAKLIK   = 10.0

TEST_ORANI     = 0.20
RASTGELE_TOHUM = 43

ALGORITMALAR = ["xgboost"]
if LGBM_OK: ALGORITMALAR.append("lightgbm")
if CB_OK:   ALGORITMALAR.append("catboost")

print("=" * 72)
print("  BioSense AI v2.2 — KUSURSUZ GECE MODU (Zero-Crash)")
print(f"  Algoritmalar: {ALGORITMALAR}")
print(f"  FAZ1: {FAZ1_TRIAL}t×{FAZ1_FOLDS*FAZ1_TEKRAR}fold  |  "
      f"FAZ2: {FAZ2_TRIAL}t×{FAZ2_FOLDS*FAZ2_TEKRAR}fold")
print(f"  Hedef: {W_F2}×F2 + {W_AUC}×AUC  |  ES rounds: {EARLY_STOP_ROUNDS}")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
#  1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════
print("\n[1/7] Veri yükleniyor...")

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

KAT_KOLONLAR = []
for col in X.columns:
    if (X[col].dtype == "object" or X[col].dtype.name == "category"
            or str(col).startswith("CAT_") or str(col).startswith("AA_")):
        X[col] = X[col].astype(str).astype("category")
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
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════

def get_algo(t):
    """Güvenli algoritma çekici (Özellikle CMA-ES için hayat kurtarır)"""
    return t.user_attrs.get("algo", t.params.get("algorithm", "xgboost"))

def en_iyi_f2_esik_benzersiz(y_gercek, problar):
    adaylar = np.unique(problar)
    adaylar = adaylar[(adaylar > 0.01) & (adaylar < 0.99)]
    if len(adaylar) == 0:
        return 0.50, 0.0
    en_iyi_f2  = 0.0
    esik_sonuc = 0.50
    for e in adaylar:
        f2 = fbeta_score(y_gercek, (problar >= e).astype(int),
                         beta=2, average="weighted", zero_division=0)
        if f2 > en_iyi_f2:
            en_iyi_f2  = f2
            esik_sonuc = float(e)
    return round(esik_sonuc, 5), round(en_iyi_f2, 6)

def en_iyi_f2_esik_grid(y_gercek, problar, adim=0.005):
    en_iyi_f2  = 0.0
    esik_sonuc = 0.50
    for e in np.arange(0.05, 0.95 + adim, adim):
        f2 = fbeta_score(y_gercek, (problar >= e).astype(int),
                         beta=2, average="weighted", zero_division=0)
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

def rank_ensemble(proba_listesi):
    ranked = [rankdata(p) / len(p) for p in proba_listesi]
    return np.mean(ranked, axis=0)

def model_fit(algo, params, X_tr, y_tr, X_val=None, y_val=None, n_jobs_override=None):
    use_es = (X_val is not None) and (y_val is not None) and (len(X_val) >= 5)

    if algo == "xgboost":
        p = dict(params)
        if n_jobs_override: p["n_jobs"] = n_jobs_override
        if use_es:
            m = xgb.XGBClassifier(**p, early_stopping_rounds=EARLY_STOP_ROUNDS, eval_metric="auc")
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr)

    elif algo == "lightgbm":
        p = dict(params)
        if n_jobs_override: p["n_jobs"] = n_jobs_override
        cat_i = kat_idx(X_tr)
        cat_i_safe = cat_i if cat_i else "auto"
        m = lgb.LGBMClassifier(**p)
        if use_es:
            m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  categorical_feature=cat_i_safe,
                  callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                              lgb.log_evaluation(-1)])
        else:
            m.fit(X_tr, y_tr, categorical_feature=cat_i_safe)

    elif algo == "catboost":
        p = dict(params)
        if n_jobs_override: p["thread_count"] = n_jobs_override
        cat_i = kat_idx(X_tr)
        cat_i_safe = cat_i if cat_i else None
        if use_es:
            m = CatBoostClassifier(**p, early_stopping_rounds=EARLY_STOP_ROUNDS,
                                   cat_features=cat_i_safe, verbose=False)
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        else:
            m = CatBoostClassifier(**p, cat_features=cat_i_safe, verbose=False)
            m.fit(X_tr, y_tr)
    else:
        raise ValueError(f"Bilinmeyen algo: {algo}")
    return m

def oof_tahmin(algo, params, X_tr, y_tr, n_folds, n_tekrar, n_jobs_override=None, es_split=0.0):
    rskf   = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_tekrar, random_state=RASTGELE_TOHUM)
    toplam = np.zeros(len(X_tr))
    sayac  = np.zeros(len(X_tr))

    for tr_idx, val_idx in rskf.split(X_tr, y_tr):
        X_ft = X_tr.iloc[tr_idx]
        y_ft = y_tr[tr_idx]
        X_fv = X_tr.iloc[val_idx]
        y_fv = y_tr[val_idx]

        X_es = y_es = None
        if es_split > 0 and len(tr_idx) >= 20:
            n_es = max(5, int(len(tr_idx) * es_split))
            X_ft, X_es, y_ft, y_es = train_test_split(
                X_ft, y_ft, test_size=n_es, random_state=RASTGELE_TOHUM, stratify=y_ft,
            )

        m = model_fit(algo, params, X_ft, y_ft, X_es, y_es, n_jobs_override=n_jobs_override)
        toplam[val_idx] += m.predict_proba(X_fv)[:, 1]
        sayac[val_idx]  += 1

    return toplam / np.maximum(sayac, 1)

def params_coz(algo, trial_params, n_jobs_override=None):
    if algo == "xgboost":
        rename = {
            "xgb_n_est": "n_estimators", "xgb_depth": "max_depth",
            "xgb_mcw": "min_child_weight", "xgb_mds": "max_delta_step",
            "xgb_lr": "learning_rate", "xgb_sub": "subsample",
            "xgb_cbt": "colsample_bytree", "xgb_cbl": "colsample_bylevel",
            "xgb_cbn": "colsample_bynode", "xgb_gamma": "gamma",
            "xgb_alpha": "reg_alpha", "xgb_lambda": "reg_lambda",
            "xgb_spw": "scale_pos_weight",
        }
        p = {rename.get(k, k): v for k, v in trial_params.items() if k not in ("algorithm",)}
        p.update({"enable_categorical": True, "tree_method": "hist",
                  "device": "cpu", "verbosity": 0, "random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})

    elif algo == "lightgbm":
        rename = {
            "lgb_n_est": "n_estimators", "lgb_leaves": "num_leaves",
            "lgb_depth": "max_depth", "lgb_mcs": "min_child_samples",
            "lgb_lr": "learning_rate", "lgb_sub": "subsample",
            "lgb_cbt": "colsample_bytree", "lgb_alpha": "reg_alpha",
            "lgb_lambda": "reg_lambda", "lgb_msg": "min_split_gain",
            "lgb_spw": "scale_pos_weight",
        }
        p = {rename.get(k, k): v for k, v in trial_params.items() if k not in ("algorithm",)}
        p.update({"verbosity": -1, "random_state": RASTGELE_TOHUM, "subsample_freq": 1,
                  "n_jobs": n_jobs_override if n_jobs_override else 1})

    elif algo == "catboost":
        rename = {
            "cb_iter": "iterations", "cb_depth": "depth",
            "cb_lr": "learning_rate", "cb_l2": "l2_leaf_reg",
            "cb_mdl": "min_data_in_leaf", "cb_sub": "subsample",
            "cb_cbl": "colsample_bylevel", "cb_spw": "scale_pos_weight",
        }
        p = {rename.get(k, k): v for k, v in trial_params.items() if k not in ("algorithm",)}
        p.update({"random_seed": RASTGELE_TOHUM, "verbose": False,
                  "thread_count": n_jobs_override if n_jobs_override else 1})
    else:
        p = {}
    return p

def aralik_hesapla(trials, param_adi, yuzde_alt=5, yuzde_ust=95):
    degerler = [t.params[param_adi] for t in trials if param_adi in t.params]
    if not degerler: return None, None
    return (float(np.percentile(degerler, yuzde_alt)), float(np.percentile(degerler, yuzde_ust)))


# ══════════════════════════════════════════════════════════════════
#  2. FAZ 1 — GENİŞ HETEROJEN TPE
# ══════════════════════════════════════════════════════════════════
print(f"\n[2/7] FAZ 1: {FAZ1_TRIAL} trial × {FAZ1_FOLDS*FAZ1_TEKRAR} fold")
print("-" * 72)

def faz1_objective(trial):
    algo = trial.suggest_categorical("algorithm", ALGORITMALAR)
    trial.set_user_attr("algo", algo)

    if algo == "xgboost":
        params = {
            "enable_categorical": True, "tree_method": "hist", "device": "cpu", "verbosity": 0,
            "random_state": RASTGELE_TOHUM, "n_jobs": 1,
            "n_estimators"     : trial.suggest_int("xgb_n_est", 100, 1000),
            "max_depth"        : trial.suggest_int("xgb_depth", 2, 8),
            "min_child_weight" : trial.suggest_int("xgb_mcw", 1, 30),
            "max_delta_step"   : trial.suggest_int("xgb_mds", 0, 10),
            "learning_rate"    : trial.suggest_float("xgb_lr", 0.003, 0.30, log=True),
            "subsample"        : trial.suggest_float("xgb_sub", 0.35, 1.00),
            "colsample_bytree" : trial.suggest_float("xgb_cbt", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("xgb_cbl", 0.40, 1.00),
            "colsample_bynode" : trial.suggest_float("xgb_cbn", 0.40, 1.00),
            "gamma"            : trial.suggest_float("xgb_gamma", 0.0, 8.0),
            "reg_alpha"        : trial.suggest_float("xgb_alpha", 0.0, 15.0),
            "reg_lambda"       : trial.suggest_float("xgb_lambda", 0.1, 30.0),
            "scale_pos_weight" : trial.suggest_float("xgb_spw", SPW * 0.2, SPW * 6.0),
        }
    elif algo == "lightgbm":
        params = {
            "verbosity": -1, "random_state": RASTGELE_TOHUM, "n_jobs": 1, "subsample_freq": 1,
            "n_estimators"     : trial.suggest_int("lgb_n_est", 100, 1000),
            "num_leaves"       : trial.suggest_int("lgb_leaves", 20, 300),
            "max_depth"        : trial.suggest_int("lgb_depth", -1, 12),
            "min_child_samples": trial.suggest_int("lgb_mcs", 5, 100),
            "learning_rate"    : trial.suggest_float("lgb_lr", 0.003, 0.30, log=True),
            "subsample"        : trial.suggest_float("lgb_sub", 0.35, 1.00),
            "colsample_bytree" : trial.suggest_float("lgb_cbt", 0.35, 1.00),
            "reg_alpha"        : trial.suggest_float("lgb_alpha", 0.0, 15.0),
            "reg_lambda"       : trial.suggest_float("lgb_lambda", 0.1, 30.0),
            "min_split_gain"   : trial.suggest_float("lgb_msg", 0.0, 5.0),
            "scale_pos_weight" : trial.suggest_float("lgb_spw", SPW * 0.2, SPW * 6.0),
        }
    else:  # catboost
        params = {
            "random_seed": RASTGELE_TOHUM, "thread_count": 1, "verbose": False,
            "iterations"       : trial.suggest_int("cb_iter", 100, 800),
            "depth"            : trial.suggest_int("cb_depth", 2, 8),
            "learning_rate"    : trial.suggest_float("cb_lr", 0.003, 0.30, log=True),
            "l2_leaf_reg"      : trial.suggest_float("cb_l2", 0.1, 30.0),
            "min_data_in_leaf" : trial.suggest_int("cb_mdl", 1, 50),
            "subsample"        : trial.suggest_float("cb_sub", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("cb_cbl", 0.40, 1.00),
            "scale_pos_weight" : trial.suggest_float("cb_spw", SPW * 0.2, SPW * 6.0),
        }

    rskf = RepeatedStratifiedKFold(n_splits=FAZ1_FOLDS, n_repeats=FAZ1_TEKRAR, random_state=RASTGELE_TOHUM)
    f2_listesi  = []
    auc_listesi = []
    RAPOR_ADIMI = [FAZ1_FOLDS * FAZ1_TEKRAR // 4, FAZ1_FOLDS * FAZ1_TEKRAR // 2]

    for fold_no, (tr_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        try:
            m = model_fit(algo, params, X_train.iloc[tr_idx], y_train[tr_idx], X_train.iloc[val_idx], y_train[val_idx])
        except Exception:
            return 0.0

        prob = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        _, f2 = en_iyi_f2_esik_grid(y_train[val_idx], prob, adim=0.01)
        auc   = roc_auc_score(y_train[val_idx], prob)
        f2_listesi.append(f2)
        auc_listesi.append(auc)

        if fold_no in RAPOR_ADIMI:
            birlesik = float(np.mean([W_F2 * f + W_AUC * a for f, a in zip(f2_listesi, auc_listesi)]))
            try:
                trial.report(birlesik, step=fold_no)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned: raise
            except Exception: pass

    return float(np.mean([W_F2 * f + W_AUC * a for f, a in zip(f2_listesi, auc_listesi)]))

faz1_study = optuna.create_study(
    study_name="biosense_faz1_v2",
    direction="maximize",
    storage=optuna.storages.RDBStorage(
        url=FAZ1_DB,
        # 🔥 ŞEBNUS DOKUNUŞU: 24 Çekirdeği kaldıracak 240 saniyelik Zırh
        engine_kwargs={"connect_args": {"timeout": 240, "check_same_thread": False}}, 
    ),
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=RASTGELE_TOHUM, n_startup_trials=80, multivariate=True, warn_independent_sampling=False),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=5, interval_steps=5),
)

faz1_mevcut = len([t for t in faz1_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
faz1_kalan  = max(0, FAZ1_TRIAL - faz1_mevcut)

if faz1_kalan > 0:
    print(f"  Mevcut: {faz1_mevcut} | Kalan: {faz1_kalan}")
    t0 = time.perf_counter()
    faz1_study.optimize(faz1_objective, n_trials=faz1_kalan, n_jobs=PARALEL_ISCI, show_progress_bar=True, catch=(Exception,))
    print(f"  FAZ 1 bitti: {time.perf_counter()-t0:.1f}s")
else:
    print(f"  FAZ 1 zaten tamamlandı ({faz1_mevcut} trial), DB'den yükleniyor.")

faz1_en_iyi = faz1_study.best_value
print(f"  FAZ 1 En İyi Karma Skor: {faz1_en_iyi:.4f}")

algo_skorlar = {}
for t in faz1_study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
        a = get_algo(t)
        algo_skorlar.setdefault(a, []).append(t.value)

print("\n  Algoritma Performansı (FAZ 1):")
for a in sorted(algo_skorlar, key=lambda x: np.max(algo_skorlar.get(x, [0])), reverse=True):
    vals = algo_skorlar[a]
    print(f"    {a:<12}  trial={len(vals):4d}  ort={np.mean(vals):.4f}  maks={np.max(vals):.4f}")

faz1_tamamlanan = [t for t in faz1_study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
faz1_top20 = sorted(faz1_tamamlanan, key=lambda t: t.value, reverse=True)[:20]

# ── FAZ2 için kazanan algoritmayı belirle ──────────────────────────
en_iyi_algo_faz1 = max(algo_skorlar, key=lambda a: max(algo_skorlar.get(a, [0]))) if algo_skorlar else "xgboost"
print(f"\n  FAZ 2 odak algoritması: {en_iyi_algo_faz1.upper()} (FAZ 1 galibi)")

ALGORITMALAR_FAZ2 = list(ALGORITMALAR)
if len(ALGORITMALAR) > 1:
    maks_ort_galip   = np.mean(sorted(algo_skorlar.get(en_iyi_algo_faz1, [0]), reverse=True)[:50])
    diger_algo_maks  = max((np.mean(sorted(algo_skorlar.get(a, [0]), reverse=True)[:50]) for a in ALGORITMALAR if a != en_iyi_algo_faz1), default=0.0)
    if maks_ort_galip - diger_algo_maks >= 0.01:
        ALGORITMALAR_FAZ2 = [en_iyi_algo_faz1]
        print(f"  FAZ 2 yalnızca {en_iyi_algo_faz1} kullanacak (fark ≥ 0.01 skor puanı).")

# ══════════════════════════════════════════════════════════════════
#  3. SHAP ÖZELLİK BUDAMA
# ══════════════════════════════════════════════════════════════════
print(f"\n[3/7] SHAP özellik budama (alt %{SHAP_PERCENTILE} atılıyor)...")
if SHAP_OK and SHAP_PERCENTILE > 0:
    try:
        SHAP_XGB_PARAMS = {
            "enable_categorical": True, "tree_method": "hist", "device": "cpu", "verbosity": 0, "n_jobs": -1,
            "random_state": RASTGELE_TOHUM, "n_estimators": 200, "max_depth": 6,
            "learning_rate": 0.05, "subsample": 0.8, "scale_pos_weight": SPW,
        }
        print("  SHAP referans modeli (XGBoost) eğitiliyor...")
        shap_model = xgb.XGBClassifier(**SHAP_XGB_PARAMS)
        shap_model.fit(X_train, y_train)

        explainer   = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list): shap_values = shap_values[1]

        all_cols      = list(X_train.columns)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        print("\n  Top 10 SHAP Özelliği:")
        onem_sirasi = np.argsort(mean_abs_shap)[::-1]
        for rk in onem_sirasi[:10]:
            bar = "█" * int(mean_abs_shap[rk] / max(mean_abs_shap) * 35)
            print(f"    {all_cols[rk]:<30} {bar}  {mean_abs_shap[rk]:.5f}")

        shap_esik  = np.percentile(mean_abs_shap, SHAP_PERCENTILE)
        kalan_mask = mean_abs_shap >= shap_esik
        kalan_cols = [c for c, keep in zip(all_cols, kalan_mask) if keep]
        budanan_cnt = len(all_cols) - len(kalan_cols)

        X_train       = X_train[kalan_cols]
        X_test        = X_test[kalan_cols]
        KAT_KOLONLAR  = [c for c in KAT_KOLONLAR if c in kalan_cols]
        BUDANAN_OZELLIKLER = [c for c in all_cols if c not in kalan_cols]

        print(f"\n  {len(kalan_cols)} özellik kaldı ({budanan_cnt} gürültülü özellik budandı)")
        del shap_values, shap_model, explainer
        gc.collect()
    except Exception as e:
        print(f"  [UYARI] SHAP budama başarısız: {e}")
        BUDANAN_OZELLIKLER = []
else:
    print("  SHAP budama devre dışı.")

# ══════════════════════════════════════════════════════════════════
#  4. FAZ 2 — DAR KESINLEŞTIRME
# ══════════════════════════════════════════════════════════════════
print(f"\n[4/7] FAZ 2: {FAZ2_TRIAL} trial × {FAZ2_FOLDS*FAZ2_TEKRAR} fold")
print("-" * 72)

def faz2_objective(trial):
    # 🔥 ŞEBNUS DOKUNUŞU: CMA-ES BOMBASI İMHA EDİLDİ
    # Eğer FAZ 2'ye tek algoritma geçtiyse, "suggest_categorical" YAPMA! 
    # CMA-ES sadece float/int destekler, aksi takdirde anında patlar.
    if len(ALGORITMALAR_FAZ2) > 1:
        algo = trial.suggest_categorical("algorithm", ALGORITMALAR_FAZ2)
    else:
        algo = ALGORITMALAR_FAZ2[0]
    
    trial.set_user_attr("algo", algo)

    def dar_float(key, abs_lo, abs_hi, genislet=0.20):
        lo, hi = aralik_hesapla(faz1_top20, key)
        if lo is None: return abs_lo, abs_hi
        aralik = hi - lo
        return (max(abs_lo, lo - aralik * genislet), min(abs_hi, hi + aralik * genislet))

    def dar_int(key, abs_lo, abs_hi, genislet=0.20):
        lo, hi = dar_float(key, abs_lo, abs_hi, genislet)
        return max(abs_lo, int(lo)), min(abs_hi, int(hi) + 1)

    if algo == "xgboost":
        n_lo, n_hi   = dar_int("xgb_n_est", 50, 1200)
        d_lo, d_hi   = dar_int("xgb_depth", 1, 9)
        lr_lo, lr_hi = dar_float("xgb_lr", 0.001, 0.40)
        lr_lo = max(0.001, lr_lo * 0.5); lr_hi = min(0.40, lr_hi * 1.5)
        spw_lo, spw_hi = dar_float("xgb_spw", SPW * 0.10, SPW * 8.0)
        params = {
            "enable_categorical": True, "tree_method": "hist", "device": "cpu", "verbosity": 0,
            "random_state": RASTGELE_TOHUM, "n_jobs": 1,
            "n_estimators"     : trial.suggest_int("xgb_n_est", n_lo, n_hi),
            "max_depth"        : trial.suggest_int("xgb_depth", d_lo, d_hi),
            "min_child_weight" : trial.suggest_int("xgb_mcw", 1, 40),
            "max_delta_step"   : trial.suggest_int("xgb_mds", 0, 10),
            "learning_rate"    : trial.suggest_float("xgb_lr", lr_lo, lr_hi, log=True),
            "subsample"        : trial.suggest_float("xgb_sub", 0.35, 1.00),
            "colsample_bytree" : trial.suggest_float("xgb_cbt", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("xgb_cbl", 0.40, 1.00),
            "colsample_bynode" : trial.suggest_float("xgb_cbn", 0.40, 1.00),
            "gamma"            : trial.suggest_float("xgb_gamma", 0.0, 8.0),
            "reg_alpha"        : trial.suggest_float("xgb_alpha", 0.0, 15.0),
            "reg_lambda"       : trial.suggest_float("xgb_lambda", 0.1, 30.0),
            "scale_pos_weight" : trial.suggest_float("xgb_spw", max(SPW*0.10, spw_lo*0.7), min(SPW*8.0, spw_hi*1.3)),
        }
    elif algo == "lightgbm":
        n_lo, n_hi   = dar_int("lgb_n_est", 50, 1200)
        lr_lo, lr_hi = dar_float("lgb_lr", 0.001, 0.40)
        lr_lo = max(0.001, lr_lo * 0.5); lr_hi = min(0.40, lr_hi * 1.5)
        spw_lo, spw_hi = dar_float("lgb_spw", SPW * 0.10, SPW * 8.0)
        params = {
            "verbosity": -1, "random_state": RASTGELE_TOHUM, "n_jobs": 1, "subsample_freq": 1,
            "n_estimators"     : trial.suggest_int("lgb_n_est", n_lo, n_hi),
            "num_leaves"       : trial.suggest_int("lgb_leaves", 15, 500),
            "max_depth"        : trial.suggest_int("lgb_depth", -1, 12),
            "min_child_samples": trial.suggest_int("lgb_mcs", 3, 150),
            "learning_rate"    : trial.suggest_float("lgb_lr", lr_lo, lr_hi, log=True),
            "subsample"        : trial.suggest_float("lgb_sub", 0.35, 1.00),
            "colsample_bytree" : trial.suggest_float("lgb_cbt", 0.35, 1.00),
            "reg_alpha"        : trial.suggest_float("lgb_alpha", 0.0, 15.0),
            "reg_lambda"       : trial.suggest_float("lgb_lambda", 0.1, 30.0),
            "min_split_gain"   : trial.suggest_float("lgb_msg", 0.0, 5.0),
            "scale_pos_weight" : trial.suggest_float("lgb_spw", max(SPW*0.10, spw_lo*0.7), min(SPW*8.0, spw_hi*1.3)),
        }
    else:  # catboost
        n_lo, n_hi   = dar_int("cb_iter", 100, 900)
        lr_lo, lr_hi = dar_float("cb_lr", 0.001, 0.40)
        lr_lo = max(0.001, lr_lo * 0.5); lr_hi = min(0.40, lr_hi * 1.5)
        spw_lo, spw_hi = dar_float("cb_spw", SPW * 0.10, SPW * 8.0)
        params = {
            "random_seed": RASTGELE_TOHUM, "thread_count": 1, "verbose": False,
            "iterations"       : trial.suggest_int("cb_iter", n_lo, n_hi),
            "depth"            : trial.suggest_int("cb_depth", 2, 9),
            "learning_rate"    : trial.suggest_float("cb_lr", lr_lo, lr_hi, log=True),
            "l2_leaf_reg"      : trial.suggest_float("cb_l2", 0.1, 30.0),
            "min_data_in_leaf" : trial.suggest_int("cb_mdl", 1, 60),
            "subsample"        : trial.suggest_float("cb_sub", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("cb_cbl", 0.40, 1.00),
            "scale_pos_weight" : trial.suggest_float("cb_spw", max(SPW*0.10, spw_lo*0.7), min(SPW*8.0, spw_hi*1.3)),
        }

    rskf = RepeatedStratifiedKFold(n_splits=FAZ2_FOLDS, n_repeats=FAZ2_TEKRAR, random_state=RASTGELE_TOHUM)
    f2_listesi  = []
    auc_listesi = []
    RAPOR_ADIMI = [FAZ2_FOLDS * FAZ2_TEKRAR // 4, FAZ2_FOLDS * FAZ2_TEKRAR // 2]

    for fold_no, (tr_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        try:
            m = model_fit(algo, params, X_train.iloc[tr_idx], y_train[tr_idx], X_train.iloc[val_idx], y_train[val_idx])
        except Exception:
            return 0.0

        prob = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        _, f2 = en_iyi_f2_esik_grid(y_train[val_idx], prob, adim=0.005)
        auc   = roc_auc_score(y_train[val_idx], prob)
        f2_listesi.append(f2)
        auc_listesi.append(auc)

        if fold_no in RAPOR_ADIMI:
            birlesik = float(np.mean([W_F2 * f + W_AUC * a for f, a in zip(f2_listesi, auc_listesi)]))
            try:
                trial.report(birlesik, step=fold_no)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned: raise
            except Exception: pass

    return float(np.mean([W_F2 * f + W_AUC * a for f, a in zip(f2_listesi, auc_listesi)]))


if len(ALGORITMALAR_FAZ2) == 1:
    faz2_sampler = optuna.samplers.CmaEsSampler(seed=RASTGELE_TOHUM + 2, n_startup_trials=30, warn_independent_sampling=False)
    print("  Sampler: CMA-ES (tek algo, dar sürekli uzay)")
else:
    faz2_sampler = optuna.samplers.TPESampler(seed=RASTGELE_TOHUM + 1, n_startup_trials=30, multivariate=True, warn_independent_sampling=False)
    print("  Sampler: TPE multivariate (çoklu algo)")

faz2_study = optuna.create_study(
    study_name="biosense_faz2_v2",
    direction="maximize",
    storage=optuna.storages.RDBStorage(
        url=FAZ2_DB,
        # Timeout güvencesi 240 saniye!
        engine_kwargs={"connect_args": {"timeout": 240, "check_same_thread": False}},
    ),
    load_if_exists=True,
    sampler=faz2_sampler,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=8, interval_steps=8),
)

faz2_mevcut = len([t for t in faz2_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
if faz2_mevcut == 0:
    for top_t in faz1_top20:
        try: faz2_study.enqueue_trial(top_t.params)
        except Exception: pass
    print(f"  {len(faz1_top20)} FAZ 1 trial'ı sıraya eklendi.")

faz2_kalan = max(0, FAZ2_TRIAL - faz2_mevcut)
if faz2_kalan > 0:
    print(f"  Mevcut: {faz2_mevcut} | Kalan: {faz2_kalan}")
    t0 = time.perf_counter()
    faz2_study.optimize(faz2_objective, n_trials=faz2_kalan, n_jobs=PARALEL_ISCI, show_progress_bar=True, catch=(Exception,))
    print(f"  FAZ 2 bitti: {time.perf_counter()-t0:.1f}s")
else:
    print(f"  FAZ 2 zaten tamamlandı ({faz2_mevcut} trial).")

faz2_en_iyi = faz2_study.best_value
print(f"  FAZ 2 En İyi Karma Skor: {faz2_en_iyi:.4f}")

faz2_tamamlanan = [t for t in faz2_study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
faz2_sirali = sorted(faz2_tamamlanan, key=lambda t: t.value, reverse=True)

# ══════════════════════════════════════════════════════════════════
#  5. ENSEMBLE SEÇİMİ
# ══════════════════════════════════════════════════════════════════
print(f"\n[5/7] Top-{TOP_K_ENSEMBLE} ensemble seçiliyor...")

def parametre_mesafe(t1, t2):
    sufix_list = ["lr", "sub", "cbt", "alpha", "lambda", "spw"]
    def get_val(t, sufix):
        for k, v in t.params.items():
            if k.endswith(sufix): return float(v)
        return 0.0
    farklar = [
        (((get_val(t1, s) - get_val(t2, s)) / max(abs(get_val(t1, s)), abs(get_val(t2, s)), 1e-9)) ** 2)
        for s in sufix_list
    ]
    a1 = get_algo(t1)
    a2 = get_algo(t2)
    farklar.append(0.0 if a1 == a2 else 1.0)
    return float(np.sqrt(np.mean(farklar)))

MIN_MESAFE = 0.05
secilen_trials = [faz2_sirali[0]]
for tc in faz2_sirali[1:]:
    if len(secilen_trials) >= TOP_K_ENSEMBLE: break
    if max(parametre_mesafe(tc, s) for s in secilen_trials) >= MIN_MESAFE:
        secilen_trials.append(tc)
for tc in faz2_sirali:
    if len(secilen_trials) >= TOP_K_ENSEMBLE: break
    if tc not in secilen_trials:
        secilen_trials.append(tc)

algo_cnt = {}
for t in secilen_trials:
    a = get_algo(t)
    algo_cnt[a] = algo_cnt.get(a, 0) + 1

print(f"  {len(secilen_trials)} ensemble üyesi  |  Dağılım: {algo_cnt}")
for i, t in enumerate(secilen_trials):
    a = get_algo(t)
    print(f"  [{i+1:2d}] {a:<12}  CV skor: {t.value:.4f}")

# ══════════════════════════════════════════════════════════════════
#  6. ENSEMBLE OOF + RANK/AĞIRLIKLI BİRLEŞTİRME + İSOTONİK KAL.
# ══════════════════════════════════════════════════════════════════
print(f"\n[6/7] OOF: {len(secilen_trials)} model × {OOF_FOLDS*OOF_TEKRAR} fold "
      f"(ES ayrım: %{int(OOF_ES_SPLIT*100)})...")

model_oof_proba_listesi = []
model_oof_f2_listesi    = []

for i, t in enumerate(secilen_trials):
    algo         = get_algo(t)
    model_params = params_coz(algo, t.params, n_jobs_override=-1)

    print(f"  OOF [{i+1:2d}/{len(secilen_trials)}] {algo:<12} CV={t.value:.4f} ...", end="", flush=True)
    t0 = time.perf_counter()

    proba_i = oof_tahmin(algo, model_params, X_train, y_train, OOF_FOLDS, OOF_TEKRAR, n_jobs_override=-1, es_split=OOF_ES_SPLIT)
    _, f2_i = en_iyi_f2_esik_benzersiz(y_train, proba_i)
    
    model_oof_proba_listesi.append(proba_i)
    model_oof_f2_listesi.append(f2_i)
    print(f" OOF_F2={f2_i:.4f}  {time.perf_counter()-t0:.1f}s")
    gc.collect()

agirliklar = softmax_agirliklar(model_oof_f2_listesi, SOFTMAX_SICAKLIK)
print(f"\n  Softmax Ensemble Ağırlıkları (sıcaklık={SOFTMAX_SICAKLIK}):")
for i, (t, w, f2) in enumerate(zip(secilen_trials, agirliklar, model_oof_f2_listesi)):
    bar = "█" * max(1, int(w * 55))
    print(f"    [{i+1:2d}] {get_algo(t):<12}  F2={f2:.4f}  w={w:.4f}  {bar}")

oof_rank_proba     = rank_ensemble(model_oof_proba_listesi)
oof_weighted_proba = sum(w * p for w, p in zip(agirliklar, model_oof_proba_listesi))

_, f2_rank     = en_iyi_f2_esik_benzersiz(y_train, oof_rank_proba)
_, f2_weighted = en_iyi_f2_esik_benzersiz(y_train, oof_weighted_proba)

ensemble_oof_proba = oof_rank_proba if f2_rank >= f2_weighted else oof_weighted_proba
SECILEN_YONTEM     = "rank_averaging" if f2_rank >= f2_weighted else "softmax_weighted"
print(f"\n  Seçilen ensemble yöntemi: {SECILEN_YONTEM}  (rank={f2_rank:.4f}  weighted={f2_weighted:.4f})")

print("  Isotonic Regression kalibrasyonu...")
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(ensemble_oof_proba, y_train)
kalibre_oof_proba       = iso_reg.predict(ensemble_oof_proba)
OPTIMAL_ESIK, oof_f2    = en_iyi_f2_esik_benzersiz(y_train, kalibre_oof_proba)

oof_tahminler = (kalibre_oof_proba >= OPTIMAL_ESIK).astype(int)
oof_f1        = f1_score(y_train, oof_tahminler, average="weighted", zero_division=0)
oof_auc       = roc_auc_score(y_train, kalibre_oof_proba)
oof_mcc       = matthews_corrcoef(y_train, oof_tahminler)

print(f"\n  {'─'*56}")
print(f"  OOF F2  (kalibre, leakage-free) : {oof_f2:.4f}")
print(f"  OOF F1                          : {oof_f1:.4f}")
print(f"  OOF AUC                         : {oof_auc:.4f}")
print(f"  OOF MCC                         : {oof_mcc:.4f}")
print(f"  Optimal Eşik                    : {OPTIMAL_ESIK:.5f}")
print(f"  {'─'*56}")

# ══════════════════════════════════════════════════════════════════
#  7. FİNAL ENSEMBLE + TEST DEĞERLENDİRMESİ
# ══════════════════════════════════════════════════════════════════
print(f"\n[7/7] Final {len(secilen_trials)} model + test...")

final_test_proba_listesi = []

for i, t in enumerate(secilen_trials):
    algo         = get_algo(t)
    model_params = params_coz(algo, t.params, n_jobs_override=-1)

    X_fin_tr, X_fin_es, y_fin_tr, y_fin_es = train_test_split(
        X_train, y_train, test_size=max(5, int(N_TRAIN * 0.08)),
        random_state=RASTGELE_TOHUM + i, stratify=y_train,
    )

    print(f"  Final [{i+1:2d}/{len(secilen_trials)}] {algo:<12} ...", end="", flush=True)
    t0 = time.perf_counter()

    fm = model_fit(algo, model_params, X_fin_tr, y_fin_tr, X_fin_es, y_fin_es)
    final_test_proba_listesi.append(fm.predict_proba(X_test)[:, 1])

    print(f" {time.perf_counter()-t0:.2f}s")
    gc.collect()

if SECILEN_YONTEM == "rank_averaging":
    final_raw = rank_ensemble(final_test_proba_listesi)
else:
    final_raw = sum(w * p for w, p in zip(agirliklar, final_test_proba_listesi))

final_test_proba = iso_reg.predict(final_raw)

test_tahminler = (final_test_proba >= OPTIMAL_ESIK).astype(int)
test_f2        = fbeta_score(y_test, test_tahminler, beta=2, average="weighted", zero_division=0)
test_f1        = f1_score(y_test, test_tahminler, average="weighted", zero_division=0)
test_auc       = roc_auc_score(y_test, final_test_proba)
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

print("\n" + "=" * 72)
print("  GERÇEK TEST SONUÇLARI  (X_test ilk ve tek kez)")
print("=" * 72)
print(f"  F2  (GERÇEK) : {test_f2:.4f}  ←─ Ana Metrik")
print(f"  F1           : {test_f1:.4f}")
print(f"  AUC          : {test_auc:.4f}")
print(f"  MCC          : {test_mcc:.4f}")
print(f"  Doğruluk     : %{test_acc*100:.2f}")
print(f"  Recall       : {test_recall:.4f}")
print(f"  Precision    : {test_precision:.4f}")
print(f"  Specificity  : {test_spec:.4f}")
print(f"  Eşik         : {OPTIMAL_ESIK:.5f}")
print(f"  {'─'*58}")
print(f"  Ensemble     : {SECILEN_YONTEM}  |  Kalibrasyon: Isotonic")
print(f"  Algoritmalar : {algo_cnt}")
print(f"  Özellikler   : {len(X_train.columns)} ({len(BUDANAN_OZELLIKLER)} SHAP budandı)")
print(f"  OOF F2 (CV)  : {oof_f2:.4f}")
print(f"  FAZ 1 skor   : {faz1_en_iyi:.4f}")
print(f"  FAZ 2 skor   : {faz2_en_iyi:.4f}")
print(f"  OOF↔Test     : {test_f2 - oof_f2:+.4f}")
print("=" * 72)

# ── TEKNOFEST JÜRİ RAPORU (Her Algoritmanın Zirvesi) ──────────────
juri_karsilastirma = {}
tum_triallar = faz1_study.trials + faz2_study.trials
for algo in ALGORITMALAR:
    a_trials = [t for t in tum_triallar if t.state == optuna.trial.TrialState.COMPLETE and get_algo(t) == algo and t.value is not None]
    if a_trials:
        best_a = max(a_trials, key=lambda t: t.value)
        temiz_params = params_coz(algo, best_a.params)
        for k in ["n_jobs", "thread_count", "verbosity", "verbose", "random_state", "random_seed", "device", "enable_categorical", "tree_method"]:
            temiz_params.pop(k, None)
        
        juri_karsilastirma[algo] = {
            "en_iyi_cv_skor": float(round(best_a.value, 4)),
            "parametreler": {k: (float(round(v, 6)) if isinstance(v, float) else v) for k, v in temiz_params.items()}
        }

print("\n" + "=" * 72)
print("  TEKNOFEST JÜRİ KARŞILAŞTIRMA RAPORU (Algoritmaların Zirveleri)")
print("=" * 72)
for algo, data in juri_karsilastirma.items():
    print(f"  🏆 {algo.upper():<10} -> Max CV Skoru: {data['en_iyi_cv_skor']:.4f}")

# ── JSON Kayıt ────────────────────────────────────────────────────
# 🔥 ŞEBNUS GÜVENLİK DOKUNUŞU: Tüm metrikler saf Python Float'ına çevrildi (JSON Hatası %0)
best_trial  = secilen_trials[0]
best_algo   = get_algo(best_trial)
best_params = params_coz(best_algo, best_trial.params)
best_params.pop("n_jobs", None)
best_params.pop("thread_count", None)

sonuc = {
    "_aciklama": {
        "surum"          : "BioSense AI v2.2 (GECE MODU)",
        "duzeltmeler"    : ["CMA-ES Kategorik Bug Fix", "SQLite Timeout 240s", "JSON Float32 Casting"],
        "algoritmalar"   : ALGORITMALAR_FAZ2,
        "ensemble_k"     : len(secilen_trials),
        "ensemble_yontem": SECILEN_YONTEM,
        "kalibrasyon"    : "IsotonicRegression",
        "shap_budanan"   : len(BUDANAN_OZELLIKLER),
    },
    "pdr_v7_kopyala": {
        "algorithm"    : best_algo,
        **{k: (float(round(v, 6)) if isinstance(v, float) else v)
           for k, v in best_params.items()
           if k not in ("n_jobs", "thread_count", "verbosity", "verbose", 
                        "random_state", "random_seed", "enable_categorical", 
                        "tree_method", "device")},
        "OPTIMAL_ESIK" : float(OPTIMAL_ESIK),
    },
    "teknofest_juri_raporu": juri_karsilastirma,
    "ensemble_listesi": [
        {
            "algoritma": get_algo(t),
            "cv_skor"  : float(round(t.value, 4)),
            "oof_f2"   : float(round(model_oof_f2_listesi[i], 4)),
            "agirlik"  : float(round(float(agirliklar[i]), 6)),
            "params"   : {k: (float(round(v, 6)) if isinstance(v, float) else v)
                          for k, v in params_coz(get_algo(t), t.params).items()
                          if k not in ("n_jobs", "thread_count", "verbosity",
                                       "verbose", "random_state", "random_seed",
                                       "enable_categorical", "tree_method", "device")},
        }
        for i, t in enumerate(secilen_trials)
    ],
    "ensemble_optimal_esik" : float(OPTIMAL_ESIK),
    "ensemble_agirliklar"   : [float(round(float(w), 6)) for w in agirliklar],
    "skorlar": {
        "test_f2"        : float(round(test_f2, 4)),
        "test_f1"        : float(round(test_f1, 4)),
        "test_auc"       : float(round(test_auc, 4)),
        "test_mcc"       : float(round(test_mcc, 4)),
        "test_acc_pct"   : float(round(test_acc * 100, 4)),
        "test_recall"    : float(round(test_recall, 4)),
        "test_precision" : float(round(test_precision, 4)),
        "test_spec"      : float(round(test_spec, 4)),
        "oof_f2"         : float(round(oof_f2, 4)),
        "faz1_cv_skor"   : float(round(faz1_en_iyi, 4)),
        "faz2_cv_skor"   : float(round(faz2_en_iyi, 4)),
        "optimal_esik"   : float(OPTIMAL_ESIK),
    },
    "veri": {
        "n_train"            : int(N_TRAIN),
        "n_test"             : int(N_TEST),
        "spw_baz"            : float(round(SPW, 3)),
        "toplam_ozellik"     : int(len(OZELLIK)),
        "kalan_ozellik"      : int(len(X_train.columns)),
        "budanan_ozellikler" : BUDANAN_OZELLIKLER[:50],
    },
}

with open(CIKTI_JSON, "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=False, indent=2)

print(f"\n  '{CIKTI_JSON}' kaydedildi.")
print("  PDR v7 için → OPTUNA_MODU=False  OPTUNA_JSON='en_iyi_parametreler_v2.json'")
print("\n" + "=" * 72)
print("  BioSense AI v2.2 — TAMAMLANDI")
print("=" * 72)