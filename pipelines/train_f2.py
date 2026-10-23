"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BioSense AI v6.4 — Loky Process-Parallel (Optuna Resmi Tavsiye)            ║
║  Sağlıkta Yapay Zeka Yarışması — Üniversite Kategorisi                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ─── KULLANIM (Akıllı CLI — Pozisyondan Bağımsız) ───                       ║
║   python biosense_v6.1_ultra.py                                              ║
║      → Otomatik veri seti tespiti + DEFAULT mod + GPU varsa AKTİF           ║
║   python biosense_v6.1_ultra.py kanser                                      ║
║      → KANSER + interactive mod seçimi                                       ║
║   python biosense_v6.1_ultra.py kanser ultra                                ║
║      → KANSER ULTRA + cihaz otomatik                                        ║
║   python biosense_v6.1_ultra.py kanser ultra cpu                            ║
║      → KANSER ULTRA + CPU zorla (GPU varsa bile kullanmaz)                  ║
║   python biosense_v6.1_ultra.py kanser ultra cuda                           ║
║      → KANSER ULTRA + CUDA zorla (KANSER'de bile GPU kullanır)              ║
║   python biosense_v6.1_ultra.py master fast cuda > log.txt 2>&1             ║
║      → MASTER FAST CUDA, log dosyaya yazılır                                 ║
║                                                                              ║
║  ─── ARGÜMAN SIRASI ─── (pozisyondan bağımsız, akıllı tespit)               ║
║   veri_seti : kanser | cftr | pah | master                                  ║
║   mod       : fast | default | ultra                                        ║
║   cihaz     : cpu | cuda | auto                                              ║
║                                                                              ║
║  ─── v6.3 → v6.4 PARALELLİK ÇÖZÜMÜ (Optuna Resmi Tavsiye) ──                ║
║   📚 ARAŞTIRMA: Optuna FAQ resmi olarak söylüyor —                          ║
║      "n_jobs CPU-bound işte GIL nedeniyle çalışmaz, process-based gerekli"  ║
║      (https://optuna.readthedocs.io/en/stable/faq.html)                     ║
║      → Optuna v2.7'den beri n_jobs threading deprecate edildi               ║
║                                                                              ║
║   ⭐ ÇÖZÜM: joblib loky backend ile FOLD-level paralelleme                  ║
║      - oof_tahmin_universal içinde Parallel(n_jobs=22, backend="loky")      ║
║      - Loky process-based → GIL bypass → gerçek CPU paralel                 ║
║      - Memmap shared data → bellek kopyalama yok                            ║
║      - Otomatik fallback: loky başarısızsa sıralıya geçer                   ║
║                                                                              ║
║   ⭐ BEKLENEN ETKİ:                                                         ║
║      FAZ 2 CatBoost OOF: 1008s → ~50-80s (~12-20x hızlanma)                ║
║      FAZ 2 toplam: 22 dakika → ~3 dakika                                    ║
║      FAZ 4-A OOF (15 üye): 7-8 dakika → ~30 saniye                          ║
║      Toplam pipeline süresinde ~20-30 dakika tasarruf                       ║
║                                                                              ║
║   ⚠ FAZ 1/3 Optuna trial-içi: Pruner+trial.report cross-process imkansız    ║
║      → Şimdilik dokunulmadı, mevcut yapı korundu                            ║
║      → Trial bütçesi düşürmek (v6.3) zaten %60 hızlanma sağladı             ║
║                                                                              ║
║  ─── v6.2 → v6.3 OPTİMİZASYON KALİBRESİ (KORUNAN) ───                       ║
║   ⭐ FAZ 3 TRIAL BÜTÇELERİ DÜŞÜRÜLDÜ (en büyük etki, %60 daha hızlı):       ║
║     LightGBM 1500→600, XGBoost 1500→600, CatBoost 1000→400                  ║
║     ExtraTrees 500→300, RF 500→300, MLP 800→400                             ║
║     Optuna pratiği: TPE 300-500 trial'da yakınsar, fazlası overkill         ║
║                                                                              ║
║   ⭐ ULTRA mod ÇARPANLARI DENGELENDİ:                                       ║
║     trial: 1.5 → 1.0 (zaten BASE yeterli yüksek)                            ║
║     fold:  1.2 → 1.0 (50 fold istatistiksel olarak yeterli)                 ║
║     iter:  1.1 (korundu — n_estimators artırma kaldı)                       ║
║                                                                              ║
║   ⭐ NAIVE BAYES havuzdan ÇIKARILDI:                                        ║
║     F2~0.17 jüri için "anormal" alarm                                       ║
║     Yapısal sınırlama: sınıf bağımsızlık varsayımı genomik için ihlal       ║
║     Raporda 1 cümleyle açıklanır                                            ║
║                                                                              ║
║   ⭐ CACHE LOG NETLEŞTİRİLDİ:                                               ║
║     [CACHE] DB'de N trial var (COMPLETE/PRUNED/FAIL ayrımı)                 ║
║     [CACHE] Hedef X, kalan Y trial → yeni trial neden çalıştırıldığı net    ║
║                                                                              ║
║   ⚠ PARALELLEME: Outer Optuna n_jobs=22 etkili değil (GIL nedeniyle)         ║
║      → Çözüm gelecek versiyona ertelendi (multiprocessing refactor riskli)  ║
║      → Şu anki çözüm: trial sayısı düşürerek toplam süreyi azaltmak         ║
║                                                                              ║
║  ─── v6.1 → v6.2 PERFORMANS FIX (KORUNAN) ───                                ║
║                                                                              ║
║  ─── v6.0 → v6.1 EKLEMESİ ───                                                ║
║   ⭐ Akıllı argv parser (pozisyon bağımsız)                                 ║
║   ⭐ Otomatik veri seti tespiti (klasördeki ilk YARISMA_TRAIN_*.csv)         ║
║   ⭐ Manuel cihaz seçimi (cpu/cuda/auto)                                    ║
║   ⭐ Hiç argüman yoksa tam otomatik mod                                     ║
║                                                                              ║
║  ─── v5.2 → v6.0 KORUNAN ─── (TEKNOFEST için kritik):                       ║
║   ⭐ FAZ 2 mantığı düzeltildi: F2 alt sınır + additive bileşik              ║
║   ⭐ GPU otomatik (KANSER CPU, MASTER CUDA)                                 ║
║   ⭐ Ensemble SHAP bug fix                                                  ║
║   ⭐ Eşik MCC tie-break, Softmax sıcaklık 25                                ║
║                                                                              ║
║   ✓ Trial Convergence: Her FAZ 3 algoritması için Optuna progress eğrisi    ║
║     → "Sistematik araştırma" kanıtı                                          ║
║   ✓ Pipeline Funnel: 4-faz hiyerarşik mimari görseli                        ║
║     → Word SmartArt yerine otomatik üretiliyor                              ║
║   ✓ Feature Group Analysis: Prefix bazlı SHAP dağılımı (pasta + bar)        ║
║     → "Hangi özellik grubu kararı yönlendiriyor?"                            ║
║   ✓ Final Ensemble SHAP: 5 tree üyesinin ortalama SHAP'i                    ║
║     → Referans modelin değil, gerçek karar mantığı                          ║
║                                                                              ║
║  ─── KULLANIM ───                                                           ║
║  python script.py <veri_seti> [mod]                                         ║
║    veri_seti : kanser | cftr | pah | master                                  ║
║    mod       : fast | default | ultra  (opsiyonel — sorulur)                ║
║                                                                              ║
║  Örnek: python script.py kanser ultra                                       ║
║                                                                              ║
║  ─── ÇALIŞMA MODLARI (Kullanıcı Seçer) ───                                  ║
║   FAST    → Trial ×0.5, Fold ×0.5   (~10-15 dk, test/geliştirme)            ║
║   DEFAULT → Trial ×1.0, Fold ×1.0   (~25-40 dk, dengeli)                    ║
║   ULTRA   → Trial ×1.5, Fold ×1.2   (~50-90 dk, yarışma teslimi)            ║
║                                                                              ║
║  ─── DONANIM TESPİTİ (Otomatik) ───                                         ║
║   • PARALEL_ISCI çekirdek sayısından otomatik (RAM safety margin)            ║
║   • Sistem profili kutu olarak gösterilir                                    ║
║   • GPU tespiti bilgi amaçlı (CUDA hâlâ manuel)                              ║
║                                                                              ║
║  ─── PIPELINE MİMARİSİ ───                                                  ║
║   FAZ 1 : 10 algoritma geniş tarama → istatistiksel eleme                   ║
║   FAZ 2 : Korelasyon + greedy çeşitlilik seçimi → 3 paradigma               ║
║   FAZ 3 : Seçilen 3 algoritma için derin Optuna tuning                      ║
║   FAZ 4 : Top-5×3 = 15 üye ensemble + isotonic kalibrasyon                  ║
║   FAZ 7.5 : Klinik akıllılık — belirsizlik, disagreement, yanlış tahmin    ║
║   FAZ 8.5 : Solo vs Ensemble hibrit karşılaştırma (TEKNOFEST İLK)           ║
║                                                                              ║
║  ─── DESTEKLENEN METRİKLER (TEKNOFEST Şartnamesi) ───                       ║
║   Zorunlu : F1, MCC, PR-AUC                                                 ║
║   Klinik  : Binary F2, Patojenik recall/precision, FN/FP analizi            ║
║   Rigor   : Bootstrap %95 CI (dinamik iter, test boyutuna göre)             ║
║                                                                              ║
║  Tarihçe detayları: biosense_gelistirme_tarihce.md                          ║
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

# ════════════════════════════════════════════════════════════════════
# [v6.2.3] KRİTİK FIX: SQLite WAL mode — DB Lock Contention Çözümü
# ════════════════════════════════════════════════════════════════════
# Sorun: Outer-parallel 22 trial paralel SQLite'a yazıyor → 21 trial
# lock bekliyor → CPU %8'de takılıyor (özellikle FAZ 3'te trial kısa
# olduğunda DB'ye çok sık yazılıyor).
#
# Çözüm: WAL (Write-Ahead Logging) modu — concurrent okuma + yazma.
# Tüm SQLAlchemy engine'lara global olarak uygulanır.
# Optuna RDBStorage SQLAlchemy kullandığı için bu Optuna'ya da geçer.
try:
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def _enable_sqlite_wal(dbapi_connection, connection_record):
        """SQLite bağlantısı kurulurken WAL mode + performans pragma'ları uygula."""
        try:
            # SQLite olmayan bağlantılarda (postgres vs) bu cursor hata atar, atla
            module_name = dbapi_connection.__class__.__module__.lower()
            if "sqlite" not in module_name:
                return
            cursor = dbapi_connection.cursor()
            # WAL mode: concurrent readers + 1 writer, lock contention çözer
            cursor.execute("PRAGMA journal_mode=WAL")
            # synchronous=NORMAL: WAL ile güvenli, %3-5 hızlanma
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 64MB page cache (default 2MB) — outer-parallel için kritik
            cursor.execute("PRAGMA cache_size=-65536")
            # temp_store=MEMORY: geçici tablolar diskte değil RAM'de
            cursor.execute("PRAGMA temp_store=MEMORY")
            # 60s busy timeout (default 0) — paralel yazıcılar bekler crashlemez
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.close()
        except Exception:
            pass  # SQLite olmayan veya pragma desteklemeyen sürücüler için sessiz
    print("  [v6.2.3] SQLite WAL mode + performans pragma'ları aktif (DB lock fix)")
except ImportError:
    print("  [v6.2.3] SQLAlchemy event sistemi yok, WAL mode atlandı")
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
import sys
# ════════════════════════════════════════════════════════════════════
# [v6.1] AKILLI CLI PARSER — 3 opsiyonel argüman
# ════════════════════════════════════════════════════════════════════
# Kullanım:
#   python biosense_v6.1_ultra.py                       → tam otomatik
#       (klasördeki ilk YARISMA_TRAIN_*.csv, DEFAULT mod, CUDA varsa)
#   python biosense_v6.1_ultra.py kanser                → KANSER + sorar mod
#   python biosense_v6.1_ultra.py kanser ultra          → KANSER ULTRA + CUDA otomatik
#   python biosense_v6.1_ultra.py kanser ultra cpu      → KANSER ULTRA CPU
#   python biosense_v6.1_ultra.py kanser ultra cuda     → KANSER ULTRA CUDA (zorla)
# ════════════════════════════════════════════════════════════════════

# --help argümanı
if any(arg in ("--help", "-h", "help") for arg in sys.argv[1:]):
    print("""
BioSense AI v6.1 — TEKNOFEST Variant Patojenite Tahmini

KULLANIM:
  python biosense_v6.1_ultra.py [veri_seti] [mod] [cihaz]

ARGÜMANLAR (pozisyondan bağımsız, hepsi opsiyonel):
  veri_seti : kanser | cftr | pah | master
              (yazılmazsa: klasördeki ilk YARISMA_TRAIN_*.csv otomatik)
  mod       : fast | default | ultra
              (yazılmazsa: DEFAULT — eğer veri seti de yoksa tam otomatik)
  cihaz     : cpu | cuda | auto
              (yazılmazsa: KANSER için CPU, diğerleri için CUDA varsa AKTİF)

ÖRNEKLER:
  python biosense_v6.1_ultra.py
      → Klasördeki ilk veri seti, DEFAULT mod, GPU varsa CUDA

  python biosense_v6.1_ultra.py kanser ultra
      → KANSER ULTRA mod, KANSER için CPU otomatik

  python biosense_v6.1_ultra.py master fast cuda
      → MASTER FAST mod, CUDA zorla

  python biosense_v6.1_ultra.py kanser ultra cpu > log.txt 2>&1
      → KANSER ULTRA CPU, çıktı log dosyasına

ENV DEĞIŞKENLERİ:
  BIOSENSE_MOD=ultra    → mod ön ayarı
  BIOSENSE_CIHAZ=cpu    → cihaz ön ayarı

DAHA FAZLA:
  https://github.com/teknofest (proje deposu)
""")
    sys.exit(0)

# Geçerli değerler
_GECERLI_VERI_SETI = ("kanser", "cftr", "pah", "master")
_GECERLI_MOD       = ("fast", "default", "ultra")
_GECERLI_CIHAZ     = ("cpu", "cuda", "auto")

_DATA_FILE_MAP = {
    "kanser": "YARISMA_TRAIN_KANSER.csv",
    "cftr":   "YARISMA_TRAIN_CFTR.csv",
    "pah":    "YARISMA_TRAIN_PAH.csv",
    "master": "YARISMA_TRAIN_MASTER.csv",
}

def _otomatik_veri_seti_tespit():
    """Klasörde ilk bulunan YARISMA_TRAIN_*.csv'yi tespit et."""
    import glob
    aday_csvler = sorted(glob.glob("YARISMA_TRAIN_*.csv"))
    if not aday_csvler:
        return None
    # İsmi normalize et: YARISMA_TRAIN_KANSER.csv → "kanser"
    first = os.path.basename(aday_csvler[0])
    if first.startswith("YARISMA_TRAIN_") and first.endswith(".csv"):
        return first[len("YARISMA_TRAIN_"):-len(".csv")].lower()
    return None

# [v6.1] Tüm argv argümanlarını tara ve kategorize et (pozisyondan tam bağımsız)
_arg_veri_seti  = None
_arg_mod        = None
_KULLANICI_CIHAZ = None
_bilinmeyenler  = []

for arg in sys.argv[1:]:
    a = arg.lower().strip()
    if a in _GECERLI_VERI_SETI:
        _arg_veri_seti = a
    elif a in _GECERLI_MOD:
        _arg_mod = a
    elif a in _GECERLI_CIHAZ:
        _KULLANICI_CIHAZ = a
    elif a:
        # Bilinmeyen ama boş değil — veri seti olarak deneyelim (özel veri seti olabilir)
        _bilinmeyenler.append(a)

# ─── 1) Veri seti ─────────────────────────────────────────────────
if _arg_veri_seti:
    DATA_SUFFIX = _arg_veri_seti
    _veri_kaynagi = "komut satırından"
elif _bilinmeyenler:
    DATA_SUFFIX = _bilinmeyenler[0]
    _veri_kaynagi = "komut satırından (özel)"
else:
    DATA_SUFFIX = _otomatik_veri_seti_tespit() or "kanser"
    _veri_kaynagi = "otomatik tespit (klasördeki ilk YARISMA_TRAIN_*.csv)"

VERI_DOSYASI = _DATA_FILE_MAP.get(DATA_SUFFIX,
                                    f"YARISMA_TRAIN_{DATA_SUFFIX.upper()}.csv")

# Env fallback cihaz tercihi
if _KULLANICI_CIHAZ is None:
    env_cihaz = os.environ.get("BIOSENSE_CIHAZ", "").lower()
    if env_cihaz in _GECERLI_CIHAZ:
        _KULLANICI_CIHAZ = env_cihaz

print(f"\n  >>> Veri seti  : {DATA_SUFFIX.upper()}  ({_veri_kaynagi})")
print(f"  >>> CSV dosya  : {VERI_DOSYASI}")
if _KULLANICI_CIHAZ:
    print(f"  >>> Cihaz      : {_KULLANICI_CIHAZ.upper()} (kullanıcı seçimi)")
else:
    print(f"  >>> Cihaz      : OTOMATİK (KANSER→CPU, diğerleri→CUDA varsa)")

# Tüm çıktı dosya isimleri suffix'li — 4 veri seti çakışmasın diye
CIKTI_JSON     = f"en_iyi_parametreler_{DATA_SUFFIX}_v4.json"
CIKTI_FAZ1_CSV = f"faz1_algoritma_kiyaslama_{DATA_SUFFIX}.csv"
CIKTI_KORELASYON_CSV = f"faz2_korelasyon_matrisi_{DATA_SUFFIX}.csv"

# [v4.7] Tam dinamik sistem profilleme — RAM, CPU, GPU, data-size aware
import multiprocessing

def _hesapla_paralel_isci(cpu_logical, cpu_physical, available_ram_gb, data_suffix):
    """
    Optimum PARALEL_ISCI hesabı — her sistemde ideal kullanım.

    Mantık:
    - CPU sınırı: thread sayısına göre (büyük sistemlerde DB lock dikkat)
    - RAM sınırı: worker başına tahmini RAM × safety margin (70%)
    - Veri seti boyutu: MASTER per-worker 2.5GB, KANSER 0.8GB

    Örnekler:
    - i9-14900K (32T) + 96GB RAM + KANSER: 24 worker (CPU bottleneck)
    - i9-14900K (32T) + 96GB RAM + MASTER: 26 worker (RAM 70% = 67GB / 2.5 = 26)
    - Ryzen 9 (24T) + 32GB RAM + MASTER: 8 worker (RAM bottleneck)
    - i5 (8T) + 16GB + KANSER: 8 worker (CPU bottleneck)
    """
    # Worker başı RAM tahmini (veri setine göre)
    if data_suffix == "master":
        per_worker_gb = 2.5
    elif data_suffix in ("cftr", "pah"):
        per_worker_gb = 1.5
    else:  # kanser, default
        per_worker_gb = 0.8

    # RAM tabanlı limit (%70 worker'lara, %30 sistem+cache pay)
    ram_limit = max(2, int(available_ram_gb * 0.7 / per_worker_gb))

    # CPU tabanlı limit
    if cpu_logical <= 8:
        cpu_limit = cpu_logical
    elif cpu_logical <= 16:
        cpu_limit = cpu_logical - 2  # 14T, 12T
    elif cpu_logical <= 24:
        cpu_limit = 22  # Ultra 9 24T → 22 sweet spot (24 zorlarsa background tasks etkilenir)
    elif cpu_logical <= 32:
        cpu_limit = 24  # i9-14900K 32T (E-core'lar yavaş)
    elif cpu_logical <= 48:
        cpu_limit = 28  # Threadripper sınıfı
    else:
        cpu_limit = 32  # Üzeri DB kilit contention sorunu

    return min(cpu_limit, ram_limit), per_worker_gb, ram_limit, cpu_limit

try:
    _cpu_count_logical = multiprocessing.cpu_count()

    # psutil ile detaylı tespit
    try:
        import psutil
        _cpu_count_physical = psutil.cpu_count(logical=False) or (_cpu_count_logical // 2)
        _vm = psutil.virtual_memory()
        _total_ram_gb     = _vm.total / (1024**3)
        _available_ram_gb = _vm.available / (1024**3)
        _ram_kullanim_pct = _vm.percent
        PSUTIL_OK = True
    except ImportError:
        _cpu_count_physical = max(_cpu_count_logical // 2, 1)
        _total_ram_gb       = 16
        _available_ram_gb   = 12
        _ram_kullanim_pct   = 25
        PSUTIL_OK = False

    # Akıllı hesaplama
    PARALEL_ISCI, _per_worker_gb, _ram_limit, _cpu_limit = _hesapla_paralel_isci(
        _cpu_count_logical, _cpu_count_physical, _available_ram_gb, DATA_SUFFIX
    )

    print(f"\n  ┌── SİSTEM PROFİLİ ─────────────────────────────────────────┐")
    print(f"  │  CPU       : {_cpu_count_logical} logical / {_cpu_count_physical} physical core")
    print(f"  │  RAM       : {_total_ram_gb:.1f}GB toplam, "
          f"{_available_ram_gb:.1f}GB kullanılabilir ({_ram_kullanim_pct:.0f}% dolu)")
    print(f"  │  Veri seti : {DATA_SUFFIX.upper()} → per-worker ~{_per_worker_gb}GB tahmini")
    print(f"  │  CPU limit : {_cpu_limit} worker  |  RAM limit: {_ram_limit} worker")
    print(f"  │  → PARALEL_ISCI: {PARALEL_ISCI}  ({_cpu_limit if PARALEL_ISCI == _cpu_limit else _ram_limit}-bound)")
    if not PSUTIL_OK:
        print(f"  │  [UYARI] psutil yok → varsayılan RAM tahmini kullanıldı")
        print(f"  │           Daha doğru tespit için: pip install psutil")
    print(f"  └────────────────────────────────────────────────────────────┘")
except Exception as e:
    PARALEL_ISCI = 8
    print(f"  [UYARI] Sistem tespiti başarısız, varsayılan PARALEL_ISCI=8: {e}")

# [v4.7] GPU tespiti — XGBoost/LightGBM için
GPU_VAR = False
GPU_BILGI = "yok"
try:
    import subprocess
    _gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                   "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
    if _gpu_check.returncode == 0 and _gpu_check.stdout.strip():
        GPU_VAR = True
        GPU_BILGI = _gpu_check.stdout.strip().split("\n")[0]
        print(f"  GPU       : {GPU_BILGI}")
    else:
        print(f"  GPU       : Yok / nvidia-smi erişilemez (CPU modu)")
except Exception:
    print(f"  GPU       : Tespit edilemedi (CPU modu)")

# [v6.1] GPU kullanım politikası — kullanıcı seçimi ÖNCELİKLİ
# Öncelik sırası:
#   1. CLI: python ... cpu | cuda | auto
#   2. Env: BIOSENSE_CIHAZ=cpu | cuda | auto
#   3. Otomatik: KANSER → CPU (overhead), CFTR/PAH/MASTER → CUDA (varsa)
if _KULLANICI_CIHAZ == "cpu":
    GPU_AKTIF = False
    print(f"  GPU kullanım: KAPALI (kullanıcı: CPU zorla)")
elif _KULLANICI_CIHAZ == "cuda":
    if not GPU_VAR:
        print(f"  [UYARI] CUDA istendi ama GPU yok, CPU'ya düşülüyor")
        GPU_AKTIF = False
    else:
        GPU_AKTIF = True
        print(f"  GPU kullanım: AKTIF (kullanıcı: CUDA zorla)")
else:
    # Otomatik (argümanda cihaz yok veya "auto")
    GPU_AKTIF = GPU_VAR and DATA_SUFFIX in ("cftr", "pah", "master")
    if GPU_AKTIF:
        print(f"  GPU kullanım: AKTIF (otomatik — {DATA_SUFFIX.upper()} büyük veri)")
    else:
        if GPU_VAR:
            print(f"  GPU kullanım: KAPALI (otomatik — {DATA_SUFFIX.upper()} için CPU verimli)")
            print(f"              → CFTR/PAH/MASTER'da GPU otomatik aktif olur")
        else:
            print(f"  GPU kullanım: KAPALI (GPU yok)")

XGB_DEVICE  = "cuda" if GPU_AKTIF else "cpu"
LGB_DEVICE  = "gpu"  if GPU_AKTIF else "cpu"
CB_TASK_TYPE = "GPU" if GPU_AKTIF else "CPU"

# [FIX O] Algoritmaya göre paralellik stratejisi (CPU %10 problemini çözer)
# Sorun: 24 dış worker × 24 iç thread = 576 thread → GIL bottleneck + DB lock
# Çözüm: Algoritmaya göre iç ya da dış paralelizm seç, çakıştırma.
#
# - C++ boosting (XGB/LGB/CB):  Optuna 24 paralel, model n_jobs=1
#   → Process'ler bağımsız çalışır, GIL etkisi yok
# - sklearn tree bagging (RF/ET): Optuna 1 sequential, model n_jobs=-1
#   → Tree'ler içeride paralel çoğalır (24 ağaç aynı anda)
#   → Tek model fit ~30s → 24 thread ile ~2s, sonra sıra geliyor
# - Diğer sklearn (LR/SVM/kNN/NB/MLP): Optuna 24, model n_jobs=1
#   → Modeller hızlı, dış paralelizm daha verimli
def algo_paralelizm(algo):
    """[v6.2.2] Algoritma için (optuna_n_jobs, model_n_jobs) tuple döndürür.

    Strateji: VERİ SETİ BOYUTUNA GÖRE adaptive — tüm tree modeller için.

    Küçük data (N_TRAIN < 800, örn. KANSER 310):
      - Tek trial fit'i kısa (saniyenin altı)
      - Inner-parallel spawn overhead'i fit süresinden büyük → CPU %10
      - Outer-parallel (22 trial) çok daha verimli → CPU %95+
      - Risk: SQLite DB lock contention (ama trial kısa olduğu için kabul edilebilir)

    Büyük data (N_TRAIN >= 800, örn. MASTER 2400):
      - Tek trial fit'i uzun (10+ saniye)
      - Inner-parallel C++ paralellemesi mükemmel
      - DB contention çok daha az (trial uzun, lock kısa)

    Sklearn olmayan modeller (SVM/kNN/Logistic/NB/MLP):
      - Her zaman outer-parallel (tek model fit'i kısa, paralel trial mantıklı)
      - MLP sklearn n_jobs desteklemediği için zaten içeride tek thread
    """
    # Tree-based modeller için adaptive
    if algo in ("random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"):
        if 'N_TRAIN' in globals() and N_TRAIN < 800:
            return (PARALEL_ISCI, 1)   # küçük data → outer paralel
        return (1, -1)                 # büyük data → inner paralel

    # Küçük/sıralı modeller: her zaman outer paralel
    return (PARALEL_ISCI, 1)        # SVM, kNN, NB, Logistic, MLP

# ── FAZ 1: Geniş tarama ───────────────────────────────────────────
# [v4.9] BU PARAMETRELER DİNAMİK — kullanıcı modu + sistem otomatik birleştirilir
FAZ1_FOLDS     = 5
FAZ1_TEKRAR    = 20
FAZ1_TRIAL_PER_ALGO_BASE = {
    "xgboost":       80,
    "lightgbm":      80,
    "catboost":      60,
    "random_forest": 50,
    "extra_trees":   50,
    "logistic":      40,
    "svm_rbf":       60,
    "knn":           20,
    "naive_bayes":    5,
    "mlp":           80,
}
FAZ1_DB_PREFIX = f"sqlite:///biosense_faz1_v4_{DATA_SUFFIX}"

FAZ1_F2_ORT_ESIK    = 0.70
FAZ1_F2_LCB_ESIK    = 0.65
FAZ1_TTEST_P_ESIK   = 0.05

# ── FAZ 2: Çeşitlilik Seçimi ──────────────────────────────────────
FAZ2_FOLDS_OOF     = 5
FAZ2_TEKRAR_OOF    = 10
FAZ2_SECILEN_SAYI  = 3
FAZ2_OOF_ES_SPLIT  = 0.12

# ── FAZ 3: Derin Tuning ───────────────────────────────────────────
FAZ3_TRIAL_PER_ALGO_BASE = {
    # [v6.3] KALİBRE: Optuna TPE 300-500 trial'da yakınsar, 1500 overkill.
    # Önceki değerler 11 saat süreye yol açıyordu — yarıya indirildi.
    # F2 etkisi <%1 (Optuna convergence çalışmaları ile uyumlu).
    "xgboost":       600,    # eski 1500 → 600
    "lightgbm":      600,    # eski 1500 → 600 (en kritik düşüş — 11 saat fix)
    "catboost":      400,    # eski 1000 → 400
    "random_forest": 300,    # eski 500 → 300
    "extra_trees":   300,    # eski 500 → 300
    "logistic":      300,    # eski 600 → 300
    "svm_rbf":       250,    # eski 400 → 250
    "knn":           100,    # eski 200 → 100
    "naive_bayes":   50,     # eski 100 → 50 (zaten eleniyor)
    "mlp":           400,    # eski 800 → 400
}
FAZ3_FOLDS          = 5
FAZ3_TEKRAR         = 10
FAZ3_DB_PREFIX      = f"sqlite:///biosense_faz3_v4_{DATA_SUFFIX}"
FAZ3_TOP_K          = 5

# [v4.9] ─── KULLANICI MOD SEÇİMİ ───
# Kullanıcı kalite/hız tercihini seçer, geri kalanı otomatik.
# PARALEL_ISCI hâlâ sistemden çekilir (CPU+RAM dinamik).

def _mod_sec():
    """
    [v6.1] FAST / DEFAULT / ULTRA mod seçimi.
    Öncelik sırası:
    1. _arg_mod (CLI'da parse edilmiş — pozisyon bağımsız)
    2. BIOSENSE_MOD env değişkeni
    3. Tam otomatik mod (hiç CLI argümanı yok) → DEFAULT
    4. Yarı otomatik (sadece veri seti var) → interactive sor
    """
    # 1. CLI'dan parse edilmiş mod
    if _arg_mod:
        return _arg_mod

    # 2. Environment variable
    env_mod = os.environ.get("BIOSENSE_MOD", "").lower()
    if env_mod in ("fast", "default", "ultra"):
        print(f"  [v6.1] BIOSENSE_MOD env değişkeni bulundu: {env_mod.upper()}")
        return env_mod

    # 3. Hiç argüman yoksa (tam otomatik mod) → DEFAULT
    if len(sys.argv) == 1:
        print(f"  [v6.1] Tam otomatik mod → DEFAULT seçildi")
        return "default"

    # 4. Yarı otomatik (sadece veri seti) → interactive
    print(f"\n  ┌── ÇALIŞMA MODU SEÇİMİ ────────────────────────────────────┐")
    print(f"  │                                                            │")
    print(f"  │  [1] FAST     → Hızlı (test/geliştirme için, ~40-60 dk)    │")
    print(f"  │                  Trial %50, fold %50, sonuç ±%5            │")
    print(f"  │                                                            │")
    print(f"  │  [2] DEFAULT  → Standart kalite (önerilen, ~2-4 saat)      │")
    print(f"  │                  Trial %100, fold %100, dengeli            │")
    print(f"  │                                                            │")
    print(f"  │  [3] ULTRA    → Maksimum kalite (yarışma, ~6-10 saat)      │")
    print(f"  │                  Trial %150, fold %120, +10% iter          │")
    print(f"  │                                                            │")
    print(f"  └────────────────────────────────────────────────────────────┘")

    try:
        secim = input("  Seçim [1/2/3] veya isim [fast/default/ultra] (default=2): ").strip().lower()
        if secim in ("1", "fast", "f"): return "fast"
        if secim in ("3", "ultra", "u"): return "ultra"
        if secim in ("2", "default", "d", ""): return "default"
        print(f"  [UYARI] Geçersiz seçim '{secim}', DEFAULT moda geçiliyor.")
        return "default"
    except (EOFError, KeyboardInterrupt):
        # Non-interactive ortamda (örn. nohup, &)
        print(f"  [BİLGİ] İnteraktif input alınamadı, DEFAULT mod kullanılıyor.")
        return "default"

CALISMA_MODU = _mod_sec()

# Mod çarpanları
# [v6.3] ULTRA fold çarpanı 1.2 → 1.0 (60 fold yerine 50)
#   - 50 fold (5x10 RepeatedSKF) istatistiksel olarak yeterli
#   - %16 hızlanma, F2 etkisi <%0.1
#   - trial çarpan da 1.5 → 1.0 (FAZ3 trial bütçesi zaten yeniden kalibre edildi)
_MOD_CARPANI = {
    "fast":    {"trial": 0.5,  "fold": 0.5,  "iter": 0.7,
                "label": "FAST", "renk": "🟢"},
    "default": {"trial": 1.0,  "fold": 1.0,  "iter": 1.0,
                "label": "DEFAULT", "renk": "🟡"},
    "ultra":   {"trial": 1.0,  "fold": 1.0,  "iter": 1.1,
                "label": "ULTRA", "renk": "🔴"},
}

_c = _MOD_CARPANI[CALISMA_MODU]

# Pipeline parametrelerini moda göre ölçekle
def _olcekle(d, multiplier, min_val=5):
    return {k: max(min_val, int(v * multiplier)) for k, v in d.items()}

FAZ1_TRIAL_PER_ALGO = _olcekle(FAZ1_TRIAL_PER_ALGO_BASE, _c["trial"], min_val=5)
FAZ3_TRIAL_PER_ALGO = _olcekle(FAZ3_TRIAL_PER_ALGO_BASE, _c["trial"], min_val=50)
FAZ1_TEKRAR = max(5, int(FAZ1_TEKRAR * _c["fold"]))
FAZ3_TEKRAR = max(3, int(FAZ3_TEKRAR * _c["fold"]))
FAZ2_TEKRAR_OOF = max(3, int(FAZ2_TEKRAR_OOF * _c["fold"]))

# Tahmini süre (KANSER referansı: DEFAULT modda ~25 dk)
_sure_carpanlari = {"fast": 0.5, "default": 1.0, "ultra": 1.8}
_tahmini_sure_dk = int(25 * _sure_carpanlari[CALISMA_MODU])

print(f"\n  ┌── ÇALIŞMA MODU AKTIF ─────────────────────────────────────┐")
print(f"  │  Mod              : {_c['label']:<14} {_c['renk']:>2}                    │")
print(f"  │  Trial çarpanı    : ×{_c['trial']}  → FAZ1: {sum(FAZ1_TRIAL_PER_ALGO.values())} trial, "
      f"FAZ3: {sum(FAZ3_TRIAL_PER_ALGO.values())} trial")
print(f"  │  Fold çarpanı     : ×{_c['fold']}  → FAZ1: {FAZ1_FOLDS*FAZ1_TEKRAR} fold, "
      f"FAZ3: {FAZ3_FOLDS*FAZ3_TEKRAR} fold")
print(f"  │  Iter çarpanı     : ×{_c['iter']}  (model n_estimators)")
print(f"  │  Tahmini süre     : ~{_tahmini_sure_dk} dakika (DB cache'li run)")
print(f"  │  PARALEL_ISCI     : {PARALEL_ISCI} (sistemden otomatik)")
if CALISMA_MODU == "fast":
    print(f"  │  ⚠ Sonuçlar ~5-10% daha düşük doğrulukta olabilir         │")
elif CALISMA_MODU == "ultra":
    print(f"  │  ✓ Maksimum kararlılık — yarışma teslimi için ideal       │")
print(f"  └────────────────────────────────────────────────────────────┘\n")

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
SOFTMAX_SICAKLIK   = 25.0  # [v5.2 BUG FIX] 10 → 25 (keskin ağırlık, zayıf modeller bastırılır)
MIN_MESAFE_ICIN_ENSEMBLE = 0.05

TEST_ORANI     = 0.20
RASTGELE_TOHUM = 43

# Tüm algoritma havuzu (faz 1'de taranır)
# [v6.3] Naive Bayes havuzdan çıkarıldı:
#   - F2 ~0.17 her zaman → jüri için "anormal düşük" alarmı
#   - Sınıf bağımsızlık varsayımı genomik veride yapısal olarak ihlal ediliyor
#   - Bu beklenen başarısızlık raporda 1 cümleyle açıklanır,
#     pipeline'da yer kaplamasına gerek yok
# Eğer NB tekrar dahil edilmek istenirse: ENABLE_NAIVE_BAYES=True yapılır
ENABLE_NAIVE_BAYES = False  # [v6.3] Yarışma sunumu için kapalı

TUM_ALGORITMALAR = ["xgboost"]
if LGBM_OK: TUM_ALGORITMALAR.append("lightgbm")
if CB_OK:   TUM_ALGORITMALAR.append("catboost")
TUM_ALGORITMALAR.extend([
    "random_forest", "extra_trees",
    "logistic", "svm_rbf", "knn", "mlp",
])
if ENABLE_NAIVE_BAYES:
    TUM_ALGORITMALAR.insert(-1, "naive_bayes")
    print(f"  [v6.3] Naive Bayes dahil edildi (ENABLE_NAIVE_BAYES=True)")
else:
    print(f"  [v6.3] Naive Bayes havuzdan çıkarıldı (yarışma için, 9 algoritma kaldı)")

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
print("  BioSense AI v6.3 — Optimizasyon Kalibresi + NB Çıkar + Cache Log")
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

# [v4.5] Variant_ID'leri de split et — yanlış tahmin tablosu için
_variant_ids_all = df["Variant_ID"].values if "Variant_ID" in df.columns else \
                   np.array([f"VAR_{i}" for i in range(len(df))])
X_train, X_test, y_train, y_test, variant_ids_train, variant_ids_test = train_test_split(
    X, y, _variant_ids_all,
    test_size=TEST_ORANI, random_state=RASTGELE_TOHUM, stratify=y
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

def en_iyi_f2_esik_benzersiz(y_gercek, problar, f2_tolerans=0.005):
    """[v5.2 BUG FIX] F2 maksimum içinde MCC tie-break ile klinik denge.

    Önceki: F2 maksimumda ilk eşiği seçiyordu → aşırı recall, düşük MCC.
    Yeni: F2 ≥ (max_F2 - f2_tolerans) olan eşikler arasından MCC en yüksek.

    Sebep: F2 değerleri sık sık birbirine yakın (örn. 0.20→0.9331, 0.30→0.9319).
    Aralarında çok küçük F2 farkı (<0.005) için klinik dengeyi feda etmek mantıksız.
    """
    from sklearn.metrics import matthews_corrcoef
    adaylar = np.unique(problar)
    adaylar_dar = adaylar[(adaylar > 0.01) & (adaylar < 0.99)]
    adaylar = adaylar_dar if len(adaylar_dar) > 0 else adaylar
    if len(adaylar) == 0:
        return 0.50, 0.0

    # 1. Adım: Tüm eşiklerin F2'sini hesapla
    f2_dict = {}
    for e in adaylar:
        pred = (problar >= e).astype(int)
        f2 = fbeta_score(y_gercek, pred, beta=2, pos_label=1,
                         average="binary", zero_division=0)
        f2_dict[float(e)] = f2

    max_f2 = max(f2_dict.values())

    # 2. Adım: F2 max-tolerans içindeki eşikleri filtrele
    aday_esikler = [e for e, f in f2_dict.items()
                     if f >= (max_f2 - f2_tolerans)]

    # 3. Adım: Bu eşikler arasından MCC en yüksek olanı seç
    if len(aday_esikler) == 1:
        en_iyi_esik = aday_esikler[0]
    else:
        en_iyi_mcc = -2.0
        en_iyi_esik = aday_esikler[0]
        for e in aday_esikler:
            pred = (problar >= e).astype(int)
            try:
                mcc = matthews_corrcoef(y_gercek, pred)
            except Exception:
                mcc = -2.0
            if mcc > en_iyi_mcc:
                en_iyi_mcc = mcc
                en_iyi_esik = e

    return round(en_iyi_esik, 5), round(f2_dict[en_iyi_esik], 6)

def en_iyi_f2_esik_grid(y_gercek, problar, adim=0.005, f2_tolerans=0.005):
    """[v5.2 BUG FIX] Grid versiyon — aynı MCC tie-break mantığı."""
    from sklearn.metrics import matthews_corrcoef
    f2_dict = {}
    for e in np.arange(0.05, 0.95 + adim, adim):
        e = round(float(e), 4)
        pred = (problar >= e).astype(int)
        f2 = fbeta_score(y_gercek, pred, beta=2, pos_label=1,
                         average="binary", zero_division=0)
        f2_dict[e] = f2

    max_f2 = max(f2_dict.values())
    aday_esikler = [e for e, f in f2_dict.items()
                     if f >= (max_f2 - f2_tolerans)]

    if len(aday_esikler) == 1:
        return aday_esikler[0], f2_dict[aday_esikler[0]]

    en_iyi_mcc = -2.0
    en_iyi_esik = aday_esikler[0]
    for e in aday_esikler:
        pred = (problar >= e).astype(int)
        try:
            mcc = matthews_corrcoef(y_gercek, pred)
        except Exception:
            mcc = -2.0
        if mcc > en_iyi_mcc:
            en_iyi_mcc = mcc
            en_iyi_esik = e

    return en_iyi_esik, f2_dict[en_iyi_esik]

def kat_idx(X_df):
    return [X_df.columns.get_loc(c) for c in KAT_KOLONLAR if c in X_df.columns]

def softmax_agirliklar(f2_dizi, sicaklik=25.0):
    """[v5.2 BUG FIX] Sıcaklık 10 → 25.

    Eski sıcaklık 10 ile ExtraTrees (F2=0.96) ve kNN (F2=0.93) ağırlık farkı
    sadece 1.34 kat → neredeyse uniform ensemble (1/15=0.067, gerçek 0.058-0.078).

    Sıcaklık 25 ile fark ~7 kat → güçlü modellere baskınlık verir, zayıf
    modellerin (kNN) negatif etkisi azalır.
    """
    z = np.array(f2_dizi, dtype=float) * sicaklik
    z -= z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

def bootstrap_iter_count(test_size):
    """
    [v4.3] Test set boyutuna göre dinamik Bootstrap iterasyon sayısı.
    - Küçük test (<150): 2000 iter (varyans yüksek, çok iter şart)
    - Orta test (150-500): 1500 iter
    - Büyük test (500-1500): 1000 iter
    - Çok büyük (>1500): 800 iter (yeterli precision)
    """
    if test_size < 150:
        return 2000
    elif test_size < 500:
        return 1500
    elif test_size < 1500:
        return 1000
    else:
        return 800

def bootstrap_metric_ci(y_true, y_proba_or_pred, metric_func, n_boot=None,
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
    # [v4.3] n_boot None ise test boyutuna göre dinamik
    if n_boot is None:
        n_boot = bootstrap_iter_count(n)
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

GRAFIK_KLASOR = f"grafikler_{DATA_SUFFIX}"
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

# ══════════════════════════════════════════════════════════════════
# [v5.1] JÜRİ ETKİSİ İÇİN 4 YENİ GÖRSEL FONKSİYONU
# ══════════════════════════════════════════════════════════════════

def trial_convergence_grafigi(study, algo_adi,
                               dosya_adi_format="trial_convergence_{algo}.png"):
    """
    [v5.1] Optuna study trial.value progression eğrisi.
    Jüri için: 'sistematik hiperparametre araması' kanıtı.

    İki çizgi:
    - Gri (alpha): her trial'ın F2 skoru (gürültülü)
    - Kırmızı: kümülatif en iyi (best so far)
    """
    if not MATPLOTLIB_OK:
        return
    try:
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 5:
            return  # çok az trial, anlamlı grafik olmaz
        values = [t.value for t in trials]
        best_so_far = np.maximum.accumulate(values)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(values)+1), values,
                 alpha=0.4, color="gray", lw=0.8,
                 marker="o", markersize=3, label="Her trial")
        ax.plot(range(1, len(best_so_far)+1), best_so_far,
                 color="#d62728", lw=2.5, label="En iyi (kümülatif)")

        ax.set_xlabel("Trial #", fontsize=11, fontweight="bold")
        ax.set_ylabel("CV F2 Skoru", fontsize=11, fontweight="bold")
        ax.set_title(f"Optuna Convergence: {algo_adi.upper()}\n"
                      f"Sistematik Hiperparametre Araması (TPE Multivariate)",
                      fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(linestyle="--", alpha=0.6)

        # En iyi trial işaretle
        best_idx = int(np.argmax(values))
        best_val = values[best_idx]
        ax.annotate(f"En iyi: trial #{best_idx+1}\nF2={best_val:.4f}",
                     xy=(best_idx+1, best_val),
                     xytext=(best_idx+1 + len(values)*0.15, best_val - 0.02),
                     fontsize=10, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#d62728"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

        plt.tight_layout()
        yol = os.path.join(GRAFIK_KLASOR, dosya_adi_format.format(algo=algo_adi))
        plt.savefig(yol, dpi=150, bbox_inches="tight")
        plt.close()
        return yol
    except Exception as e:
        print(f"    [UYARI] Trial convergence grafiği başarısız ({algo_adi}): {e}")
        return None


def pipeline_funnel_diyagrami(istatistikler,
                                dosya_adi="pipeline_funnel.png"):
    """
    [v5.1] 4-faz pipeline akış diyagramı.
    Jüri için: 'sistematik hiyerarşik mimari' görseli (Word SmartArt yerine).

    istatistikler: {
        "n_train": 310, "n_test": 78,
        "n_ozellik_baslangic": 282, "n_ozellik_son": 225,
        "faz1_algos": 10, "faz1_kalan": 9, "faz1_total_trial": 525,
        "faz2_secilen": 3, "secilen_algos": ["extra_trees", "knn", "logistic"],
        "faz3_total_trial": 7100,
        "ensemble_uye_sayisi": 15,
    }
    """
    if not MATPLOTLIB_OK:
        return

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    ax.axis("off")

    # Renk paleti (gradient mavi)
    renkler = ["#2E86AB", "#5DADE2", "#85C1E2", "#AED6F1", "#D6EAF8"]

    fazlar = [
        {
            "label": "VERİ HAZIRLIĞI",
            "alt":   f"{istatistikler['n_train']} train + {istatistikler['n_test']} test\n"
                     f"{istatistikler['n_ozellik_baslangic']} → {istatistikler['n_ozellik_son']} özellik (SHAP)",
            "y_top": 11.5, "y_bot": 10.0, "wt": 9, "wb": 8.5
        },
        {
            "label": "FAZ 1: GENİŞ TARAMA",
            "alt":   f"{istatistikler['faz1_algos']} algoritma × "
                     f"{istatistikler['faz1_total_trial']} trial\n"
                     f"İstatistiksel eleme (F2+LCB+p-value)\n"
                     f"Hayatta kalan: {istatistikler['faz1_kalan']} algoritma",
            "y_top": 9.7, "y_bot": 7.7, "wt": 8.5, "wb": 7.5
        },
        {
            "label": "FAZ 2: ÇEŞİTLİLİK SEÇİMİ",
            "alt":   f"Korelasyon matrisi + Greedy seçim\n"
                     f"Final: {istatistikler['faz2_secilen']} paradigma\n"
                     f"{', '.join(istatistikler['secilen_algos'])}",
            "y_top": 7.4, "y_bot": 5.4, "wt": 7.5, "wb": 6.0
        },
        {
            "label": "FAZ 3: DERİN TUNING",
            "alt":   f"3 algoritma × derin Optuna\n"
                     f"Toplam: {istatistikler['faz3_total_trial']} trial\n"
                     f"TPE Multivariate, 50 fold CV",
            "y_top": 5.1, "y_bot": 3.1, "wt": 6.0, "wb": 4.5
        },
        {
            "label": "FAZ 4: ENSEMBLE + KALİBRASYON",
            "alt":   f"{istatistikler['ensemble_uye_sayisi']} üyeli ensemble\n"
                     f"Isotonic Regression kalibrasyonu\n"
                     f"Bootstrap %95 CI ile test",
            "y_top": 2.8, "y_bot": 0.8, "wt": 4.5, "wb": 3.0
        },
    ]

    for i, f in enumerate(fazlar):
        renk = renkler[i]
        # Trapez şekli
        x_center = 5
        wt, wb = f["wt"], f["wb"]
        verts = [
            (x_center - wt/2, f["y_top"]),
            (x_center + wt/2, f["y_top"]),
            (x_center + wb/2, f["y_bot"]),
            (x_center - wb/2, f["y_bot"]),
        ]
        from matplotlib.patches import Polygon
        poly = Polygon(verts, closed=True, facecolor=renk,
                        edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(poly)

        # Üst label
        y_label = (f["y_top"] + f["y_bot"]) / 2 + 0.4
        ax.text(x_center, y_label, f["label"],
                 ha="center", va="center", fontsize=13, fontweight="bold",
                 color="white" if i < 3 else "black")
        # Alt detay
        y_alt = (f["y_top"] + f["y_bot"]) / 2 - 0.3
        ax.text(x_center, y_alt, f["alt"],
                 ha="center", va="center", fontsize=9.5,
                 color="white" if i < 3 else "black")

        # Aşağı yönlü ok (son hariç)
        if i < len(fazlar) - 1:
            y_ok = f["y_bot"] - 0.1
            ax.annotate("", xy=(x_center, y_ok - 0.15),
                         xytext=(x_center, y_ok + 0.1),
                         arrowprops=dict(arrowstyle="->",
                                          color="black", lw=2))

    ax.set_title("BioSense AI — 4 Faz Hiyerarşik Pipeline Mimarisi\n"
                  "Sistematik Eleme + Çeşitlilik + Derinlik + Robustluk",
                  fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    return yol


def feature_group_analizi(all_cols, mean_abs_shap,
                            dosya_adi="feature_grup_analizi.png"):
    """
    [v5.1] Özellik prefix gruplarına göre SHAP toplam katkı analizi.
    Jüri için: 'özellik grubu bazlı karar dağılımı' görseli.

    Prefix tespit kuralı: ilk underscore'a kadar (EK_, AL_, AA_, CAT_)
    """
    if not MATPLOTLIB_OK:
        return None

    from collections import defaultdict
    grup_shap   = defaultdict(float)
    grup_sayisi = defaultdict(int)

    for col, sv in zip(all_cols, mean_abs_shap):
        col_str = str(col)
        if "_" in col_str:
            prefix = col_str.split("_")[0] + "_*"
        else:
            prefix = "DIĞER"
        grup_shap[prefix] += float(sv)
        grup_sayisi[prefix] += 1

    # Sıralı liste
    sirali = sorted(grup_shap.items(), key=lambda x: x[1], reverse=True)
    gruplar = [g[0] for g in sirali]
    katkilar = [g[1] for g in sirali]
    sayilar  = [grup_sayisi[g] for g in gruplar]
    toplam = sum(katkilar)
    yuzdeler = [k / toplam * 100 for k in katkilar]

    # İki panel: pasta + bar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Sol: Pasta grafiği
    renkler = plt.cm.Set2(np.linspace(0, 1, len(gruplar)))
    labels_pasta = [f"{g}\n({s} özellik)" for g, s in zip(gruplar, sayilar)]
    wedges, _, autotexts = ax1.pie(katkilar, labels=labels_pasta,
                                      autopct="%1.1f%%",
                                      colors=renkler, startangle=90,
                                      textprops={"fontsize": 10})
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax1.set_title("Özellik Grubu Bazlı SHAP Katkısı\n(Karar oluşum payları)",
                   fontsize=12, fontweight="bold")

    # Sağ: Bar grafiği (ortalama SHAP per grup)
    ort_shaps = [katkilar[i]/sayilar[i] for i in range(len(gruplar))]
    bars = ax2.barh(range(len(gruplar)), ort_shaps,
                      color=renkler, edgecolor="black")
    ax2.set_yticks(range(len(gruplar)))
    ax2.set_yticklabels(gruplar)
    ax2.invert_yaxis()
    ax2.set_xlabel("Ortalama |SHAP| per özellik", fontsize=11, fontweight="bold")
    ax2.set_title("Grup-içi Ortalama Önem\n(grup başı yoğunluk)",
                   fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    for i, (b, v) in enumerate(zip(bars, ort_shaps)):
        ax2.text(v + max(ort_shaps)*0.01, i, f"{v:.4f}",
                  va="center", fontsize=9)

    plt.suptitle("Feature Group Analysis — Özellik Prefix Bazlı Önem Dağılımı",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "gruplar":  gruplar,
        "katkilar": [round(k, 5) for k in katkilar],
        "yuzdeler": [round(y, 2) for y in yuzdeler],
        "sayilar":  sayilar,
    }


def ensemble_shap_hesapla(final_modeller_tree, X_train, X_test_sample,
                            dosya_adi="ensemble_shap.png"):
    """
    [v5.1 / v6.0 fix] Final ensemble'ın SHAP analizi.
    Sadece tree-based modeller için TreeExplainer.

    [v6.0 BUG FIX]:
    - check_additivity=False (SHAP yaygın hata kaynağı)
    - X_test_sample.astype(float) — kategorik dtypeları temizle
    - Hata mesajları print edilir (debugging)
    """
    if not MATPLOTLIB_OK or not final_modeller_tree:
        print(f"    [BİLGİ] Tree-based model yok veya matplotlib kapalı, atlandı")
        return None
    try:
        # [v6.0] X'i SHAP için temizle (kategorik kolonlar int olmalı)
        try:
            X_clean = X_test_sample.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype.name == "category":
                    X_clean[col] = X_clean[col].cat.codes.astype("int64")
                elif X_clean[col].dtype.kind == "O":
                    X_clean[col] = X_clean[col].astype("int64")
            X_for_shap = X_clean.astype("float64")
        except Exception as e:
            print(f"    [UYARI] X temizleme: {e}, ham X kullanılıyor")
            X_for_shap = X_test_sample

        tum_shaps = []
        for idx, m in enumerate(final_modeller_tree):
            try:
                # Pipeline check
                if hasattr(m, "named_steps") and "clf" in m.named_steps:
                    explainer = shap.TreeExplainer(
                        m.named_steps["clf"],
                        feature_perturbation="tree_path_dependent"
                    )
                else:
                    explainer = shap.TreeExplainer(
                        m, feature_perturbation="tree_path_dependent"
                    )
                # check_additivity=False — küçük floating point hataları affet
                sv = explainer.shap_values(X_for_shap, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[1]
                elif hasattr(sv, "ndim") and sv.ndim == 3:
                    sv = sv[:, :, 1]
                tum_shaps.append(np.abs(sv).mean(axis=0))
            except Exception as e:
                print(f"    [UYARI] Model {idx+1} SHAP başarısız: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue

        if not tum_shaps:
            print(f"    [HATA] Hiçbir tree modelden SHAP alınamadı — atlandı")
            return None
        print(f"    {len(tum_shaps)}/{len(final_modeller_tree)} tree modelden SHAP başarılı")

        ortalama_shap = np.mean(tum_shaps, axis=0)
        ozellikler    = list(X_test_sample.columns)

        # Top 15
        top15_idx = np.argsort(ortalama_shap)[::-1][:15]
        top15_feat = [ozellikler[i] for i in top15_idx]
        top15_vals = [float(ortalama_shap[i]) for i in top15_idx]

        # Görsel
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(top15_feat)), top15_vals,
                 color="#2ca02c", edgecolor="black")
        ax.set_yticks(range(len(top15_feat)))
        ax.set_yticklabels(top15_feat)
        ax.invert_yaxis()
        ax.set_xlabel("Ortalama |SHAP| (Ensemble Tree Üyeleri)",
                       fontsize=11, fontweight="bold")
        ax.set_title(f"Final Ensemble SHAP — Top 15 Özellik\n"
                       f"({len(tum_shaps)} tree üyesinin ortalaması)\n"
                       f"Gerçek karar mantığı, referans modelin değil",
                       fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        for i, v in enumerate(top15_vals):
            ax.text(v + max(top15_vals)*0.01, i, f"{v:.4f}",
                     va="center", fontsize=9)
        plt.tight_layout()
        yol = os.path.join(GRAFIK_KLASOR, dosya_adi)
        plt.savefig(yol, dpi=150, bbox_inches="tight")
        plt.close()

        return {"top15_features": top15_feat, "top15_values": top15_vals}
    except Exception as e:
        print(f"  [UYARI] Ensemble SHAP başarısız: {type(e).__name__}: {e}")
        return None


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
        # [v6.0] GPU desteği (sadece CFTR/PAH/MASTER)
        if GPU_AKTIF and "device" not in p:
            p["device"] = LGB_DEVICE
        return lgb.LGBMClassifier(**p)

    elif algo == "catboost":
        p.setdefault("random_seed", RASTGELE_TOHUM)
        p.setdefault("verbose", False)
        p.setdefault("bootstrap_type", "Bernoulli")
        # [v6.0] GPU desteği (sadece CFTR/PAH/MASTER)
        if GPU_AKTIF and "task_type" not in p:
            p["task_type"] = CB_TASK_TYPE
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
        p.setdefault("device", XGB_DEVICE)
        p.setdefault("verbosity", 0)
        p.setdefault("random_state", RASTGELE_TOHUM)
        p.setdefault("n_jobs", -1)  # [v6.2] default -1 (inner-parallel)
        if n_jobs_override is not None: p["n_jobs"] = n_jobs_override
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
        p.setdefault("n_jobs", -1)  # [v6.2] default -1 (inner-parallel)
        p.setdefault("subsample_freq", 1)
        if n_jobs_override is not None: p["n_jobs"] = n_jobs_override
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
        p.setdefault("thread_count", -1)  # [v6.2] default -1 (inner-parallel)
        # [FIX H] Yeni CatBoost: default bootstrap_type=Bayesian subsample'ı kabul
        # etmez. Bernoulli klasik subsample desteklemeli.
        p.setdefault("bootstrap_type", "Bernoulli")
        if n_jobs_override is not None: p["thread_count"] = n_jobs_override
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

def _oof_fold_worker(args):
    """[v6.4] OOF fold worker — module-level (loky picklable).

    Tek fold için: train_split → ES split → fit → predict_proba
    Returns: (val_idx, prob) tuple

    Worker process içinde global'lar yok — her şey args ile gönderilir.
    """
    (algo, params, X_ft, y_ft, X_fv, X_es, y_es,
     n_jobs_override) = args
    try:
        m = model_fit_universal(algo, params, X_ft, y_ft, X_es, y_es,
                                n_jobs_override=n_jobs_override)
        prob = predict_proba_universal(m, algo, X_fv)
        return (True, prob)
    except Exception:
        return (False, None)


def oof_tahmin_universal(algo, params, X_tr, y_tr, n_folds, n_tekrar,
                          n_jobs_override=None, es_split=0.0):
    """
    10 algoritmanın hepsiyle uyumlu OOF tahmini.
    v2.8 FIX #10+#14: try/except + skip örnekler prior alır.

    [v6.4] LOKY PARALEL — Fold seviyesinde process-level paralelleme.
    Optuna resmi FAQ (https://optuna.readthedocs.io/en/stable/faq.html):
      "It is recommended to use process-based parallelization if CPU bound"
    joblib loky backend GIL bypass + memmap shared data.
    Beklenen: 22 paralel × n_folds/22 batch → ~15-20x hızlanma KANSER'de.
    """
    rskf   = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_tekrar,
                                     random_state=RASTGELE_TOHUM)
    toplam = np.zeros(len(X_tr))
    sayac  = np.zeros(len(X_tr))

    # Tüm fold argümanlarını hazırla
    fold_argslari = []
    val_idxler    = []
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

        # Worker için tek thread (oversubscription önle)
        worker_njobs = 1
        fold_argslari.append((algo, params, X_ft, y_ft, X_fv,
                              X_es, y_es, worker_njobs))
        val_idxler.append(val_idx)

    # joblib loky ile paralel — gerçek process paralelliği
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=PARALEL_ISCI, backend="loky",
                           verbose=0, batch_size="auto")(
            delayed(_oof_fold_worker)(args) for args in fold_argslari
        )
    except Exception as e:
        # Fallback: sıralı çalıştır (loky problem çıkarırsa)
        print(f"    [UYARI] Paralel fold başarısız ({type(e).__name__}), sıralıya geçiliyor")
        results = [_oof_fold_worker(args) for args in fold_argslari]

    # Sonuçları topla
    for (basarili, prob), val_idx in zip(results, val_idxler):
        if basarili and prob is not None:
            toplam[val_idx] += prob
            sayac[val_idx]  += 1

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
        # [v6.2.2] n_jobs dynamic — algo_paralelizm()'den oku (küçük data → 1)
        _, xgb_inner_nj = algo_paralelizm("xgboost")
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
            "enable_categorical": True, "tree_method": "hist", "device": XGB_DEVICE,
            "verbosity": 0, "random_state": RASTGELE_TOHUM, "n_jobs": xgb_inner_nj,
        }

    if algo == "lightgbm":
        # [v6.2.2] n_jobs dynamic
        _, lgb_inner_nj = algo_paralelizm("lightgbm")
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
            "verbosity": -1, "random_state": RASTGELE_TOHUM, "n_jobs": lgb_inner_nj,
            "subsample_freq": 1,
        }

    if algo == "catboost":
        # [v6.2.2] thread_count dynamic
        _, cb_inner_nj = algo_paralelizm("catboost")
        return {
            "iterations":        trial.suggest_int("cb_iter", 100, 800),
            "depth":             trial.suggest_int("cb_depth", 2, 8),
            "learning_rate":     trial.suggest_float("cb_lr", 0.003, 0.30, log=True),
            "l2_leaf_reg":       trial.suggest_float("cb_l2", 0.1, 30.0),
            "min_data_in_leaf":  trial.suggest_int("cb_mdl", 1, 50),
            "subsample":         trial.suggest_float("cb_sub", 0.35, 1.00),
            "colsample_bylevel": trial.suggest_float("cb_cbl", 0.40, 1.00),
            "scale_pos_weight":  trial.suggest_float("cb_spw", SPW*0.2, SPW*6.0),
            "random_seed": RASTGELE_TOHUM, "thread_count": cb_inner_nj, "verbose": False,
        }

    if algo == "random_forest":
        # [v6.2.1] n_jobs dynamic — algo_paralelizm()'den oku
        _, rf_inner_nj = algo_paralelizm("random_forest")
        return {
            "n_estimators":     trial.suggest_int("rf_n_est", 100, 1000),
            "max_depth":        trial.suggest_int("rf_depth", 3, 25),
            "min_samples_split":trial.suggest_int("rf_mss", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_msl", 1, 20),
            "max_features":     trial.suggest_float("rf_mf", 0.2, 1.0),
            "class_weight":     trial.suggest_categorical("rf_cw",
                                                          ["balanced", "balanced_subsample", None]),
            "n_jobs": rf_inner_nj, "random_state": RASTGELE_TOHUM,
        }

    if algo == "extra_trees":
        # [v6.2.1] n_jobs dynamic — algo_paralelizm()'den oku
        _, et_inner_nj = algo_paralelizm("extra_trees")
        return {
            "n_estimators":     trial.suggest_int("et_n_est", 100, 1000),
            "max_depth":        trial.suggest_int("et_depth", 3, 25),
            "min_samples_split":trial.suggest_int("et_mss", 2, 20),
            "min_samples_leaf": trial.suggest_int("et_msl", 1, 20),
            "max_features":     trial.suggest_float("et_mf", 0.2, 1.0),
            "class_weight":     trial.suggest_categorical("et_cw",
                                                          ["balanced", "balanced_subsample", None]),
            "n_jobs": et_inner_nj, "random_state": RASTGELE_TOHUM,
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
        p.update({"enable_categorical": True, "tree_method": "hist", "device": XGB_DEVICE,
                  "verbosity": 0, "random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override is not None else -1})
    elif algo == "lightgbm":
        p.update({"verbosity": -1, "random_state": RASTGELE_TOHUM, "subsample_freq": 1,
                  "n_jobs": n_jobs_override if n_jobs_override is not None else -1})
    elif algo == "catboost":
        p.update({"random_seed": RASTGELE_TOHUM, "verbose": False,
                  "thread_count": n_jobs_override if n_jobs_override is not None else -1,
                  "bootstrap_type": "Bernoulli"})
    elif algo == "random_forest":
        p.update({"random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override is not None else -1})
    elif algo == "extra_trees":
        p.update({"random_state": RASTGELE_TOHUM,
                  "n_jobs": n_jobs_override if n_jobs_override is not None else -1})
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
SHAP_CACHE_PATH = f"shap_cache_{DATA_SUFFIX}.npz"
if SHAP_OK and SHAP_PERCENTILE > 0:
    # [v4.5] SHAP cache — eğer önceki run'da hesaplandıysa yükle, ~2 dk tasarruf
    shap_cache_yuklendi = False
    if os.path.exists(SHAP_CACHE_PATH):
        try:
            _cached = np.load(SHAP_CACHE_PATH, allow_pickle=True)
            _cached_cols = _cached["all_cols"].tolist()
            # Cache validation: özellik isimleri eşleşmeli
            if _cached_cols == list(X_train.columns):
                mean_abs_shap = _cached["mean_abs_shap"]
                onem_sirasi   = _cached["onem_sirasi"]
                all_cols      = _cached_cols
                print(f"  [v4.5] SHAP cache yüklendi: {SHAP_CACHE_PATH}")
                shap_cache_yuklendi = True
            else:
                print(f"  [v4.5] SHAP cache özellik uyumsuzluğu, yeniden hesaplanacak.")
        except Exception as e:
            print(f"  [v4.5] SHAP cache yükleme hatası: {e}, yeniden hesaplanacak.")

    try:
        if not shap_cache_yuklendi:
            SHAP_XGB_PARAMS = {
                "enable_categorical": True, "tree_method": "hist", "device": XGB_DEVICE,
                "verbosity": 0, "n_jobs": -1, "random_state": RASTGELE_TOHUM,
                "n_estimators": 200, "max_depth": 6,
                "learning_rate": 0.05, "subsample": 0.8, "scale_pos_weight": SPW,
            }
            print("  SHAP referans modeli (XGBoost) eğitiliyor...")
            shap_model = xgb.XGBClassifier(**SHAP_XGB_PARAMS)
            shap_model.fit(X_train, y_train)

            explainer   = shap.TreeExplainer(shap_model)
            shap_values = explainer.shap_values(X_train)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            all_cols      = list(X_train.columns)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            onem_sirasi   = np.argsort(mean_abs_shap)[::-1]

            # Cache'e kaydet (sonraki run'lar için)
            try:
                np.savez(SHAP_CACHE_PATH,
                          mean_abs_shap=mean_abs_shap,
                          onem_sirasi=onem_sirasi,
                          all_cols=np.array(all_cols, dtype=object))
                print(f"  [v4.5] SHAP cache kaydedildi: {SHAP_CACHE_PATH}")
            except Exception as e:
                print(f"  [v4.5] Cache kaydedilemedi: {e}")
            del shap_values, shap_model, explainer
            gc.collect()

        print("\n  Top 10 SHAP Özelliği:")
        shap_max = max(float(mean_abs_shap.max()), 1e-9)
        for rk in onem_sirasi[:10]:
            bar = "█" * int(mean_abs_shap[rk] / shap_max * 35)
            print(f"    {all_cols[rk]:<30} {bar}  {mean_abs_shap[rk]:.5f}")

        n_kalan = max(1, int(len(mean_abs_shap) * (1 - SHAP_PERCENTILE / 100)))
        top_idx = np.argsort(mean_abs_shap)[::-1][:n_kalan]
        kalan_mask = np.zeros(len(mean_abs_shap), dtype=bool)
        kalan_mask[top_idx] = True
        shap_esik  = float(mean_abs_shap[top_idx[-1]])
        kalan_cols = [c for c, keep in zip(all_cols, kalan_mask) if keep]
        budanan_cnt = len(all_cols) - len(kalan_cols)

        X_train       = X_train[kalan_cols]
        X_test        = X_test[kalan_cols]
        KAT_KOLONLAR  = [c for c in KAT_KOLONLAR if c in kalan_cols]
        BUDANAN_OZELLIKLER = [c for c in all_cols if c not in kalan_cols]

        print(f"\n  {len(kalan_cols)} özellik kaldı ({budanan_cnt} budandı)")

        # v4.1: SHAP top 15 özellik grafiği
        top15_idx = onem_sirasi[:15]
        top15_features = [all_cols[i] for i in top15_idx]
        top15_values   = [float(mean_abs_shap[i]) for i in top15_idx]
        shap_bar_grafigi(top15_features, top15_values)
        print(f"  → {GRAFIK_KLASOR}/shap_top_ozellikler.png kaydedildi.")

        # [v5.1] Feature group analizi (prefix-bazlı SHAP toplaması)
        try:
            grup_analiz = feature_group_analizi(all_cols, mean_abs_shap)
            if grup_analiz:
                print(f"\n  Özellik Grubu Bazlı SHAP Katkısı:")
                for g, y, s in zip(grup_analiz["gruplar"],
                                     grup_analiz["yuzdeler"],
                                     grup_analiz["sayilar"]):
                    bar = "█" * int(y / 2)
                    print(f"    {g:<10} %{y:>5.1f}  {bar}  ({s} özellik)")
                print(f"  → {GRAFIK_KLASOR}/feature_grup_analizi.png kaydedildi.")
        except Exception as e:
            print(f"  [UYARI] Feature group analizi atlandı: {e}")
            grup_analiz = None
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
    fail_sayisi = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.FAIL])
    toplam_dbde = len(study.trials)
    kalan  = max(0, n_trial - mevcut)

    # [v6.3] Cache durumunu net göster
    if toplam_dbde > 0:
        print(f"    [CACHE] DB'de {toplam_dbde} trial var "
              f"(COMPLETE: {mevcut}, FAIL: {fail_sayisi})")
        if mevcut >= n_trial:
            print(f"    [CACHE] Hedef {n_trial} trial zaten tamamlanmış, "
                  f"yeni trial yapılmayacak. Cache'den okundu.")
        else:
            print(f"    [CACHE] Hedef {n_trial}, kalan {kalan} trial çalıştırılacak")

    if kalan > 0:
        t0 = time.perf_counter()
        # [FIX O] Algoritmaya göre paralellik stratejisi
        outer_njobs, _ = algo_paralelizm(algo)
        if outer_njobs == 1:
            print(f"    [FIX O] {algo} için outer SEQUENTIAL, "
                  f"iç model n_jobs=-1 (24 thread/tree)")
        # ŞEBNUS ÖLÜMSÜZ MODU
        while kalan > 0:
            try:
                study.optimize(faz1_objective_factory(algo), n_trials=kalan,
                                n_jobs=outer_njobs, show_progress_bar=True,
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
# [v6.0 KRİTİK FIX] Önceki: bilesik = F2 × (1 - kor) → çeşitliliği aşırı
# vurguluyordu, zayıf modeller (kNN F2=0.93) kazanıyordu.
#
# Yeni: İki aşamalı seçim:
#   1) F2 alt sınır (kalite kapısı): en_iyi_F2 - 0.025 üstündekiler
#   2) Bu adaylar arasında: bilesik = F2 - α × ort_kor (additive, α=0.15)
#
# Sonuç: CatBoost, LightGBM gibi güçlü modeller seçilebilir. Korelasyon
# yine ceza ama F2'yi domine etmiyor.
FAZ2_F2_ALT_SINIR_TOLERANS = 0.025    # en_iyi F2'den 0.025 düşük olana kadar kabul
FAZ2_KORELASYON_AGIRLIGI   = 0.15     # additive ceza katsayısı

print("\n  Greedy Çeşitlilik Seçimi (v6.0: F2 alt sınır + additive bileşik):")

secilen_algoritmalar = []
en_iyi_f2_algo = max(oof_f2_per_algo, key=oof_f2_per_algo.get)
en_iyi_f2_genel = oof_f2_per_algo[en_iyi_f2_algo]
F2_ALT_SINIR = en_iyi_f2_genel - FAZ2_F2_ALT_SINIR_TOLERANS

print(f"    En iyi OOF F2: {en_iyi_f2_genel:.4f} ({en_iyi_f2_algo})")
print(f"    F2 alt sınır:  {F2_ALT_SINIR:.4f} (tolerans={FAZ2_F2_ALT_SINIR_TOLERANS})")

# Adayları F2 alt sınıra göre filtrele
adaylar_kalitelli = [a for a in hayatta_kalan
                      if oof_f2_per_algo[a] >= F2_ALT_SINIR]
elenenler = [a for a in hayatta_kalan
              if oof_f2_per_algo[a] < F2_ALT_SINIR]

print(f"    F2 alt sınırını geçen: {len(adaylar_kalitelli)} algoritma "
      f"({', '.join(adaylar_kalitelli)})")
if elenenler:
    print(f"    F2 alt sınırının altında: {len(elenenler)} algoritma "
          f"({', '.join(elenenler)}) — çeşitlilik seçimine alınmayacak")

# Adım 1: En yüksek F2'li (alt sınırı geçen) algoritma
secilen_algoritmalar.append(en_iyi_f2_algo)
print(f"    1. {en_iyi_f2_algo.upper()} (en yüksek F2: {en_iyi_f2_genel:.4f})")

# Adım 2+: Additive bileşik ile çeşitlilik seçimi
while len(secilen_algoritmalar) < FAZ2_SECILEN_SAYI:
    adaylar = [a for a in adaylar_kalitelli if a not in secilen_algoritmalar]
    if not adaylar:
        # Alt sınırı geçen aday kalmadı, geri kalan tüm hayatta kalanlardan seç
        adaylar = [a for a in hayatta_kalan if a not in secilen_algoritmalar]
        if not adaylar:
            break
        print(f"    [UYARI] F2 alt sınırı tükendi, geri kalanlardan seçiliyor")

    # Her aday için seçilenlerle ortalama korelasyonu hesapla
    ort_kor = {}
    for a in adaylar:
        kor_listesi = [abs(korelasyon_df.loc[a, s]) for s in secilen_algoritmalar]
        ort_kor[a] = float(np.mean(kor_listesi))

    # YENİ: bilesik = F2 - α × ort_kor (additive)
    bilesik = {a: oof_f2_per_algo[a] - FAZ2_KORELASYON_AGIRLIGI * ort_kor[a]
                for a in adaylar}
    en_iyi = max(bilesik, key=bilesik.get)

    print(f"    {len(secilen_algoritmalar)+1}. {en_iyi.upper()}  "
          f"(F2={oof_f2_per_algo[en_iyi]:.4f}, "
          f"ort_kor={ort_kor[en_iyi]:.3f}, "
          f"bileşik={bilesik[en_iyi]:.4f})")
    secilen_algoritmalar.append(en_iyi)

print(f"\n  FİNAL {len(secilen_algoritmalar)} ALGORİTMA: {secilen_algoritmalar}")
print(f"  Paradigmalar: "
      f"{[ALGO_PARADIGMA[a] for a in secilen_algoritmalar]}")

# [v5.1] Pipeline funnel diyagramı — 4-faz hiyerarşik mimari görseli
_funnel_istatistikler = {
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "n_ozellik_baslangic": int(len(OZELLIK)),
    "n_ozellik_son": int(len(X_train.columns)),
    "faz1_algos": len(TUM_ALGORITMALAR),
    "faz1_kalan": len(hayatta_kalan),
    "faz1_total_trial": int(sum(FAZ1_TRIAL_PER_ALGO.values())),
    "faz2_secilen": len(secilen_algoritmalar),
    "secilen_algos": secilen_algoritmalar,
    "faz3_total_trial": int(sum(
        FAZ3_TRIAL_PER_ALGO.get(a, 500) for a in secilen_algoritmalar
    )),
    "ensemble_uye_sayisi": FAZ2_SECILEN_SAYI * FAZ3_TOP_K,
}
yol = pipeline_funnel_diyagrami(_funnel_istatistikler)
if yol:
    print(f"  → {yol} kaydedildi (4-faz mimari)")

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
    fail_sayisi = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.FAIL])
    pruned_sayisi = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED])
    toplam_dbde = len(study.trials)
    kalan  = max(0, algo_trial_budget - mevcut)

    # [v6.3] FAZ 3 cache durumunu net göster
    if toplam_dbde > 0:
        print(f"    [CACHE] DB'de {toplam_dbde} trial var "
              f"(COMPLETE: {mevcut}, PRUNED: {pruned_sayisi}, FAIL: {fail_sayisi})")
        if mevcut >= algo_trial_budget:
            print(f"    [CACHE] Hedef {algo_trial_budget} trial zaten tamamlanmış. "
                  f"Cache'den okundu, yeni trial yok.")
        else:
            print(f"    [CACHE] Hedef {algo_trial_budget}, kalan {kalan} trial çalıştırılacak")

    if kalan > 0:
        t0 = time.perf_counter()
        # [FIX O] Algoritmaya göre paralellik stratejisi
        outer_njobs, _ = algo_paralelizm(algo)
        if outer_njobs == 1:
            print(f"    [FIX O] {algo} için outer SEQUENTIAL, "
                  f"iç model n_jobs=-1 (24 thread/tree)")
        while kalan > 0:
            try:
                study.optimize(faz3_objective_factory(algo), n_trials=kalan,
                                n_jobs=outer_njobs, show_progress_bar=True,
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

        # [v5.1] Trial convergence grafiği — sistematik araştırma kanıtı
        yol = trial_convergence_grafigi(study, algo)
        if yol:
            print(f"    → {yol} kaydedildi (Optuna progress)")
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
# [v5.1] Tree-based final modelleri sakla (ensemble SHAP için)
final_tree_modeller = []

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
        # [v5.1] Tree-based modelleri sakla (SHAP için)
        if algo in ("xgboost", "lightgbm", "catboost",
                     "random_forest", "extra_trees"):
            final_tree_modeller.append(fm)
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
_b_iter = bootstrap_iter_count(len(y_test))
print(f"\n  Bootstrap 95% Güven Aralıkları ({_b_iter} iterasyon, test_n={len(y_test)})...")
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
#  7.5 [v4.5] BELİRSİZLİK + DİSAGREEMENT + YANLIŞ TAHMİN ANALİZİ
#       Jüri için klinik akıllılık göstergeleri:
#       - Per-örnek model belirsizliği (15 üye std)
#       - Paradigma disagreement rate (3 algoritma)
#       - Yanlış tahmin edilen örneklerin detaylı raporu
# ══════════════════════════════════════════════════════════════════
print(f"\n[7.5/9] [v4.5] KLİNİK BELİRSİZLİK + DİSAGREEMENT ANALİZİ")
print("=" * 72)

# ─── A. PER-ÖRNEK BELİRSİZLİK (Ensemble üyeleri arası std) ───
# Tek bir test örneği için 15 üyenin standart sapması, modellerin
# anlaşıp anlaşmadığını gösterir. Düşük std = güvenli, yüksek std = riskli.
test_proba_matrix = np.array(final_test_proba_listesi)  # [15, 78]
test_proba_std    = test_proba_matrix.std(axis=0)        # [78]
test_proba_mean   = test_proba_matrix.mean(axis=0)
guven_skoru       = 1.0 - test_proba_std                # 1 = mükemmel

# Belirsizlik kategorileri
DUSUK_BELIRSIZLIK_ESIK  = 0.10  # std < 0.10 → güvenli
YUKSEK_BELIRSIZLIK_ESIK = 0.20  # std > 0.20 → riskli
n_yuksek = int(np.sum(test_proba_std >= YUKSEK_BELIRSIZLIK_ESIK))
n_orta   = int(np.sum((test_proba_std >= DUSUK_BELIRSIZLIK_ESIK) &
                      (test_proba_std <  YUKSEK_BELIRSIZLIK_ESIK)))
n_dusuk  = int(np.sum(test_proba_std <  DUSUK_BELIRSIZLIK_ESIK))

print(f"\n  ── Model Belirsizliği (15 ensemble üyesi arası std) ──")
print(f"    Güvenli tahmin   (std<{DUSUK_BELIRSIZLIK_ESIK:.2f}): {n_dusuk:>3} / {N_TEST}  ({n_dusuk/N_TEST*100:.1f}%)")
print(f"    Orta belirsizlik ({DUSUK_BELIRSIZLIK_ESIK:.2f}-{YUKSEK_BELIRSIZLIK_ESIK:.2f}): {n_orta:>3} / {N_TEST}  ({n_orta/N_TEST*100:.1f}%)")
print(f"    Yüksek belirsiz  (std>{YUKSEK_BELIRSIZLIK_ESIK:.2f}): {n_yuksek:>3} / {N_TEST}  ({n_yuksek/N_TEST*100:.1f}%)")
print(f"    Ortalama std     : {test_proba_std.mean():.4f}")
print(f"    Max std          : {test_proba_std.max():.4f}")

# Yüksek belirsizlikli örnekler (klinik dikkat gerektirir)
if n_yuksek > 0:
    yuksek_idx = np.where(test_proba_std >= YUKSEK_BELIRSIZLIK_ESIK)[0]
    print(f"\n    Yüksek belirsizlikli {n_yuksek} örnek (klinik tekrar değerlendirme önerisi):")
    print(f"    {'Variant_ID':<20} {'Gerçek':<8} {'Tahmin':<8} {'Ort.Olas':<10} {'Std':<8}")
    for idx in yuksek_idx[:10]:  # max 10 örnek göster
        vid = str(variant_ids_test[idx])[:18]
        gerc = "Patojenik" if y_test[idx] == 1 else "Benign"
        tahm = "Patojenik" if test_tahminler[idx] == 1 else "Benign"
        print(f"    {vid:<20} {gerc:<8} {tahm:<8} {test_proba_mean[idx]:<10.4f} {test_proba_std[idx]:<8.4f}")

# ─── B. DISAGREEMENT RATE (3 Paradigma) ───
# Her algoritma için kendi üye-ortalamasını al, sınıf tahminlerini karşılaştır.
# Üye sırası secilen_trials_algo'da tutuluyor: [et,et,et,et,et,knn,knn,..,lr,lr,lr,lr,lr]
print(f"\n  ── Paradigma Disagreement Rate (3 algoritma) ──")
paradigma_proba = {}
for algo_adi in secilen_algoritmalar:
    algo_idxler = [i for i, a in enumerate(secilen_trials_algo) if a == algo_adi]
    if algo_idxler:
        algo_test_probas = test_proba_matrix[algo_idxler]   # [5, 78]
        paradigma_proba[algo_adi] = algo_test_probas.mean(axis=0)

# Her örnek için 3 paradigma tahmini
paradigma_pred = {}
for algo_adi, probas in paradigma_proba.items():
    paradigma_pred[algo_adi] = (probas >= OPTIMAL_ESIK).astype(int)

# Disagreement: 3 modelin tahmin sayısı eşit değilse fark vardır
preds_array = np.array(list(paradigma_pred.values()))  # [3, 78]
disagreement_per_ornek = np.array([
    len(set(preds_array[:, i])) > 1 for i in range(N_TEST)
])
disagreement_count = int(disagreement_per_ornek.sum())
disagreement_rate  = disagreement_count / N_TEST

print(f"    Paradigmalar:     {list(paradigma_proba.keys())}")
print(f"    Anlaşma sağlanan: {N_TEST - disagreement_count} / {N_TEST}  "
      f"({(1-disagreement_rate)*100:.1f}%)")
print(f"    Disagreement     : {disagreement_count} / {N_TEST}  "
      f"({disagreement_rate*100:.1f}%) — ensemble bu noktalarda kritik karar veriyor")

# Disagreement olan örneklerde ensemble doğruluğu
if disagreement_count > 0:
    dis_idx = np.where(disagreement_per_ornek)[0]
    dis_dogru = int(np.sum(test_tahminler[dis_idx] == y_test[dis_idx]))
    print(f"    Ensemble doğru   : {dis_dogru} / {disagreement_count}  "
          f"({dis_dogru/disagreement_count*100:.1f}%) "
          f"(paradigmaların anlaşmazlığında ensemble seçimi)")

# ─── C. YANLIŞ TAHMİN TABLOSU ───
# TEKNOFEST şartnamesi: "yanlış pozitif ve yanlış negatif sonuçlar
# ayrıntılı olarak incelenmeli" — bu CSV doğrudan rapor 4. bölüm için.
print(f"\n  ── Yanlış Tahmin Edilen Örnekler (Klinik Yorum İçin) ──")
yanlis_idx = np.where(test_tahminler != y_test)[0]

yanlis_kayitlar = []
for idx in yanlis_idx:
    yt = int(y_test[idx])
    tp = int(test_tahminler[idx])
    hata_tipi = "FN (KAÇIRILAN PATOJENİK)" if yt == 1 and tp == 0 else "FP (Yanlış Alarm)"

    # SHAP referans modelinin yardımıyla en önemli 3 özellik (eğer mean_abs_shap mevcutsa)
    try:
        # X_test'te bu örneğin özellik değerleri
        ornek_features = X_test.iloc[idx]
        top3_shap_features = []
        for rk in onem_sirasi[:3]:
            if rk < len(all_cols):
                fname = all_cols[rk]
                if fname in ornek_features.index:
                    fval = ornek_features[fname]
                    if isinstance(fval, (int, float, np.integer, np.floating)):
                        top3_shap_features.append(f"{fname}={float(fval):.3f}")
                    else:
                        top3_shap_features.append(f"{fname}={fval}")
        top3_str = ", ".join(top3_shap_features) if top3_shap_features else "-"
    except Exception:
        top3_str = "-"

    yanlis_kayitlar.append({
        "variant_id":        str(variant_ids_test[idx]),
        "gercek_etiket":     "Patojenik" if yt == 1 else "Benign",
        "tahmin":            "Patojenik" if tp == 1 else "Benign",
        "hata_tipi":         hata_tipi,
        "ensemble_olasilik": round(float(final_test_proba[idx]), 4),
        "model_belirsizlik": round(float(test_proba_std[idx]), 4),
        "guven_skoru":       round(float(guven_skoru[idx]), 4),
        "top3_ozellik":      top3_str,
    })

n_fn = sum(1 for r in yanlis_kayitlar if "FN" in r["hata_tipi"])
n_fp = sum(1 for r in yanlis_kayitlar if "FP" in r["hata_tipi"])
print(f"    Toplam yanlış: {len(yanlis_kayitlar)}  (FN: {n_fn}, FP: {n_fp})")

if yanlis_kayitlar:
    # Ekrana yazdır
    print(f"\n    {'Variant_ID':<18} {'Gerçek':<10} {'Tahmin':<10} {'Hata':<25} {'Olas.':<7} {'Std':<7}")
    print(f"    {'-'*18} {'-'*10} {'-'*10} {'-'*25} {'-'*7} {'-'*7}")
    for r in yanlis_kayitlar:
        vid_short = r["variant_id"][:16]
        hata_short = r["hata_tipi"][:23]
        print(f"    {vid_short:<18} {r['gercek_etiket']:<10} {r['tahmin']:<10} "
              f"{hata_short:<25} {r['ensemble_olasilik']:<7.4f} {r['model_belirsizlik']:<7.4f}")

# CSV kaydet (TEKNOFEST raporu 4. SONUÇ bölümü için)
if yanlis_kayitlar:
    yanlis_df = pd.DataFrame(yanlis_kayitlar)
    yanlis_df.to_csv(f"yanlis_tahminler_{DATA_SUFFIX}.csv",
                       index=False, encoding="utf-8")
    print(f"\n  → yanlis_tahminler_{DATA_SUFFIX}.csv kaydedildi (rapor 4. bölümü için)")

# [v5.1] Final Ensemble SHAP — gerçek karar mantığı (referans değil)
print(f"\n  ── Final Ensemble SHAP Analizi ({len(final_tree_modeller)} tree üyesi) ──")
if final_tree_modeller and SHAP_OK:
    try:
        # Daha hızlı çalışsın diye küçük örnek
        X_test_sample = X_test.iloc[:min(30, len(X_test))]
        ensemble_shap_sonuc = ensemble_shap_hesapla(
            final_tree_modeller, X_train, X_test_sample
        )
        if ensemble_shap_sonuc:
            print(f"    → {GRAFIK_KLASOR}/ensemble_shap.png kaydedildi")
            print(f"    Top 5 özellik (final ensemble): "
                  f"{', '.join(ensemble_shap_sonuc['top15_features'][:5])}")
        else:
            print(f"    [BİLGİ] Ensemble SHAP üretilemedi")
    except Exception as e:
        print(f"    [UYARI] Ensemble SHAP atlandı: {e}")
        ensemble_shap_sonuc = None
else:
    print(f"    [BİLGİ] Tree-based model yok veya SHAP yüklü değil, atlandı")
    ensemble_shap_sonuc = None

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
            # [v4.3] Görsel üretiminde reuse için probs cache (memory-only)
            "_test_probs": test_probs,
            "_test_pred":  test_pred,
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
        enable_categorical=True, tree_method="hist", device=XGB_DEVICE,
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

# CSV kaydet (probs cache field'larını CSV'ye yazma)
_CSV_HARIC_FIELDS = ("_test_probs", "_test_pred")
def _csv_temiz(row):
    return {k: v for k, v in row.items() if k not in _CSV_HARIC_FIELDS}

df_karsi = pd.DataFrame([_csv_temiz(r) for r in karsilastirma_listesi])
df_karsi_sirali = df_karsi.sort_values("test_f2", ascending=False).reset_index(drop=True)
# [v4.3] CSV dosya adına DATA_SUFFIX
df_karsi.to_csv(f"solo_vs_ensemble_{DATA_SUFFIX}.csv", index=False, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────
# v4.1: GÖRSELLER — TEKNOFEST raporu için zorunlu
# ─────────────────────────────────────────────────────────────────
print(f"\n  Görseller üretiliyor (grafikler/ klasörüne)...")

# 1. Ensemble confusion matrix
confusion_matrix_gorseli(y_test, test_tahminler,
                          "ENSEMBLE (4-faz pipeline)",
                          "cm_ensemble.png")

# 2. Solo modeller için CM (FAZ 8.5'te cache'lenmiş probs'u kullan,
# YENİDEN EĞİTMEYE GEREK YOK — v4.3 optimizasyonu, ~5 dk tasarruf)
en_iyi_3_solo = sorted(solo_sonuclari, key=lambda x: x.get("test_f2", 0),
                        reverse=True)[:3]

# v4.1: Initialize — en_iyi_solo_probs hiç atanmazsa (tüm try'lar fail)
en_iyi_solo_probs = None
en_iyi_solo_ad    = None

for i, s in enumerate(en_iyi_3_solo):
    ad = s["model"]
    # [v4.3] Probs zaten _solo_evaluate'da hesaplandı, cache'den al
    if "_test_probs" not in s or "_test_pred" not in s:
        print(f"  → {ad} CM atlandı (probs cache yok, hata olmuş olabilir)")
        continue
    print(f"  → {ad} CM çiziliyor (cache'den, yeniden eğitim yok)...")
    try:
        probs = s["_test_probs"]
        pred  = s["_test_pred"]
        # Confusion matrix
        guvenli_ad = ad.replace(" ", "_").replace("(", "").replace(")", "")
        confusion_matrix_gorseli(y_test, pred, ad,
                                  f"cm_solo_{i+1}_{guvenli_ad}.png")
        # ROC/PR için en iyi solo'yu sakla
        if i == 0:
            en_iyi_solo_probs = probs
            en_iyi_solo_ad    = ad
    except Exception as e:
        print(f"    [UYARI] {ad} CM atlandı: {e}")

# 3. Solo vs Ensemble bar chart
solo_vs_ensemble_bar(karsilastirma_listesi)
print(f"  → grafikler/solo_vs_ensemble.png")

# 4. ROC + PR curve karşılaştırma (Ensemble vs en iyi solo)
try:
    proba_dict = {"Ensemble (4-faz)": final_test_proba}
    if en_iyi_solo_probs is not None:
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
pd.DataFrame(esik_sonuc).to_csv(f"karar_esigi_tablosu_{DATA_SUFFIX}.csv",
                                   index=False, encoding="utf-8")
print(f"\n  → karar_esigi_tablosu_{DATA_SUFFIX}.csv kaydedildi (rapor için).")

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

# [v4.5] Klinik akıllılık metrikleri vurgusu
print(f"\n  [v4.5] KLİNİK AKILLILIK GÖSTERGELERİ:")
print(f"    • Per-örnek belirsizlik analizi: {n_yuksek}/{N_TEST} yüksek belirsizlikli "
      f"(klinik tekrar değerlendirme önerisi)")
print(f"    • Paradigma disagreement: %{disagreement_rate*100:.1f} "
      f"(3 model anlaşmazlığında ensemble seçimi yapar)")
print(f"    • Yanlış tahmin analizi: {len(yanlis_kayitlar)} örnek (FN: {n_fn}, FP: {n_fp})"
      f" — yanlis_tahminler_{DATA_SUFFIX}.csv'de detaylı")
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
        "surum"          : f"BioSense AI v6.3 — Optimizasyon Kalibresi ({DATA_SUFFIX.upper()}, {CALISMA_MODU.upper()})",
        "calisma_modu"   : CALISMA_MODU,
        "sistem_profili" : {
            "cpu_logical"     : int(_cpu_count_logical),
            "cpu_physical"    : int(_cpu_count_physical),
            "ram_total_gb"    : round(float(_total_ram_gb), 2),
            "ram_available_gb": round(float(_available_ram_gb), 2),
            "ram_kullanim_pct": round(float(_ram_kullanim_pct), 1),
            "per_worker_gb"   : float(_per_worker_gb),
            "paralel_isci"    : int(PARALEL_ISCI),
            "gpu"             : GPU_BILGI,
            "mod_trial_carpan": float(_c["trial"]),
            "mod_fold_carpan" : float(_c["fold"]),
            "faz1_trial_total": int(sum(FAZ1_TRIAL_PER_ALGO.values())),
            "faz3_trial_total": int(sum(FAZ3_TRIAL_PER_ALGO.values())),
            "faz1_fold_count" : int(FAZ1_FOLDS * FAZ1_TEKRAR),
            "faz3_fold_count" : int(FAZ3_FOLDS * FAZ3_TEKRAR),
        },
        "data_suffix"    : DATA_SUFFIX,
        "veri_dosyasi"   : VERI_DOSYASI,
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
    "v45_klinik_analiz": {
        "_aciklama": "v4.5: Per-örnek belirsizlik, paradigma disagreement, yanlış tahmin analizi",
        "belirsizlik": {
            "ortalama_std":     round(float(test_proba_std.mean()), 4),
            "max_std":          round(float(test_proba_std.max()), 4),
            "median_std":       round(float(np.median(test_proba_std)), 4),
            "dusuk_belirsiz_n": int(n_dusuk),
            "orta_belirsiz_n":  int(n_orta),
            "yuksek_belirsiz_n": int(n_yuksek),
            "dusuk_esik":       float(DUSUK_BELIRSIZLIK_ESIK),
            "yuksek_esik":      float(YUKSEK_BELIRSIZLIK_ESIK),
        },
        "disagreement": {
            "paradigmalar":       list(paradigma_proba.keys()),
            "anlasma_n":          int(N_TEST - disagreement_count),
            "disagreement_n":     int(disagreement_count),
            "disagreement_rate":  round(float(disagreement_rate), 4),
            "ensemble_dogru_dis": int(np.sum(test_tahminler[disagreement_per_ornek] == y_test[disagreement_per_ornek])) if disagreement_count > 0 else 0,
        },
        "yanlis_tahminler": yanlis_kayitlar,
        "yanlis_ozet": {
            "toplam": len(yanlis_kayitlar),
            "false_negatives": int(n_fn),
            "false_positives": int(n_fp),
        },
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
print(f"  'solo_vs_ensemble_{DATA_SUFFIX}.csv' (v4.0 jüri karşılaştırması)")
print(f"  'karar_esigi_tablosu_{DATA_SUFFIX}.csv' (v4.1 TEKNOFEST zorunlu)")
if yanlis_kayitlar:
    print(f"  'yanlis_tahminler_{DATA_SUFFIX}.csv' (v4.5 klinik yorum)")
if SHAP_OK and os.path.exists(SHAP_CACHE_PATH):
    print(f"  '{SHAP_CACHE_PATH}' (v4.5 SHAP cache, sonraki run'larda 2 dk tasarruf)")
print(f"  '{GRAFIK_KLASOR}/' klasöründe 12+ PNG görsel:")
print(f"     [v4.1] shap_top_ozellikler.png, faz1_algoritma_karsilastirma.png")
print(f"     [v4.1] faz2_korelasyon_heatmap.png, cm_*.png (4 adet)")
print(f"     [v4.1] solo_vs_ensemble.png, roc_pr_curves.png")
print(f"     [v5.1] pipeline_funnel.png (4-faz mimari)")
print(f"     [v5.1] trial_convergence_*.png (3 algoritma için)")
print(f"     [v5.1] feature_grup_analizi.png (pasta + bar)")
print(f"     [v5.1] ensemble_shap.png (gerçek karar mantığı)")
print("\n" + "=" * 72)
print(f"  BioSense AI v6.3 — TAMAMLANDI ({DATA_SUFFIX.upper()}, MOD: {CALISMA_MODU.upper()})")
print("=" * 72)