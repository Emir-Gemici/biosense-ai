# Kullanım

## Ortam

Python 3.13 ve `requirements.txt` kullanılır. Bağımlılıklar hazırlık ortamında görülen sürümlere sabitlenmiştir; özgün Mayıs 2026 eğitim ortamının kilit dosyası değildir. Yeni bir temiz ortamda tam eğitim doğrulaması yapılmamıştır.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git config core.hooksPath .githooks
```

Linux/macOS'ta etkinleştirme komutu `source .venv/bin/activate` şeklindedir. CUDA varsayılan değildir; GPU kullanımı ilgili kütüphane ve sürücü desteğine bağlıdır. Başlatıcı varsayılan olarak CPU kullanır.

## Sentetik deneme

```sh
python scripts/make_demo_data.py
python scripts/smoke_demo.py
python -m streamlit run apps/evaluate.py
```

Üreteç yalnızca Python standart kütüphanesindeki sabit tohumlu rastgele sayı üretecini kullanır. 240 eğitim ve 80 test satırı oluşturur. Gerçek biyolojik dağılımları taklit etme iddiası yoktur. Eğitim ve test ID'leri ayrıdır; tekrar çalıştırıldığında mevcut dosyaların üzerine yazmaz. Başka bir klasör için `--output-dir data/demo-2` kullanılabilir.

Arayüzde TRAIN ve TEST CSV'leri seçilir. Parametre JSON'u isteğe bağlıdır; verilmediğinde seçilen algoritmanın varsayılan parametreleri kullanılır. Etiketli test dosyası için metrikler hesaplanır. Etiketsiz test dosyasında tahmin modu açılır. Yapay örnekte `1` yalnızca yapay pozitif sınıftır; herhangi bir klinik anlam taşımaz.

## Yetkili yerel veri

Profil ve beklenen dosya adı:

- `kanser`: `YARISMA_TRAIN_KANSER.csv`
- `cftr`: `YARISMA_TRAIN_CFTR.csv`
- `pah`: `YARISMA_TRAIN_PAH.csv`
- `master`: `YARISMA_TRAIN_MASTER.csv`

Ana eğitim kodu hedef kolonu `Label` olarak bekler; iki sınıf bulunmalıdır. Sayısal kodlamada `1` pozitif sınıf kabul edilir. İsteğe bağlı `Variant_ID` özelliklerden çıkarılır. Diğer kolonlar özellik olarak işlenir. Kategorik tipler ve `CAT_`/`AA_` önekleri kategorik işleme için kullanılır. Arayüzde hedef ve ID kolon adları değiştirilebilir.

Veriyi paylaşmaya yetkin olmasan bile kullanım yetkin kapsamında **yerelde** işleyebilirsin; bu depo sana veri için yeni bir kullanım veya paylaşım izni vermez.

```sh
python train.py kanser --data-dir /private/input --objective f2 --mode fast --device cpu --dry-run
python train.py kanser --data-dir /private/input --objective f2 --mode fast --device cpu
python train.py kanser --data-dir /private/input --objective f1 --mode fast --device cpu
```

`--output-dir` varsayılan olarak `runs/` olur; altında profil ve hedefe göre `kanser_f2` veya `kanser_f1` açılır. Giriş ve çıktı klasörlerinin aynı olması veya birbirini içermesi engellenir. Giriş CSV'si kopyalanmaz; alt süreç onu mutlak yoldan okur. Grafikler, JSON/CSV sonuçlar, SQLite veritabanları ve önbellekler çalışma çıktısı klasöründe kalır. Aynı profil/hedef tekrar çalıştırıldığında bu klasör yeniden kullanılır; bağımsız deney için farklı `--output-dir` seç.

Eğitim kodu modül yüklenirken çalışır. `pipelines/train_f2.py` veya `train_f1.py` dosyalarını kütüphane gibi `import` etme; `train.py` başlatıcısını kullan.

## F1 ve F2

F2 sürümü v7.9'dan, F1 sürümü v7.9x'ten gelir. F1 sürümünde bazı eski değişken ve ara çıktı adlarında `F2` yazısı korunmuştur. Optimizasyon/eşik adımlarında bu değerler F1'i ifade edebilir; nihai `test_f1` ve `test_f2` alanları kendi metriklerini hesaplar. Dengesiz sınıf dağılımlarındaki bazı seçim adımları MCC/AUC bileşimini kullanır.

Streamlit arayüzünün OOF'tan yeniden eşik seçme seçeneği **F2 odaklıdır**. F1 pipeline'ı çalıştırmak arayüzün OOF seçimini F1'e dönüştürmez. F1 JSON'u kullanılırsa JSON eşiği seçeneğini ve ilgili kolonların/parametrelerin eşleşmesini ayrıca kontrol et.

## Değerlendirme sınırları

ID çakışması arayüzde işlemi durdurur. ID bulunmaması veya ID'lerin farklı olması biyolojik olarak ilişkili örneklerin ayrıldığı garantisini vermez. Deneyler arasında aynı test setine göre model/eşik seçmek değerlendirmeyi etkiler. Bu yayın mevcut araştırma yöntemlerini belgelemektedir; dış doğrulama yapıldığı iddiasında değildir.

Arayüzün tahmin/sonuç indirmeleri de özel veri içerebilir. Bunları issue, pull request veya ekran görüntüsü içine ekleme. Uygulama varsayılan olarak yalnızca `127.0.0.1` adresinde dinler.
