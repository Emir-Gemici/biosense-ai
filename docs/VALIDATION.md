# Yayına hazırlık doğrulaması

Bu kayıt yalnızca repoya hazırlık doğrulamasını kapsar. Yarışma verileriyle eğitim veya bilimsel performans doğrulaması yapılmamıştır.

## Yerel sonuçlar — 23 Ekim 2026

- Python 3.13.7 / Windows ortamında bütün güncel Python dosyaları sözdizimi kontrolünden geçti. 25 eski kaynak kopyası da aktarım öncesi ayrıştırıldı.
- **6 otomatik test geçti:** dosya izin listesi; ikili dosya/bağlantı/anahtar kalıpları; Git ignore kuralları; staged içerik ve geçmişte silinmiş yasak dosyayı yakalama; sentetik verinin deterministik ve ayrık olması; demo üretimi ve başlatıcı yol kontrolleri.
- Gerçek arayüz yardımcılarıyla **lojistik regresyon ve CatBoost** sentetik eğitim/tahmin kontrolü geçti. Görülmemiş kategori, sayısal eksik değer, olasılık aralığı ve karmaşıklık matrisi toplamı doğrulandı.
- Streamlit `AppTest` ile ilk sayfa açılışında uygulama istisnası oluşmadı. Tarayıcıdaki bütün kullanıcı etkileşimleri uçtan uca sınanmadı.
- Başlatıcının yardım çıktısı, olmayan dosyayı reddetmesi, çakışan giriş/çıktı yollarını reddetmesi ve `--dry-run` sırasında çıktı üretmemesi doğrulandı.
- Girdi yolu ve yardım metni uyarlamaları dışarıda tutularak F1/F2 eğitim kodlarının AST'leri özgün sürümlerle karşılaştırıldı; eğitim mantığındaki düğümler aynı kaldı.
- İzin verilen **23 takip edilen dosya** ile erişilebilen bütün commit ağaçları denetlendi. Yasak dosya yolu, ikili içerik veya tarayıcının tanıdığı kimlik bilgisi kalıbı bulunmadı. 25 eski eğitim sürümü ayrı commit'lerde korunuyor.

## Ortam sınırları

Windows'un kısıtlı çalışma ortamı Git'in `sh` sürecinin sinyal kanalını oluşturmasını engelledi. Bu hazırlıkta her commit öncesi aynı Python paylaşım kontrolü doğrudan çalıştırıldı; yalnızca kanca başlatımı komut düzeyinde atlandı. Depoda `core.hooksPath=.githooks` ayarı korunuyor. Git kancalarının normal kullanıcı terminalinde ayrıca doğrulanması gerekir.

Streamlit test yardımcılarının oluşturduğu geçici klasörün süreç sonundaki temizliğinde Windows izin uyarısı görüldü; model ve arayüz kontrolleri bundan önce başarıyla tamamlandı. Depo testleri, Windows'ta erişilebilir çalışma klasörü ve salt okunur Git nesneleri için güvenli temizlik kullanır.

GitHub Actions iş akışı eklenmiştir; uzak bir çalıştırma henüz yapılmamıştır. Bağımlılıklar mevcut kurulumda doğrulandı; boş bir sanal ortam kurulumu ve tam Optuna eğitimi bu çalışmanın kapsamında çalıştırılmadı. Gerçek veri üzerinden eski performans iddiaları doğrulanmış sayılmaz.
