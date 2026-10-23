# Veri ve paylaşım politikası

TEKNOFEST veri setinin paylaşılmaması proje sahibinin açık koşuludur. Bu depo **kod ve yeniden yazılmış proje belgeleri** içerir. Veri arşivleri açılmamış, gerçek veri satırları örneklere kopyalanmamıştır.

## Depoya alınmayan içerikler

- Ham yarışma verileri ve dağıtım arşivleri.
- Eğitim/test bölünmeleri, temizlenmiş veriler, kodlanmış özellikler, alt kümeler ve satır bazlı tahminler.
- Eğitilmiş modeller, SHAP önbellekleri, Optuna/SQLite veritabanları, eğitim günlükleri ve sonuç JSON/CSV dosyaları.
- Yarışma raporlarının asılları, itiraz belgeleri, CV'ler, imzalar, kişisel belgeler ve özel yazışmalar.
- Kaynağı veya yeniden dağıtım izni kesinleştirilmemiş diğer veri setleri.

Başka bir veri setinin lisansının açık olması, TEKNOFEST'e ait bir kopyasının veya türevinin bu depoya eklenebileceği anlamına gelmez. Bu yayın hiçbir gerçek veri setini dağıtmaz.

## Katmanlar

1. Depo, özel çalışma klasöründen ayrı bir alt klasördür. Orijinal dosyalar yerlerinde kalır.
2. `.gitignore` varsayılan olarak bütün dosyaları dışlar. Yalnızca gözden geçirilmiş tam dosya yolları istisnadır.
3. `scripts/check_public_repo.py`, Git indeksindeki **gerçek staged içeriği** inceler; çalışma kopyasındaki farklı bir dosyaya güvenmez.
4. Commit kontrolü izin listesini, dosya boyutunu, metin biçimini, bağlantıları ve yaygın gizli anahtar kalıplarını kontrol eder.
5. Push kontrolü ve CI, erişilebilen bütün commit ağaçlarını tarar. Son commit'te silinmiş bir yasak dosya eski commit'lerdeyse yine engellenir.

Klonladıktan sonra kancaları etkinleştir:

```sh
git config core.hooksPath .githooks
python scripts/check_public_repo.py --history
```

Kontroller, metin içine gizlenmiş her türlü özel veriyi otomatik olarak tanıyamaz; manuel içerik incelemesinin yerini tutmaz. Git kancaları atlanabilir ve başka bir Git istemcisinin Python'u bulması gerekir. CI uzaktaki kontroldür; yanlış yüklemeyi geriye döndüremez. Bu nedenle özel veriler mümkünse depo dışında tutulmalıdır.

## Yeni dosya ekleme

İçeriği incele, `ALLOWED_FILES` listesini ve `.gitignore` istisnasını birlikte güncelle, ardından kontrolleri çalıştır. Veri dosyasını `git add -f` ile zorla eklemek paylaşım yöntemi değildir. Sentetik örnekler bile dosya olarak commit edilmez; yalnızca veri üretim kodu paylaşılır.
