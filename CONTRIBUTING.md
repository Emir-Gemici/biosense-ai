# Katkı

Çalışmayı küçük, açıklanabilir değişiklikler halinde yap. Bir hatayı anlatırken gerçek yarışma satırlarını veya kişisel bilgileri paylaşma; mümkünse sentetik üreteçle tekrar üret.

```sh
git config core.hooksPath .githooks
python -m compileall -q train.py apps pipelines scripts tests
python -m unittest discover -s tests -v
python scripts/smoke_demo.py
python scripts/check_public_repo.py --history
```

Son komut Git indeksini kontrol eder. Yeni bir kaynak/belge dosyasını önce gözden geçir, `.gitignore` ve `scripts/check_public_repo.py` içindeki izin listelerine ekle, sonra açık dosya adıyla stage et. Kod içine gömülü satırlar ve ekran görüntüleri de veri paylaşımıdır; yalnızca uzantı değiştirmek çözüm değildir.

CI standart kütüphane testlerini, sözdizimini ve veri korumasını doğrular. ML yardımcıları ve Streamlit duman testi bağımlılık kurulumu gerektirir ve yerelde ayrıca çalıştırılır. Tam eğitim uzun sürebilir; her belge değişikliğinde yeniden eğitim gerekmez.

Arşiv commit'leri eski sürümlerin izidir. Geçmişe yeni bir veri dosyası ekleme; normal geliştirmeyi güncel `main` üzerinden sürdür.
