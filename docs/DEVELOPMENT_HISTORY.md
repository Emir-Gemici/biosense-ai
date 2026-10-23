# Geliştirme geçmişi

Bu proje Git kullanılmadan yerelde geliştirilmişti. Yayına hazırlık sırasında 25 farklı eğitim kodu dosyası bulundu. Bu gerçek dosyalar kronolojik dosya değiştirme sırasıyla ayrı commit'lere aktarıldı; çalışma uydurma parçalara bölünmedi.

**Commit tarihleri gerçek aktarım zamanıdır.** Eski dosya tarihleri commit açıklamalarında ve `source-history.json` içinde tutulur. Dosya değiştirme zamanı; kopyalama, düzenleme veya geri yükleme nedeniyle değişebileceğinden kesin yazarlık/geliştirme tarihi kabul edilmez. Önceden var olmayan bir Git geçmişinin geri bulunduğu iddia edilmez.

## Kodda görülen aşamalar

- **v2–v3:** çoklu algoritma, Optuna optimizasyonu, eşik seçimi ve sonuç üretimi üzerine ilk sürümler.
- **v4–v5:** algoritma havuzu ve değerlendirme/ensemble akışının genişlemesi.
- **v6:** çalışma modu, cihaz seçimi, sistem kaynaklarına göre paralellik ve önbellek düzenlemeleri.
- **v7.0–v7.5:** trial ilerlemesi, iş yüküne göre paralellik ve hesaplama bütçesi düzenlemeleri.
- **v7.7–v7.9:** train/test kategorik uyumu, stratified bootstrap ve dengesiz veri için MCC/AUC temelli seçim adımları.
- **v7.9x:** ayrı F1 optimizasyon seçeneği ve ayrı önbellek/çıktı adları.
- **Paylaşım hazırlığı:** veri içermeyen depo, taşınabilir başlatıcı, sentetik demo, veri koruması, testler ve kullanım belgeleri.

Bu başlıklar kod ve yerel notların özetidir; eski notlarda yazan hız veya başarı rakamları yeniden ölçülmüş sonuçlar olarak taşınmamıştır. Dosya adı ile dosya içindeki sürüm başlığı her zaman aynı değildir; örneğin ilk `biosense_v2_ultra.py` başlığında v2.2 yazar.

## Özgün kaynaklar ve düzenlemeler

`source-history.json`, 25 eğitim kopyası yanında F1 sürümü ve güncel arayüzün özgün dosya adını, boyutunu, değiştirme zamanını ve SHA-256 özetini içerir. Yarışma verilerinin hash'leri veya içerikleri burada yer almaz.

SHA-256 özetleri özgün dosyanın ham baytlarına aittir. Git'in LF satır sonu normalizasyonu nedeniyle `git show` ile alınan içeriğin özeti farklı olabilir.

Arşiv commit'leri özgün eğitim kodunu korur. Güncel sürümlerde eski kullanım başlığı sadeleştirildi, hatalı depo bağlantısı kaldırıldı ve `BIOSENSE_DATA_DIR` üzerinden ayrı giriş klasörü desteği eklendi. Eğitim algoritması yeniden tasarlanmadı. Arayüzde mevcut ID çakışması uyarısına işlemi durdurma eklendi.

Arşiv sürümleri yalnızca inceleme içindir. Eski hatalar ve ağır varsayılan eğitim bütçeleri içerebilir; yeni kullanım için `main` dalındaki başlatıcı tercih edilmelidir.

```sh
git log --oneline --reverse
git log --format=fuller -- pipelines/train_f2.py
git diff <ilk-commit> <sonraki-commit> -- pipelines/train_f2.py
```

Git, yazar tarihi ile commit tarihini ayrı tutabilir; bu aktarımda eski tarihler atanmadı. [Git commit belgeleri](https://git-scm.com/docs/git-commit).

## Arşiv commit'leri

Her madde gerçek bir eski kaynak dosyasını gösterir; tarihler yerel dosya değiştirme zamanlarıdır.

- `2026-05-13T22:18:22+03:00` — `biosense_v2_ultra.py` — commit `3a1981bbff`.
- `2026-05-16T09:23:07+03:00` — `biosense_v2.7_ultra.py` — commit `a70b88f7a7`.
- `2026-05-16T22:39:59+03:00` — `biosense_v3.3_ultra.py` — commit `85e744a726`.
- `2026-05-17T18:29:37+03:00` — `biosense_v3.5_ultra.py` — commit `69734daff1`.
- `2026-05-18T00:35:00+03:00` — `biosense_v3.6_ultra.py` — commit `743f18952d`.
- `2026-05-18T10:33:20+03:00` — `biosense_v4_ultra.py` — commit `da8d3a0829`.
- `2026-05-18T16:14:24+03:00` — `biosense_v4.1_ultra.py` — commit `abfb5bc19c`.
- `2026-05-19T10:04:35+03:00` — `biosense_v4.3_ultra.py` — commit `929d2327d2`.
- `2026-05-19T14:57:10+03:00` — `biosense_v4.5_ultra.py` — commit `8342f0fe88`.
- `2026-05-20T10:41:27+03:00` — `biosense_v5.1_ultra.py` — commit `d26fa5c6cc`.
- `2026-05-23T11:49:02+03:00` — `biosense_v5.5_ultra.py` — commit `df6c08f932`.
- `2026-05-23T16:46:43+03:00` — `biosense_v6.0_ultra.py` — commit `516792bc8c`.
- `2026-05-23T16:54:15+03:00` — `biosense_v6.1_ultra.py` — commit `f016de4df6`.
- `2026-05-24T15:42:50+03:00` — `biosense_v6.2.1_ultra.py` — commit `f1c7f1af2b`.
- `2026-05-24T17:04:28+03:00` — `biosense_v6.2.2_ultra.py` — commit `deafbf35e7`.
- `2026-05-25T16:50:08+03:00` — `biosense_v6.3_ultra.py` — commit `9ceb497f4d`.
- `2026-05-25T17:01:58+03:00` — `biosense_v6.4_ultra.py` — commit `8945d1929a`.
- `2026-05-26T11:53:53+03:00` — `biosense_ai_v6.5.py` — commit `ec7f1d0a87`.
- `2026-05-26T16:06:21+03:00` — `biosense_ai_v7.0.py` — commit `b49d812c64`.
- `2026-05-27T14:37:22+03:00` — `biosense_ai_v7.1.py` — commit `c08288d4dc`.
- `2026-05-27T20:29:25+03:00` — `biosense_ai_v7.2.py` — commit `f6a4e7ca05`.
- `2026-05-27T21:54:37+03:00` — `biosense_ai_v7.5.py` — commit `ba0d58b1fa`.
- `2026-05-28T10:59:24+03:00` — `biosense_ai_v7.7.py` — commit `9cc33f81cc`.
- `2026-05-28T16:44:51+03:00` — `biosense_ai_v7.8.py` — commit `f427b92292`.
- `2026-05-29T13:56:35+03:00` — `biosense_ai_v7.9.py` — commit `bb45033235`.
