# 🐝 Bal Arısı Hastalıklarının Görüntü Tabanlı Derin Öğrenme ile Sınıflandırılması

**Öğrenci:** Muhammed Emir Seçer  
**Bölüm:** Mekatronik Mühendisliği, Fırat Üniversitesi  
**İletişim:** muhammedemirsecer@gmail.com  
**Proje Türü:** Bitirme Tezi — Deneysel Çalışma  

---

## 📌 Projenin Amacı ve Motivasyonu

Bal arıları, küresel tarımsal ekosistemin sürdürülebilirliği açısından kritik öneme sahiptir. Dünya genelinde bitki türlerinin %80'inden fazlasının tozlaşması arılar aracılığıyla gerçekleşmekte; bu durum arıcılığı hem ekolojik hem ekonomik açıdan vazgeçilmez kılmaktadır. Bununla birlikte, Varroa akarları, kovan parazitleri ve çevresel stres faktörleri gibi tehditler her yıl milyonlarca koloninin yok olmasına yol açmaktadır.

Geleneksel teşhis yöntemleri uzman bilgisi ve yoğun iş gücü gerektirmektedir. Bu çalışmada, söz konusu soruna **görüntü tabanlı otomatik hastalık tespiti** yaklaşımıyla çözüm üretilmesi hedeflenmiştir. Derin öğrenme mimarileri kullanılarak bir arının fotoğrafından hastalığın otomatik olarak sınıflandırılması amaçlanmış; bu doğrultuda birden fazla model sistematik biçimde karşılaştırılmış ve sonuçlar akademik bir makale formatında raporlanmıştır.

---

## 📂 Proje Klasör Yapısı

```
BeeDiseasesProject/
│
├── README.md                          ← Bu dosya
│
├── report/
│   └── bee_diseases_paper.tex         ← IEEE formatında LaTeX makale
│
├── part_one/
│   └── notebook_1_eda.ipynb           ← Keşifsel Veri Analizi
│
├── part_two/
│   └── notebook_2_baseline_efficientnet.ipynb   ← EfficientNetV2-S
│
├── part_three/
│   └── notebook_3_hybrid_cnn_boost.ipynb        ← Hibrit CNN + Gradyan Artırma
│
├── part_four/
│   └── notebook_4_swin_transformer.ipynb        ← Swin Transformer
│
└── part_five/
    └── notebook_5_final_ensemble.ipynb          ← Final Ensemble
```

---

## 🗂️ Veri Seti

**BeeImage Dataset** — Jenny Yang, Harvard Dataverse, 2018  
Kaggle: `emirsecer/beediseasesdataset`

| Sınıf ID | Hastalık / Durum     | Görüntü Sayısı | Oran   |
|----------|----------------------|----------------|--------|
| 0        | Karınca Sorunları    | 457            | %8.8   |
| 1        | Küçük Kovan Böceği   | 579            | %11.2  |
| 2        | Sağlıklı             | 3.384          | %65.4  |
| 3        | Soyulmuş Kovan       | 251            | %4.9   |
| 4        | Kayıp Kraliçe        | 29             | %0.6   |
| 5        | Varroa               | 472            | %9.1   |
| —        | **Toplam**           | **5.172**      | %100   |

Veri seti belirgin bir **sınıf dengesizliği** barındırmaktadır. Sağlıklı sınıfı ile Kayıp Kraliçe sınıfı arasındaki oran 116:1'e ulaşmaktadır. Bu dengesizliği gidermek amacıyla sınıf ağırlıklı kayıp fonksiyonu ve kapsamlı veri artırma teknikleri kullanılmıştır.

**Değerlendirme Protokolü:** 5-katlı Stratified K-Fold çapraz doğrulama (Fold 0 üzerinden raporlama). Eğitim: 4.137 — Doğrulama: 1.035 görüntü.

---

## 🔬 Part One — Keşifsel Veri Analizi (EDA)

**Dosya:** `part_one/notebook_1_eda.ipynb`

### Amaç
Modelleme aşamasına geçmeden önce veri setinin derinlemesine anlaşılması; sınıf dağılımı, görüntü boyutları, görsel kalite ve potansiyel sorunların tespit edilmesi.

### Yapılanlar
- Sınıf başına görüntü sayıları ve oranlarının hesaplanması; sınıf dengesizliğinin sayısal olarak belgelenmesi
- Her sınıftan örnek görüntülerin görselleştirilmesi; hastalıkların görsel karakteristiklerinin incelenmesi
- Görüntü boyutu dağılımı analizi (ortalama: ~73×72 piksel; yüksek varyans gözlemi)
- Renk kanalı istatistiklerinin (ortalama, standart sapma) sınıf bazında karşılaştırılması
- Kayıp Kraliçe sınıfındaki düşük çözünürlüklü ve bulanık görüntülerin tespit edilmesi
- Sınıf ağırlıklarının hesaplanması: [1.89, 1.49, 0.25, 3.43, 5.00, 1.83]
- Veri artırma stratejisinin EDA bulgularına dayalı olarak belirlenmesi

### Sonuç
EDA, Kayıp Kraliçe sınıfının hem görüntü adedi hem de kalite açısından kritik bir zayıf nokta oluşturduğunu ortaya koymuştur. Küçük Kovan Böceği ile Varroa sınıflarının görsel benzerliği de bu aşamada tespit edilmiştir. Bu bulgular, sonraki notebook'lardaki tasarım kararlarının temelini oluşturmuştur.

---

## 🔬 Part Two — EfficientNetV2-S (Temel Model)

**Dosya:** `part_two/notebook_2_baseline_efficientnet.ipynb`

### Amaç
Modern ve verimli bir CNN mimarisi olan EfficientNetV2-S ile güçlü bir temel (baseline) oluşturmak. Transfer öğrenme yönteminin bu veri setindeki potansiyelini değerlendirmek.

### Mimari
ImageNet-1K ağırlıklarıyla önceden eğitilmiş EfficientNetV2-S omurgasına özel bir sınıflandırma başlığı eklenmiştir:

```
Dropout(0.2) → Linear(1280, 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512, 6)
```

### Teknik Detaylar
- **İnce Ayar Stratejisi:** 2 aşamalı — ilk 5 epoch omurga dondurulmuş (yalnızca başlık eğitilmiş), ardından tüm ağ serbest bırakılmış
- **Optimize Edici:** AdamW (lr=3×10⁻⁴, ağırlık azalması=10⁻⁴)
- **Zamanlayıcı:** CosineAnnealingLR
- **Kayıp Fonksiyonu:** Sınıf ağırlıklı CrossEntropy + Etiket yumuşatma (λ=0.05)
- **Karma Hassasiyet:** AMP (Automatic Mixed Precision) ile GPU bellek optimizasyonu
- **Gradyan Kırpma:** max_norm=1.0
- **Test Zamanı Artırma (TTA):** 5 farklı dönüşüm
- **Açıklanabilirlik:** Grad-CAM görselleştirmeleri üretilmiştir

### Sonuç
| Metrik   | Değer    |
|----------|----------|
| Doğruluk | **%98,55** |
| F1 Skoru | **%98,55** |

Literatürdeki VGG-19 tabanlı en iyi sonucu (%98,65) yalnızca 0,10 puan geride bırakan bu model, çok daha modern ve verimli bir mimariyle elde edilmiştir. Sınıf bazında mükemmel (1.00) F1 değerleri: Karınca Sorunları, Soyulmuş Kovan, Kayıp Kraliçe.

---

## 🔬 Part Three — Hibrit CNN + Gradyan Artırma

**Dosya:** `part_three/notebook_3_hybrid_cnn_boost.ipynb`

### Amaç
Derin öğrenmenin güçlü öznitelik çıkarma kapasitesini klasik makine öğrenmesi yöntemlerinin yorumlanabilirliği ve esnekliğiyle birleştiren hibrit bir yaklaşım araştırmak. Bu, literatürde bu veri seti için daha önce denenmemiş özgün bir yöntemdir.

### Yöntem
EfficientNetV2-S son sınıflandırma katmanı kaldırılarak saf bir öznitelik çıkarıcıya dönüştürülmüş ve her görüntü için 1.280 boyutlu bir gömme vektörü elde edilmiştir. Bu vektörler üç farklı gradyan artırma sınıflandırıcısına beslenmiştir:

| Model     | Hiperparametre Optimizasyonu | Özel Özellik |
|-----------|------------------------------|--------------|
| XGBoost   | Optuna (50 deneme)           | GPU hızlandırmalı `hist` yöntemi |
| LightGBM  | Optuna (50 deneme)           | Yaprak bazlı büyüme + erken durdurma |
| CatBoost  | Optuna (30 deneme)           | Ordered Boosting, simetrik ağaçlar |

Üç modelin F1 ağırlıklı yumuşak oylama ensemble'ı oluşturulmuştur.

**Not:** XGBoost'un yeni sürümlerinde `gpu_hist` parametresi kaldırılmıştır. Doğru kullanım: `tree_method='hist'` + `device='cuda'`.

### Sonuç
| Metrik   | Değer    |
|----------|----------|
| Doğruluk | **%91,01** |
| F1 Skoru | **%90,61** |

Uçtan uca derin öğrenme modelleriyle rekabet edememiştir. Ancak bu beklenen bir sonuçtur; zira öznitelikler hastalık tespiti için optimize edilmemiş, genel amaçlı bir ağdan elde edilmiştir. Bununla birlikte bu yaklaşım; **yorumlanabilirlik** (öznitelik önem analizi), **hesaplama verimliliği** ve **modülerlik** açısından değerli içgörüler sunmaktadır.

---

## 🔬 Part Four — Swin Transformer

**Dosya:** `part_four/notebook_4_swin_transformer.ipynb`

### Amaç
Vision Transformer ailesinin en güçlü temsilcilerinden biri olan Swin Transformer'ı bu veri setinde **literatürde ilk kez** uygulamak ve kaydırmalı pencere dikkat mekanizmasının arı hastalığı tespitindeki performansını ölçmek.

### Neden Swin Transformer?
Geleneksel CNN'ler yerel örüntüleri iyi modellerken uzak konumdaki bağlamsal ilişkileri yakalamakta yetersiz kalabilir. Swin Transformer, **kaydırmalı pencere tabanlı öz-dikkat mekanizması** sayesinde hem yerel (doku, renk, morfoloji) hem de küresel (vücut şekli, uzamsal düzenleme) bilgiyi eş zamanlı modelleyebilir. Bu özelliği onu hastalık tespiti gibi ince görsel farklılıkların belirleyici olduğu görevler için teorik olarak üstün kılmaktadır.

### Mimari
`swin_small_patch4_window7_224` (timm kütüphanesi, ImageNet-22K ağırlıkları)  
Embed boyutu: 768 — Özel sınıflandırma başlığı:

```
LayerNorm(768) → Dropout(0.2) → Linear(768, 512) → GELU → Dropout(0.3) → Linear(512, 6)
```

### Teknik Detaylar
- **İnce Ayar:** 2 aşamalı — ilk 5 epoch yalnızca başlık, ardından tüm ağ
- **Zamanlayıcı:** OneCycleLR (%10 warmup + kosinüs soğutma) → CosineAnnealingLR
- **Warmup:** Transformer mimarilerinde kararlı başlangıç için kritik öneme sahiptir
- **TTA:** 5 dönüşüm
- **GPU:** Kaggle T4 × 2, Karma Hassasiyet (AMP)

### Sonuç
| Metrik   | Değer    |
|----------|----------|
| Doğruluk | **%98,74** ⭐ Literatür rekoru |
| F1 Skoru | **%98,75** ⭐ Literatür rekoru |

Mevcut literatür rekoru olan %98,65'i (Liang, 2022 — VGG-19) aşmıştır. Bu veri setinde Swin Transformer'ın ilk uygulaması olması bakımından özgün bir katkı niteliği taşımaktadır.

---

## 🔬 Part Five — Final Ensemble

**Dosya:** `part_five/notebook_5_final_ensemble.ipynb`

### Amaç
En yüksek performanslı iki modeli (EfficientNetV2-S + Swin Transformer) birleştirerek bireysel modellerin zayıf noktalarını telafi eden, daha gürbüz bir sistem oluşturmak. CNN ve Transformer mimarisinin tamamlayıcılığını araştırmak.

### Yöntem
Her iki model için 5×TTA uygulanmış olasılık vektörleri elde edilmiş; ardından beş farklı birleştirme stratejisi sistematik biçimde test edilmiştir:

| Strateji            | Ağırlıklar      | Doğruluk |
|---------------------|-----------------|----------|
| **Eşit Ortalama**   | 0.5 / 0.5       | **%98,65** |
| F1 Ağırlıklı        | 0.49 / 0.51     | %98,55   |
| Swin Ağırlıklı      | 0.4 / 0.6       | %98,45   |
| Maksimum Olasılık   | —               | %98,26   |
| Geometrik Ortalama  | —               | %98,36   |


### Sonuç
| Metrik   | Değer    |
|----------|----------|
| Doğruluk | **%98,65** |
| F1 Skoru | **%98,65** |

Toplam 1.035 örnekten yalnızca **14 tanesi** (%1,35) yanlış sınıflandırılmıştır. Hataların %71,4'ü Küçük Kovan Böceği ↔ Varroa sınıfları arasında gerçekleşmiştir; bu iki sınıfın yüksek görsel benzerliği ve düşük görüntü çözünürlüğü (~73×72 piksel) bu durumu açıklamaktadır.

---

## 📊 Genel Sonuçlar ve Literatür Karşılaştırması

| Model | Doğruluk (%) | F1 (%) | Literatürde İlk Mi? |
|-------|-------------|--------|---------------------|
| **Swin Transformer** | **98,74** ⭐ | **98,75** ⭐ | ✅ Evet |
| VGG-19 [Liang, 2022] | 98,65 | — | — |
| **Final Ensemble** | **98,65** | **98,65** | ✅ Evet |
| **EfficientNetV2-S** | **98,55** | **98,55** | ✅ Evet |
| VGG-19 [Kaplan Berkaya, 2021] | 98,07 | 94,19 | — |
| CNN+MLFB [Metlek, 2021] | 95,04 | 95,04 | — |
| BeeNet [Yoo, 2023] | 94,50 | — | — |
| CM+SVM [Kılıç, 2024] | 94,03 | 89,45 | — |
| CNN [Üzen, 2019] | 92,42 | — | — |
| ResNet-50 [Margapuri, 2020] | 91,90 | — | — |
| **Hibrit CNN+GB** | **91,01** | **90,61** | ✅ Evet |
| DenseNet-121 [Chawane, 2022] | 91,60 | 88,25 | — |
| SMOTE+CNN [Karthiga, 2021] | 84,00 | — | — |

---

## 🏆 Özgün Akademik Katkılar

Bu çalışma, BeeImage veri seti üzerinde aşağıdaki beş özgün katkıyı sunmaktadır:

1. **Swin Transformer'ın ilk uygulanması** ve mevcut literatür rekorunun kırılması (%98,74)
2. **EfficientNetV2-S'nin ilk kapsamlı değerlendirmesi** (%98,55)
3. **CNN + Transformer heterojen ensemble** stratejisinin sistematik araştırılması
4. **EfficientNet öznitelikleri + XGBoost/LightGBM/CatBoost** hibrit yaklaşımı ve Optuna optimizasyonu
5. **Grad-CAM açıklanabilirlik analizi** ile model karar mekanizmalarının görselleştirilmesi

---

## ⚙️ Teknik Ortam

| Bileşen | Versiyon / Detay |
|---------|-----------------|
| Platform | Kaggle Notebooks |
| GPU | Tesla T4 × 2 |
| Python | 3.12 |
| PyTorch | 2.0 |
| CUDA | 11.8 |
| timm | 0.9+ |
| XGBoost | 2.0+ |
| LightGBM | 4.0+ |
| CatBoost | 1.2+ |
| Optuna | 3.0+ |
--- 




