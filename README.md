# ✌️ SignVoice - Turkish Sign Language (TİD) Recognition System (v1.1.0)

[English](#english) | [Türkçe](#türkçe)

---

<a name="english"></a>
## 🇬🇧 English

![SignVoice Demo](demo.gif)

### 📌 About The Project
**SignVoice (v1.1.0)** is a real-time computer vision system designed to recognize and translate **Turkish Sign Language (TİD)** gestures into text and spoken audio. By extracting 3D hand landmark spatial data using MediaPipe and classifying it with Machine Learning algorithms, it achieves high-accuracy recognition with near-zero latency via webcam input.

> 🚀 **Engineering Highlights:** 
> - **Spatial Normalization:** Model accuracy was boosted from **80.14% to 88.49%** by implementing **Min-Max Spatial Normalization**, ensuring scale and position invariance regardless of hand distance or placement.
> - **Temporal Buffer & TTS:** Features a confidence-filtered buffer mechanism for smooth word construction and leverages AI-powered Neural TTS for realistic Turkish audio playback.

---

### 📊 Dataset & Feature Engineering
- **Total Dataset Size:** 34,849 custom processed samples.
- **Landmark Extraction:** 21 hand landmarks extracted per hand (3D coordinates: X, Y, Z), supporting up to 2 hands simultaneously (**126 total spatial features**).
- **Preprocessing & Filtering:** Applied Min-Max Spatial Normalization and confidence thresholding (>50%) to suppress false positives and noise.

---

### 🛠️ Tech Stack & Libraries
- **Language:** `Python 3.12`
- **Computer Vision & Tracking:** `OpenCV`, `MediaPipe`
- **Machine Learning:** `Scikit-Learn` (Random Forest Classifier)
- **Data Processing:** `Pandas`, `NumPy`
- **Text-to-Speech (TTS) & Audio:** `Edge-TTS` (Microsoft Neural Voices), `Pygame`, `Asyncio`

---

### 📂 Repository Structure
```text
SignVoice/
├── 1_veri_toplama.py       # Raw dataset collection from webcams/images
├── dataset_processor.py    # Spatial normalization (min-max) & feature pipeline
├── helpers.py              # 3D vector geometry, word assembly & Edge-TTS engine
├── 2_model_egitimi.py      # Random Forest model training & accuracy evaluation
├── 3_kamera_testi.py       # Real-time webcam inference with confidence scores
├── isaret_dili_verileri.csv# Processed 154-feature dataset (Generated)
└── isaret_dili_modeli.p   # Trained Random Forest classifier binary (Generated)
```

---

### ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/irfanmetekendirci-lang/SignVoice.git](https://github.com/irfanmetekendirci-lang/SignVoice.git)
   cd SignVoice
   ```

2. **Install required packages:**
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy edge-tts pygame
   ```

3. **Run real-time webcam inference:**
   ```bash
   python 3_kamera_testi.py
   ```

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

### 📌 Proje Hakkında
**SignVoice (v1.1.0)**, **Türk İşaret Dili (TİD)** hareketlerini gerçek zamanlı olarak algılayan, metne dönüştüren ve sesli olarak seslendiren bir bilgisayarlı görü (Computer Vision) projesidir. MediaPipe kullanarak ellerin 3 boyutlu eklem noktalarını çıkarır ve Makine Öğrenmesi algoritmalarıyla sınıflandırarak kamera üzerinden düşük gecikmeyle yüksek doğrulukta tahmin üretir.

> 🚀 **Mühendislik Detayları:** 
> - **Konumsal Normalizasyon:** Elde edilen koordinatlara **Min-Max Konumsal Normalizasyon** uygulanarak elin konumundan bağımsızlık sağlanmış ve model başarımı **%80.14'ten %88.49'a çıkarılmıştır.**
> - **Zamanlayıcı ve Doğal Seslendirme:** Kararsız tahminleri süzmek için **%50 güven eşiği** (confidence threshold) ve zamanlayıcı algoritması eklenmiş; oluşan kelimeler Microsoft Edge-TTS'in yapay zeka Türkçe ses modeli (`tr-TR-AhmetNeural`) ile insan sesine yakın kalitede seslendirilmiştir.

---

### 📊 Veri Seti ve Özellik Mühendisliği (Feature Engineering)
- **Toplam Veri:** İşlenmiş **34.849 satırlık** özel veri seti.
- **Nokta Tespiti:** Ekran üzerindeki her bir el için 21 adet 3D eklem noktası (X, Y, Z) takip edilir. Çift el desteği ile toplamda **126 konumsal nitelik** işlenir.
- **Filtreleme Katmanı:** Tahmin güven oranı %50'nin altında kalan kararsız harfler süzülerek kelimeye gürültü eklenmesi engellenir.

---

### 🛠️ Kullanılan Teknolojiler
- **Dil:** `Python 3.12`
- **Görüntü İşleme & Takip:** `OpenCV`, `MediaPipe`
- **Makine Öğrenmesi:** `Scikit-Learn` (Random Forest)
- **Veri Analizi:** `Pandas`, `NumPy`
- **Yapay Zeka Seslendirme (TTS):** `Edge-TTS`, `Pygame`, `Asyncio`

---

### ⚙️ Kurulum ve Kullanım

1. **Repoyu klonlayın:**
   ```bash
   git clone [https://github.com/irfanmetekendirci-lang/SignVoice.git](https://github.com/irfanmetekendirci-lang/SignVoice.git)
   cd SignVoice
   ```

2. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy edge-tts pygame
   ```

3. **Kamera testini başlatın:**
   ```bash
   python 3_kamera_testi.py
   ```

### ⌨️ Klavye Kısayolları

* **`s`**: Ekranda biriken kelimeyi seslendirir.
* **`c`**: Ekranda biriken kelimeyi sıfırlar/temizler.
* **`q`**: Kamerayı ve uygulamayı kapatır.