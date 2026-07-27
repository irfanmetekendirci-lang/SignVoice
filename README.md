# ✌️ SignVoice - Turkish Sign Language (TİD) Recognition System (v1.0.0)

[English](#english) | [Türkçe](#türkçe)

---

<a name="english"></a>
## 🇬🇧 English

### 📌 About The Project
**SignVoice (v1.0.0)** is a real-time computer vision system designed to recognize and translate **Turkish Sign Language (TİD)** gestures into text. By extracting 3D hand landmark spatial data using MediaPipe and classifying it with Machine Learning algorithms, it achieves high-accuracy recognition with near-zero latency via webcam input.

> 🚀 **Engineering Highlight:** The model accuracy was boosted from **80.14% to 88.49%** by implementing **Min-Max Spatial Normalization**, ensuring scale and position invariance regardless of hand distance or placement in the camera frame.

---

### 📊 Dataset & Feature Engineering
- **Total Dataset Size:** 34,849 custom processed samples.
- **Landmark Extraction:** 21 hand landmarks extracted per hand (3D coordinates: X, Y, Z), supporting up to 2 hands simultaneously (**126 total spatial features**).
- **Preprocessing & Normalization:** Applied Min-Max Spatial Normalization to map 3D coordinates relative to hand bounding boxes, eliminating distance and position noise.

---

### 🛠️ Tech Stack & Libraries
- **Language:** Python 3.12
- **Computer Vision & Tracking:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Data Processing:** Pandas, NumPy

---

### 📂 Repository Structure
SignVoice/
 ├── 1_veri_toplama.py          # Raw dataset collection from webcams/images
 ├── dataset_processor.py       # Spatial normalization (min-max) & feature pipeline
 ├── helpers.py                 # 3D vector geometry & finger angle calculations
 ├── 2_model_egitimi.py         # Random Forest model training & accuracy evaluation
 ├── 3_kamera_testi.py          # Real-time webcam inference with confidence scores
 ├── isaret_dili_verileri.csv   # Processed 154-feature dataset (Generated)
 └── isaret_dili_modeli.p       # Trained Random Forest classifier binary (Generated)

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

### 📌 Proje Hakkında
**SignVoice (v1.0.0)**, **Türk İşaret Dili (TİD)** hareketlerini gerçek zamanlı olarak algılayan ve metne dönüştüren bir bilgisayarlı görü (Computer Vision) projesidir. MediaPipe kullanarak ellerin 3 boyutlu eklem noktalarını çıkarır ve Makine Öğrenmesi algoritmalarıyla sınıflandırarak kamera üzerinden düşük gecikmeyle yüksek doğrulukta tahmin üretir.

> 🚀 **Mühendislik Detayı:** Elde edilen koordinatlara **Min-Max Konumsal Normalizasyon (Spatial Normalization)** uygulanarak, elin kameraya olan uzaklığından ve konumundan bağımsız hale getirilmiş ve model başarımı **%80.14'ten %88.49'a çıkarılmıştır.**

---

### 📊 Veri Seti ve Özellik Mühendisliği (Feature Engineering)
- **Toplam Veri:** İşlenmiş 34.849 satırlık özel veri seti.
- **Nokta Tespiti:** Ekran üzerindeki her bir el için 21 adet 3D eklem noktası (X, Y, Z) takip edilir. Çift el desteği ile toplamda **126 konumsal nitelik** işlenir.
- **Ön İşleme:** Konum ve ölçek bağımsızlığı sağlamak adına koordinatlar normalize edilerek gürültülü veriler elenmiştir.

---

### 🛠️ Kullanılan Teknolojiler
- **Dil:** Python 3.12
- **Görüntü İşleme & Takip:** OpenCV, MediaPipe
- **Makine Öğrenmesi:** Scikit-Learn (Random Forest)
- **Veri Analizi:** Pandas, NumPy

---

## 🛠️ Installation & Setup / Kurulum

1. Clone the repository / Repoyu klonlayın:  
git clone https://github.com/irfanmetekendirci-lang/SignVoice.git

2. Install required packages / Gerekli kütüphaneleri yükleyin:  
pip install opencv-python mediapipe scikit-learn pandas numpy

3. Run real-time webcam inference / Kamera testini başlatın:  
python 3_kamera_testi.py