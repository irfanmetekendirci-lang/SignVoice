import cv2
import numpy as np
import pickle
import mediapipe as mp
from helpers import elAcilari_hesapla, filtreli_tahmini_bul, kelimeye_harf_ekle, kelimeyi_seslendir
from collections import deque, Counter
import time

# --- 1. EĞİTİLMİŞ MODELİ YÜKLEME ---
try:
    with open('isaret_dili_modeli.p', 'rb') as f:
        model = pickle.load(f)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"HATA: Model dosyası yüklenemedi! {e}")
    exit()

# --- 2. MEDIAPIPE EL TAKİP KURULUMU ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 3. KAMERA BAŞLATMA ---
cap = cv2.VideoCapture(0)

tahmin_havuzu = deque(maxlen=7)  

mevcut_kelime = ""
son_harf = ""
baslangic_zamani = time.time()
harf_eklendi_mi = False


print("\nKamera başlatıldı. Çıkmak için 'q' tuşuna basın...\n")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamıyor.")
        continue

    # Aynalama ve Renk Dönüşümü
    image = cv2.flip(image, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []

    # El Tespiti Var mı Kontrolü
    if results.multi_hand_landmarks:
        # Tespiti ekrana çiz
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # 1. Min-Max Sınırları
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)

            width_x = (max_x - min_x) if (max_x - min_x) > 0 else 1.0
            height_y = (max_y - min_y) if (max_y - min_y) > 0 else 1.0
            depth_z = (max_z - min_z) if (max_z - min_z) > 0 else 1.0

            # 2. Koordinat Normalizasyonu (63 Eleman)
            for lm in hand_landmarks.landmark:
                data_aux.extend([
                    (lm.x - min_x) / width_x,
                    (lm.y - min_y) / height_y,
                    (lm.z - min_z) / depth_z
                ])

            # 3. Parmak Açıları (14 Eleman)
            acilar = elAcilari_hesapla(hand_landmarks.landmark)
            data_aux.extend(acilar)

        # Tek el varsa kalan 77 elemanı 0.0 ile doldur
        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0.0] * 77)

        # Modeldan Tahmin ve Yüzde (Olasılık) Alma
        try:
            # Tüm sınıfların olasılık dağılımı (Örn: [0.01, 0.94, 0.05...])
            probabilities = model.predict_proba([np.array(data_aux)])[0]
            
            # En yüksek olasılıklı sınıfın indeksi ve yüzdesi
            max_idx = np.argmax(probabilities)
            confidence = probabilities[max_idx] * 100
            ham_tahmin = model.classes_[max_idx]

           # 1. Filtreleme Katmanı
            kararli_harf = filtreli_tahmini_bul(ham_tahmin)

            # 2. Anlık Harfi Ekrana Yaz
            metin = f"Harf: {kararli_harf} (%{confidence:.1f})"
            cv2.putText(image, metin, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            # 3. KELİME BİRLEŞTİRME (Güvenlik Kapısı)
            if confidence > 50:
                mevcut_kelime, son_harf, baslangic_zamani, harf_eklendi_mi = kelimeye_harf_ekle(
                    kararli_harf, son_harf, baslangic_zamani, mevcut_kelime, harf_eklendi_mi
                )
            else:
                # Güven %50'nin altındaysa zamanlayıcıyı sıfırla ki hatalı harf birikmesin
                baslangic_zamani = time.time()

            # 4. Oluşan Kelimeyi Ekranın Altına Yaz
            cv2.putText(image, f"Kelime: {mevcut_kelime}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
        except Exception as e:
            pass
    else:
        # El Kadrajda Yoksa Bilgi Yazısı Göster
        cv2.putText(image, "El Bekleniyor...", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        tahmin_havuzu.clear()

    cv2.imshow('SignVoice - Canlı Kamera Testi', image)

    key = cv2.waitKey(1) & 0xFF

    # 'q' tuşuna basılırsa kamerayı kapat
    if key == ord('q'):
        break

    # 'c' tuşuna basılırsa kelimeyi sıfırla
    if key == ord('c'):
        mevcut_kelime = ""
        son_harf = ""
        harf_eklendi_mi = False
        print("Kelime sıfırlandı!")

    # 's' tuşuna basılırsa biriken kelimeyi seslendir
    if key == ord('s'):
        kelimeyi_seslendir(mevcut_kelime)

cap.release()
cv2.destroyAllWindows()