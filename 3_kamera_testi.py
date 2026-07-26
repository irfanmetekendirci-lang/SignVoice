# Bu dosyanın amacı Kameradan gelen canlı koordinatları alıp, önceden eğittiğimiz o beyin dosyasına (.p) sormak
# İlk dosyadan farkı ilk dosya kamerayı açmamızı sağlarken bu üzerinde işlem yaptırmasıdır

import cv2
import mediapipe as mp
import numpy as np
import pickle

# 1. Eğitilmiş modelimizi yüklüyoruz
with open("isaret_dili_modeli.p", "rb") as dosya: # 'rb' = Read Binary (İkili modda oku)
    model = pickle.load(dosya)

print("Eğitilmiş yapay zeka modeli hafızaya yüklendi!")

# 2. Kamera ve MediaPipe kurulumu
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Kamerayı başlatıyoruz
cap = cv2.VideoCapture(0)

# 3. Kameranın çalışacağı sonsuz döngü
while True:
    durum, kare = cap.read()
    if not durum: 
        break
        
    # Fotoğraf veri seti aynalanmadığı için cv2.flip'i kaldırabiliriz veya orijinal tutabiliriz.
    # Eğer tahminler yine şaşarsa alt satırdaki flip'i kaldırılıp denenecek:
    #kare = cv2.flip(kare, 1) 
    
    RGB_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    
    # MediaPipe ile eli tespit ediyoruz
    sonuclar = hands.process(RGB_kare)

    # 126 elemanlı veri listemizi hazırlıyoruz
    veri_listesi = []

    # Eğer ekranda tespit edilen el/eller varsa:
    if sonuclar.multi_hand_landmarks:
        for hand_landmarks in sonuclar.multi_hand_landmarks:
            mp_drawing.draw_landmarks(kare, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)

            width_x = (max_x - min_x) if (max_x - min_x) > 0 else 1.0
            height_y = (max_y - min_y) if (max_y - min_y) > 0 else 1.0
            depth_z = (max_z - min_z) if (max_z - min_z) > 0 else 1.0

            for lm in hand_landmarks.landmark:
                veri_listesi.extend([
                    (lm.x - min_x) / width_x,
                    (lm.y - min_y) / height_y,
                    (lm.z - min_z) / depth_z
                ])

        if len(sonuclar.multi_hand_landmarks) == 1:
            veri_listesi.extend([0.0] * 63)

        tahmin = model.predict([veri_listesi])
        tahmin_edilen_kelime = tahmin[0]

        # Tahmin edilen kelimeyi canlı kameranın üzerine yazdırıyoruz
        cv2.putText(kare, f"Tahmin: {tahmin_edilen_kelime}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # İşlenmiş kareyi ekranda gösteriyoruz
    cv2.imshow("SignVoice - Canli Isaret Dili Testi", kare)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()