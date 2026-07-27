import os
import cv2
import mediapipe as mp
import pandas as pd
from helpers import elAcilari_hesapla

# --- 1. MEDIAPIPE EL TAKİP KURULUMU ---
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

DATASET_PATH = "dataset"
tum_veriler = []

if not os.path.exists(DATASET_PATH):
    print(f"HATA: '{DATASET_PATH}' klasörü bulunamadı!")
else:
    print("Toplu veri işleme başlatılıyor, lütfen bekleyin...\n")
    
    for etiket in os.listdir(DATASET_PATH):
        etiket_yolu = os.path.join(DATASET_PATH, etiket)
        
        if os.path.isdir(etiket_yolu):
            foto_sayisi = 0
            islenen_foto = 0
            
            for foto_adi in os.listdir(etiket_yolu):
                foto_yolu = os.path.join(etiket_yolu, foto_adi)
                
                resim = cv2.imread(foto_yolu)
                if resim is None:
                    continue
                
                foto_sayisi += 1
                
                img_rgb = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
                sonuclar = hands.process(img_rgb)
                
                veri_listesi = []
                
                if sonuclar.multi_hand_landmarks:
                    islenen_foto += 1
                    
                    # En fazla 2 eli işleme alıyoruz
                    eller = sonuclar.multi_hand_landmarks[:2]
                    
                    for hand_landmarks in eller:
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

                        # 2. Koordinat Normalizasyonu (63 Adet)
                        for lm in hand_landmarks.landmark:
                            veri_listesi.extend([
                                (lm.x - min_x) / width_x,
                                (lm.y - min_y) / height_y,
                                (lm.z - min_z) / depth_z
                            ])
                        
                        # 3. Parmak Açıları (14 Adet)
                        acilar = elAcilari_hesapla(hand_landmarks.landmark)
                        veri_listesi.extend(acilar)

                    # Eğer tek el varsa 2. el için 77 adet 0.0 doldur
                    if len(eller) == 1:
                        veri_listesi.extend([0.0] * 77)

                    veri_listesi.append(etiket)
                    tum_veriler.append(veri_listesi)

            print(f"[{etiket}] sınıfı işlendi: {foto_sayisi} fotoğraftan {islenen_foto} tanesinde el bulundu.")

    # 154 Özellik + 1 Etiket = 155 Sütun
    sutunlar = [f'f_{i}' for i in range(154)]
    sutunlar.append('etiket')

    df = pd.DataFrame(tum_veriler, columns=sutunlar)
    df.to_csv("isaret_dili_verileri.csv", index=False)
    
    print("\n--------------------------------------------------")
    print(f"İŞLEM TAMAMLANDI! Toplam {len(df)} satırlık veri elde edildi.")