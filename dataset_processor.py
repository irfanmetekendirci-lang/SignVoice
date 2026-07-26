import os
import cv2
import mediapipe as mp
import pandas as pd

# --- 1. MEDIAPIPE EL TAKİP KURULUMU ---
mp_hands = mp.solutions.hands

# Statik fotoğrafları işleyeceğimiz için static_image_mode=True yapıyoruz.
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

DATASET_PATH = "dataset"  # Fotoğrafları kopyaladığımız klasör
tum_veriler = []

# Klasör varlık kontrolü
if not os.path.exists(DATASET_PATH):
    print(f"HATA: '{DATASET_PATH}' klasörü bulunamadı! Kopyalama işleminin bitmesini bekleyin.")
else:
    print("Toplu veri işleme başlatılıyor, lütfen bekleyin...\n")
    
    # dataset içindeki tüm alt klasörleri (A, B, C...) sırayla geziyoruz
    for etiket in os.listdir(DATASET_PATH):
        etiket_yolu = os.path.join(DATASET_PATH, etiket)
        
        # Eğer bu bir klasörse işleme al
        if os.path.isdir(etiket_yolu):
            foto_sayisi = 0
            islenen_foto = 0
            
            for foto_adi in os.listdir(etiket_yolu):
                foto_yolu = os.path.join(etiket_yolu, foto_adi)
                
                # Resmi okuyoruz
                resim = cv2.imread(foto_yolu)
                if resim is None:
                    continue  # Okunamayan veya bozuk resmi atla
                
                foto_sayisi += 1
                
                # BGR -> RGB Renk Dönüşümü
                img_rgb = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
                sonuclar = hands.process(img_rgb)
                
                veri_listesi = []
                
                # Fotoğrafta el tespit edildiyse koordinatları alıyoruz
                if sonuclar.multi_hand_landmarks:
                    islenen_foto += 1
                    
                    for hand_landmarks in sonuclar.multi_hand_landmarks:
                        # 1. Elin kapladığı alanın Min ve Max sınırlarını buluyoruz
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        z_coords = [lm.z for lm in hand_landmarks.landmark]

                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        min_z, max_z = min(z_coords), max(z_coords)

                        # Sıfıra bölünme hatasını önlemek için min-max farkını kontrol ediyoruz
                        width_x = (max_x - min_x) if (max_x - min_x) > 0 else 1.0
                        height_y = (max_y - min_y) if (max_y - min_y) > 0 else 1.0
                        depth_z = (max_z - min_z) if (max_z - min_z) > 0 else 1.0

                        # 2. Her koordinatı [0, 1] aralığına normalize ediyoruz
                        for lm in hand_landmarks.landmark:
                            veri_listesi.extend([
                                (lm.x - min_x) / width_x,
                                (lm.y - min_y) / height_y,
                                (lm.z - min_z) / depth_z
                            ])

                    # Eğer tek el varsa kalan 63 koordinatı 0.0 ile doldur
                    if len(sonuclar.multi_hand_landmarks) == 1:
                        veri_listesi.extend([0.0] * 63)

                    veri_listesi.append(etiket)
                    tum_veriler.append(veri_listesi)

            print(f"[{etiket}] sınıfı işlendi: {foto_sayisi} fotoğraftan {islenen_foto} tanesinde el bulundu.")

    # CSV Sütun isimlerini oluşturuyoruz (x_0, y_0, z_0 ... z_41, etiket)
    sutunlar = []
    for i in range(42):
        sutunlar.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    sutunlar.append('etiket')

    # Pandas ile tablo yapıp CSV dosyasına yazıyoruz
    df = pd.DataFrame(tum_veriler, columns=sutunlar)
    df.to_csv("isaret_dili_verileri.csv", index=False)
    
    print("\n--------------------------------------------------")
    print(f"İŞLEM TAMAMLANDI! Toplam {len(df)} satırlık veri elde edildi.")
    print("Yeni veriler 'isaret_dili_verileri.csv' dosyasına başarıyla kaydedildi.")