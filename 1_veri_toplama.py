import cv2
import mediapipe as mp
import csv
import os

# --- 1. KONTROL PANELİ ---
ETIKET = "Evet"             # Buraya toplamak istediğiniz işaret dilini yazın  
CSV_YOLU = "isaret_dili_verileri.csv"
HEDEF_ORNEK_SAYISI = 100  

veri_listesi = []
ornek_sayaci = 0
kayit_basladi = False

# --- 2. MEDIAPIPE VE KAMERA ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Çift el algılamayı aktif ettik
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

kamera = cv2.VideoCapture(0)
print(f"'{ETIKET}' için çift el uyumlu sistem hazır. Başlamak için 's' basın.")

while True:
    durum, kare = kamera.read()
    if not durum: break
        
    kare = cv2.flip(kare, 1)
    RGB_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    sonuclar = hands.process(RGB_kare)
    
    # UI Göstergeleri
    durum_metni = f"Kayit: {'KAYDEDILIYOR...' if kayit_basladi else 'BEKLENIYOR (s basin)'}"
    cv2.putText(kare, durum_metni, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if kayit_basladi else (0, 255, 0), 2)
    cv2.putText(kare, f"Ornek: {ornek_sayaci}/{HEDEF_ORNEK_SAYISI}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # İskeletleri ekrana çiz
    if sonuclar.multi_hand_landmarks:
        for el_noktalari in sonuclar.multi_hand_landmarks:
            mp_drawing.draw_landmarks(kare, el_noktalari, mp_hands.HAND_CONNECTIONS)

    # VERİ TOPLAMA ALGORİTMASI (ÇİFT EL UYUMLU)
    if kayit_basladi and ornek_sayaci < HEDEF_ORNEK_SAYISI:
        tek_kare_verisi = []
        
        # Eğer ekranda en az bir el tespit edildiyse
        if sonuclar.multi_hand_landmarks:
            # Algılanan ellerin koordinatlarını topla
            for el_noktalari in sonuclar.multi_hand_landmarks:
                for lm in el_noktalari.landmark:
                    tek_kare_verisi.extend([lm.x, lm.y, lm.z])
            
            # Eğer sadece TEK EL algılandıysa, ikinci el yerine 63 tane 0.0 ekle (Sabit boyut koruması!)
            if len(sonuclar.multi_hand_landmarks) == 1:
                tek_kare_verisi.extend([0.0] * 63)
                
        else:
            # Eğer o karede hiç el algılanamadıysa 126 tane 0.0 ekle
            tek_kare_verisi.extend([0.0] * 126)
            
        # Sonuna etiketi yapıştır ve listeye ekle
        tek_kare_verisi.append(ETIKET)
        veri_listesi.append(tek_kare_verisi)
        ornek_sayaci += 1
                
    cv2.imshow("SignVoice Veri Toplama", kare)
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('s'): kayit_basladi = True
    if tus == ord('q') or ornek_sayaci >= HEDEF_ORNEK_SAYISI: break

kamera.release()
cv2.destroyAllWindows()

# --- 4. 2 EL UYUMLU CSV YAZMA ---
if len(veri_listesi) > 0:
    sutunlar = []
    # 1. El için (0-20) ve 2. El için (21-41) toplam 42 noktanın başlıkları
    for i in range(42):
        sutunlar.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    sutunlar.append("etiket")
    
    dosya_mevcut = os.path.exists(CSV_YOLU)
    with open(CSV_YOLU, mode="a", newline="", encoding="utf-8") as f:
        yazar = csv.writer(f)
        if not dosya_mevcut: yazar.writerow(sutunlar)
        yazar.writerows(veri_listesi)
    print(f"\nBaşarılı! Çift el uyumlu {ornek_sayaci} adet veri kaydedildi.")