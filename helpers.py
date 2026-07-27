import cv2
import mediapipe as mp
import numpy as np

# Açı hesaplamak için fonksiyon tanımlıyoruz
def aci_hesapla(a, b, c):
    """
    Üç nokta (a, b, c) arasındaki 3D açıyı hesaplar.
    b noktası merkez (köşe/bükülme) noktasıdır.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b      # b merkez noktasından a ve c'ye giden vektörler
    bc = c - b

    # İki vektörü skaler çarpıyoruz ki formüldeki cos ile açıyı bulalım
    dot_product = np.dot(ba, bc)
    # kosinüsün açısını bulmak için skaler çarpım sonucunu vektörlerin büyüklüklerinin çarpımına böleriz
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    cosine_angel = dot_product / (norm_ba*norm_bc)
    angel = np.degrees(np.arccos(cosine_angel))

    return angel


# Parmakların Bükülme Noktalarındaki Açıların Hesabı
def elAcilari_hesapla(landmarks):
    """
    MediaPipe'ın 21 landmark noktasından 14 temel parmak bükülme açısını çıkarır.
    """
    pts = [[lm.x, lm.y, lm.z] for lm in landmarks]    # Bize 21 noktanında x, y, z koordinatlarını tutar / pts[1] başparmak kök eklemini verir

    angels = [
        # Başparmak için iki tane bükülme vardır çünkü başparmağın kökü bileğe (pts[0]) bağlı değildir
                aci_hesapla(pts[1], pts[2], pts[3]),
                aci_hesapla(pts[2], pts[3], pts[4]),
        
                # İşaret Parmağı (3 Açı)
                aci_hesapla(pts[0], pts[5], pts[6]),
                aci_hesapla(pts[5], pts[6], pts[7]),
                aci_hesapla(pts[6], pts[7], pts[8]),
                
                # Orta Parmak (3 Açı)
                aci_hesapla(pts[0], pts[9], pts[10]),
                aci_hesapla(pts[9], pts[10], pts[11]),
                aci_hesapla(pts[10], pts[11], pts[12]),
                
                # Yüzük Parmağı (3 Açı)
                aci_hesapla(pts[0], pts[13], pts[14]),
                aci_hesapla(pts[13], pts[14], pts[15]),
                aci_hesapla(pts[14], pts[15], pts[16]),
                
                # Serçe Parmak (3 Açı)
                aci_hesapla(pts[0], pts[17], pts[18]),
                aci_hesapla(pts[17], pts[18], pts[19]),
                aci_hesapla(pts[18], pts[19], pts[20])
    ]

    return angels
        