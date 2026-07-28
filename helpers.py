import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time
import asyncio
import edge_tts
import pygame
import os

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



# Son 7 tahmini havuzda tutacak olan kuyruk.
tahmin_havuzu = deque(maxlen=7)  

def filtreli_tahmini_bul(anlık_tahmin):
    """
    Anlık tahmini havuza ekler ve en çok tekrar eden kararlı harfi döner.
    """
    if anlık_tahmin is not None:
        tahmin_havuzu.append(anlık_tahmin)

    else:
        return ""

    # Havuzdaki karakterleri saydıracağız
    sayac = Counter(tahmin_havuzu)

    # Havuzda en çok tekrar eden ilk eleman (en çok tekrar eden en başta yer alır) ve sayısını getirelim
    enCok_tekrarlı_karakter = sayac.most_common(1)[0][0] # sonucu bir tuple ([('A', 5)] gibi ) old. içindeki harfe [0][0] ile eriştik

    return enCok_tekrarlı_karakter if enCok_tekrarlı_karakter else ""



# Harfleri birleştirerek kelime yapacağız
def kelimeye_harf_ekle(kararli_harf, son_harf, baslangic_zamani, mevcut_kelime, harf_eklendiMi):

    # Eğer harf yoksa zamanı sıfırla ve çık
    if not kararli_harf:
        return mevcut_kelime, "", time.time(), False

    # 1. Harf DEĞİŞTİ ise: Yeni harfi kaydet ve zamanı ŞİMDİYE eşitle (Sayacı baştan başlat)
    if kararli_harf != son_harf:
        son_harf = kararli_harf
        baslangic_zamani = time.time()      # yeni gelen harfi son harfe eşitledikten sonra 2 saniye geçmesini bekleyeceğiz
        harf_eklendiMi = False              # daha harfi ekledemdik

    # 2. Harf AYNI kalıyorsa: Süreyi hesapla
    gecen_sure = time.time() - baslangic_zamani

    if gecen_sure >= 2.0 and not harf_eklendiMi:
        mevcut_kelime += son_harf
        harf_eklendiMi = True

    return mevcut_kelime, son_harf, baslangic_zamani, harf_eklendiMi




# Kelimeleri sese dönüştüreceğimiz fonksiyon

import asyncio
import edge_tts
import pygame
import os

def kelimeyi_seslendir(metin):
    if not metin.strip():
        return

    async def _ses_ureteci():
        # tr-TR-AhmetNeural veya tr-TR-EmelNeural kullanabilirsin
        VOICE = "tr-TR-AhmetNeural" 
        OUTPUT_FILE = "temp_ses.mp3"
        
        communicate = edge_tts.Communicate(metin, VOICE)
        await communicate.save(OUTPUT_FILE)

    try:
        print(f"Seslendiriliyor (Edge-TTS): {metin}")
        
        # Asenkron sesi oluşturup MP3 olarak kaydet
        asyncio.run(_ses_ureteci())

        # Pygame ile sesi çal
        pygame.mixer.init()
        pygame.mixer.music.load("temp_ses.mp3")
        pygame.mixer.music.play()

        # Ses bitene kadar bekle
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Pygame'i kapat ve geçici ses dosyasını sil
        pygame.mixer.quit()
        if os.path.exists("temp_ses.mp3"):
            os.remove("temp_ses.mp3")

    except Exception as e:
        print(f"Edge-TTS Seslendirme Hatası: {e}")