import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# csv dosyamızı pandas ile okuyoruz
veriler = pd.read_csv("isaret_dili_verileri.csv")

# Verimizin başarıyla okunup okunmadığını kontrol etmek için ilk 5 satırını yazdırıyoruz
print("Veri setinin ilk 5 satırı:")
print(veriler.head())

X = veriler.drop(columns=['etiket'])  # 'etiket' sütununu kaldırıyoruz ve x değişkenine atıyoruz
y = veriler['etiket']  # 'etiket' sütununu y değişkenine atıyoruz

# Kontrol çıktılarımız
print("X'in (Koordinatların) Boyutu:", X.shape)
print("y'nin (Etiketlerin) Boyutu:", y.shape)

# Verinin %80'ini eğitime, %20'sini teste ayırıyoruz (test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) #random_state ise her rastgele değer için 42 sağlar

# Kontrol çıktılarımız
print("\nEğitim verisi (X_train) boyutu:", X_train.shape)
print("Test verisi (X_test) boyutu:", X_test.shape)

# Boş bir Random Forest modeli oluşturuyoruz
model = RandomForestClassifier(n_estimators=500)

# Yapay zekaya eğitimi başlatıyoruz (Eğitim soruları ve Eğitim cevapları verilir)
model.fit(X_train, y_train)
print("\nModel eğitimi başarıyla tamamlandı!")

# Yapay zekaya hiç görmediği X_test sorularını verip tahmin yapmasını istiyoruz
tahminler = model.predict(X_test)

# Gerçek cevaplar (y_test) ile yapay zekanın tahminlerini karşılaştırıp başarı oranını hesaplıyoruz
basari_orani = accuracy_score(y_test, tahminler)

print(f"Modelin Doğruluk Oranı (Başarısı): %{basari_orani * 100:.2f}")

# Eğitilmiş yapay zeka modelini 'isaret_dili_modeli.p' dosyasına kaydediyoruz
with open("isaret_dili_modeli.p", "wb") as dosya:
    pickle.dump(model, dosya)

print("\nModel 'isaret_dili_modeli.p' olarak başarıyla kaydedildi!")