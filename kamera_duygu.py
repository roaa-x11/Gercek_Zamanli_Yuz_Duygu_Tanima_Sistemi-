import cv2  # Görüntü işleme için OpenCV
import numpy as np  # NumPy dizileri ve matris işlemleri
from tensorflow.keras.models import load_model  # Eğitilmiş modeli yüklemek için
from collections import deque, Counter  # Stabilizasyon için kuyruk ve sayım
import os, platform  # İşletim sistemi ve bip sesi işlemleri

# === MODELİ YÜKLE ===
try:
    model = load_model("duygu_modeli.keras")  # Eğitilmiş duygu tanıma modelini yükler
except:
    print("❌ Model yüklenemedi! Dosya mevcut mu?")
    exit()

# === DUYGU ETİKETLERİ ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Modelin sınıf çıktıları

# === YÜZ ALGILAMA ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Haar cascade ile yüz tespiti

# === DUYGU STABİLİZASYONU ===
emotion_history = deque(maxlen=8)  # Son 8 tahmini saklayan kuyruk (ani değişimi önlemek için)
last_stabilized_emotion = None  # En son sabitlenen duygu

# === SESLİ BİLDİRİM (DUYGU DEĞİŞİNCE BİP) ===
def play_beep():
    if platform.system() == "Windows":  # Windows sistemlerde
        import winsound
        winsound.Beep(1000, 200)  # 1000 Hz frekans, 200 ms
    else:
        os.system('printf "\a"')  # Mac/Linux sistemlerde terminal bip sesi

# === KAMERA BAŞLAT ===
cap = cv2.VideoCapture(0)  # Bilgisayarın varsayılan kamerasını aç
if not cap.isOpened():  # Kamera açılamazsa uyarı ver
    print("❌ Kamera açılamadı!")
    exit()

print("🎥 Sistem başlatıldı. Gerçek zamanlı duygu tanıma çalışıyor...")

# === ANA DÖNGÜ ===
while True:
    ret, frame = cap.read()  # Kameradan bir kare al
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gri formata dönüştür (model böyle bekliyor)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Yüzleri tespit et

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]  # Yüz bölgesini al (region of interest)
        roi = cv2.resize(roi, (48, 48))  # Modelin giriş boyutuna küçült
        roi = cv2.equalizeHist(roi)  # Işık farklarını azaltmak için kontrast artır
        roi = roi.astype("float32") / 255.0  # Normalize (0-1 arası)
        roi = np.expand_dims(roi, axis=-1)  # (48, 48) → (48, 48, 1)
        roi = np.expand_dims(roi, axis=0)   # (48, 48, 1) → (1, 48, 48, 1)

        prediction = model.predict(roi, verbose=0)  # Duygu tahmini yap

        if prediction.shape[1] == len(emotion_labels):  # Model çıktısı bekleniyorsa
            emotion = emotion_labels[np.argmax(prediction)]  # En yüksek skorlu duygu etiketi
        else:
            print("⚠️ Model çıktısı beklenenden farklı!")  # Hatalı çıktı varsa uyar
            emotion = "Bilinmiyor"

        emotion_history.append(emotion)  # Tahmini kuyrukta sakla
        stabilized_emotion = Counter(emotion_history).most_common(1)[0][0]  # En sık tahmini al

        if stabilized_emotion != last_stabilized_emotion:  # Yeni bir duygu tespit edildiyse
            print(f"🔔 Yeni Duygu: {stabilized_emotion}")
            play_beep()
            last_stabilized_emotion = stabilized_emotion

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Yüzün etrafına kutu çiz
        cv2.putText(frame, stabilized_emotion, (x, y-10),        # Üstüne sabitlenmiş duyguyu yaz
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Duygu Tanıma", frame)  # Kameradaki görüntüyü ekranda göster

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q tuşuna basıldığında çık
        break

cap.release()  # Kamerayı serbest bırak
cv2.destroyAllWindows()  # OpenCV pencerelerini kapat
