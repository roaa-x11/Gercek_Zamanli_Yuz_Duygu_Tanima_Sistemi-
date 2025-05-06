import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
import os
import platform

# === MODEL VE LABEL TANIMLARI ===
model = load_model("duygu_modeli.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === DUYGU TARİHÇESİ ===
emotion_history = deque(maxlen=8)
last_stabilized_emotion = None  # En son gösterilen duygu

# === BİP SESİ FONKSİYONU ===
def play_beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 200)  # frekans: 1000Hz, süre: 200ms
    else:
        # Linux/macOS için bip sesi
        os.system('printf "\a"')  # ASCII bell karakteri

# === KAMERA BAŞLAT ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # Duyguyu tarihçeye ekle ve stabilize et
        emotion_history.append(emotion)
        stabilized_emotion = Counter(emotion_history).most_common(1)[0][0]

        # Eğer duygu değiştiyse bip sesi çal
        if stabilized_emotion != last_stabilized_emotion:
            print(f"🔔 Yeni Duygu: {stabilized_emotion}")
            play_beep()
            last_stabilized_emotion = stabilized_emotion

        # Görsel çizimler
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, stabilized_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Duygu Tanıma", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
