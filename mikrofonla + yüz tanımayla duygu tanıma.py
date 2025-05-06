import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from collections import deque, Counter

# === YÜZ DUYGU MODELİ ===
model = load_model("duygu_modeli.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === DUYGU GEÇMİŞİ (Stabilizasyon için) ===
emotion_history = deque(maxlen=10)  # Son 10 tahmini tutar

# === SES DUYGU ANALİZİ (Senin modelinle entegre edilmeli) ===
def analyze_audio_emotion(audio_path):
    return "Neutral"  # Placeholder

# === ANA FONKSİYON ===
def run_duygu_sistemi():
    cap = cv2.VideoCapture(0)
    fs = 44100
    duration = 3
    print("🎥 Kamera açılıyor, ses kaydı başlatılıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.equalizeHist(roi)  # Kontrast iyileştirme
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)
            current_emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Tahmini geçmişe ekle
            emotion_history.append(current_emotion)
            most_common_emotion = Counter(emotion_history).most_common(1)[0][0]

            # Görselde göster
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{most_common_emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Negatif duygu varsa ses analizi yap
            if most_common_emotion in ["Fear", "Angry"] and confidence > 0.75:
                print(f⚠️ GÖRSEL DUYGU ALGILANDI: {most_common_emotion} ({confidence:.2f})")

                print("🔴 Ses kaydı başlatılıyor...")
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                audio_filename = "audio.wav"
                write(audio_filename, fs, recording)
                print("🎧 Ses kaydı tamamlandı.")

                audio_emotion = analyze_audio_emotion(audio_filename)
                print(f"🎙️ SES DUYGU ALGILANDI: {audio_emotion}")

                if audio_emotion in ["Fear", "Angry"]:
                    print("🚨 Ses ve yüz ifadesine göre kullanıcı korkmuş veya sinirli!")

        cv2.imshow("Yüz Duygu Tanıma", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_duygu_sistemi()
