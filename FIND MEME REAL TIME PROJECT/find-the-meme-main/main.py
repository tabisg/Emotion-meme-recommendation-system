import cv2
import random
import numpy as np
import csv
import time
from pathlib import Path
from collections import Counter
from deepface import DeepFace
import matplotlib.pyplot as plt

# ===================== PATH SETUP =====================
BASE_DIR = Path(__file__).resolve().parent
MEMES_FOLDER = BASE_DIR / "memes"
LOG_FILE = BASE_DIR / "emotion_log.csv"

# ===================== EMOTIONS =====================
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ===================== LOAD MEMES =====================
def load_memes_by_emotion():
    meme_dict = {e: [] for e in EMOTIONS}

    if not MEMES_FOLDER.exists():
        print("❌ Memes folder not found!")
        return meme_dict

    for file in MEMES_FOLDER.iterdir():
        if file.suffix.lower() in ('.jpg', '.png', '.jpeg'):
            img = cv2.imread(str(file))
            if img is not None:
                for emotion in EMOTIONS:
                    if emotion in file.stem.lower():
                        meme_dict[emotion].append(img)
                        break
    return meme_dict

# ===================== CSV LOGGER =====================
def init_csv():
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Emotion", "Confidence"])

def log_emotion(emotion, confidence):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%H:%M:%S"),
            emotion,
            round(confidence, 2)
        ])

# ===================== ANALYTICS =====================
def show_analytics():
    emotions = []

    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotions.append(row["Emotion"])

    if not emotions:
        return

    counts = Counter(emotions)
    labels = counts.keys()
    values = counts.values()

    plt.figure()
    plt.bar(labels, values)
    plt.title("Emotion Frequency Analysis")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

    print("\n📊 Session Summary")
    print("Most Frequent Emotion:", counts.most_common(1)[0][0])

# ===================== MAIN FUNCTION =====================
def main():
    init_csv()
    meme_library = load_memes_by_emotion()

    cap = cv2.VideoCapture(0)

    current_emotion = None
    displayed_meme = None

    print("🎥 Emotion Recognition System Started (Press Q to Exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if result:
                res = result[0]
                emotion = res['dominant_emotion']
                confidence = res['emotion'][emotion] / 100

                # Change meme only if emotion changes
                if emotion != current_emotion:
                    if confidence > 0.5:
                        memes = meme_library.get(emotion, [])
                        if memes:
                            displayed_meme = random.choice(memes)
                            log_emotion(emotion, confidence)
                            current_emotion = emotion

                if displayed_meme is not None:
                    h, w, _ = frame.shape
                    meme_resized = cv2.resize(displayed_meme, (int(w * 0.8), h))
                    combined = np.hstack((frame, meme_resized))

                    cv2.putText(
                        combined,
                        f"Emotion: {emotion.upper()} ({confidence:.2f})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow("Emotion Mirror System", combined)
                else:
                    cv2.imshow("Emotion Mirror System", frame)

            else:
                cv2.imshow("Emotion Mirror System", frame)

        except Exception:
            cv2.imshow("Emotion Mirror System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    show_analytics()

# ===================== RUN =====================
if __name__ == "__main__":
    main()
