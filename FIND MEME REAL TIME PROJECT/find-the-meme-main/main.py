import cv2
import random
import numpy as np
from pathlib import Path
from deepface import DeepFace

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
MEMES_FOLDER = BASE_DIR / "memes"

def load_memes_by_emotion():
    """Groups your meme photos by the emotion in their filename."""
    # DeepFace emotion categories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    meme_dict = {e: [] for e in emotions}
    
    if not MEMES_FOLDER.exists():
        print(f"Error: Folder {MEMES_FOLDER} not found!")
        return meme_dict

    for file in MEMES_FOLDER.iterdir():
        if file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            img = cv2.imread(str(file))
            if img is not None:
                filename = file.stem.lower()
                # If 'happy' is in the filename, add it to the happy list
                for e in emotions:
                    if e in filename:
                        meme_dict[e].append(img)
                        break
    return meme_dict

def main():
    meme_library = load_memes_by_emotion()
    cap = cv2.VideoCapture(0)
    
    # State tracking to stop memes from flickering/changing too fast
    current_emotion = None
    displayed_meme = None

    print("Project Active: Mirroring your emotions with memes!")

    while True:
        ret, frame = cap.read()
        if not ret: break

        try:
            # Detect what you are feeling
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            if results:
                res = results[0]
                detected_emotion = res['dominant_emotion'] # e.g., 'happy'
                
                # ONLY change the photo if your emotion changes
                if detected_emotion != current_emotion:
                    available_memes = meme_library.get(detected_emotion, [])
                    if available_memes:
                        displayed_meme = random.choice(available_memes)
                    current_emotion = detected_emotion

                # Display Logic: Show your face and the matching meme side-by-side
                if displayed_meme is not None:
                    # Resize meme to match your webcam height
                    h, w, _ = frame.shape
                    meme_resized = cv2.resize(displayed_meme, (int(w * 0.8), h))
                    combined = np.hstack((frame, meme_resized))
                    
                    # Label the emotion on screen
                    cv2.putText(combined, f"YOU FEEL: {detected_emotion.upper()}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Emotion Mirror', combined)
                else:
                    cv2.imshow('Emotion Mirror', frame)
            else:
                cv2.imshow('Emotion Mirror', frame)

        except Exception:
            cv2.imshow('Emotion Mirror', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()