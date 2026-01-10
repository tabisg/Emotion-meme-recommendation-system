
# Finding The Meme

**Can you make the meme face? Find the meme by making that face!**

A fun computer vision project that shows a random meme image when a face is detected in your webcam feed!

## Creator/Dev
**tabishahmad**

## How to Run
1. Make sure you have Python 3.11+ installed.
2. Install dependencies (in your project folder):
   ```
   pip install opencv-contrib-python numpy
   ```
3. Run the app:
   ```
   python main.py
   ```
4. Allow webcam access. When your face appears, a random meme will be shown next to your webcam feed.
5. Press 'q' to quit.

## Adding More Memes
- Place meme images (JPG/PNG) in the `memes/` folder.
- The filename (without extension) will be used as the meme name.
- You can add as many meme images as you want—just drop them in the folder and restart the app!

## File Structure
meme/
├── main.py                # AI Emotion Detection Script
├── requirements.txt       # Project Dependencies
├── memes/                 # Folder for your meme images
│   ├── happy_victory.jpg  # Keyword: happy
│   ├── neutral_stare.jpg  # Keyword: neutral
│   ├── ...                # Add more memes here
└── sad_face.png          # Keyword: sad

Enjoy making meme faces and seeing instant meme reactions!
