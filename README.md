# Naruto Jutsu Detector 🥷✨

A real-time computer vision project that detects hand seals (jutsus) from Naruto using your webcam and overlays awesome visual effects (VFX)! 

This project uses **OpenCV**, **MediaPipe**, and custom Machine Learning to track hand gestures and trigger animated effects like the Rasengan, Chidori, Shadow Clone, and Fireball jutsu.

## ✨ Features

- **Real-Time Hand Tracking:** Utilizes MediaPipe's robust hand tracking and face mesh solutions.
- **Dynamic VFX:** Procedurally generated and sprite-based visual effects that track with your hands and face.
- **Jutsu Arsenal:**
  - **Shadow Clone Jutsu:** Cross your index fingers to duplicate yourself on screen!
  - **Rasengan:** Hold your hands apart like you're holding a sphere to form a spinning chakra ball.
  - **Chidori:** Grab one wrist with your other hand to generate intense lightning.
  - **Fireball Jutsu:** Perform the Tiger seal to breathe fire (tracks your mouth).
- **Custom ML Model:** Train your own gesture recognizer to expand the jutsu library!

## 📂 Project Structure

- `detector.py`: The main application script. Runs the webcam, tracks hands/face, predicts the gesture, and renders the VFX.
- `collect_data.py`: A utility script to record 3D landmark data of your own custom hand seals and save them to `gestures.csv`.
- `train_model.py`: Reads the collected data and trains a custom machine learning model, saving it as `gesture_model.pkl`. `detector.py` automatically uses this model if it exists.
- `vfx/`: Directory containing the sprites and videos used for the visual effects (e.g., `smoke.png`, `lightning.png`, `rasengan.png`, `rasengen.mp4`, `tiger.mov`).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed, along with the following libraries:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn
```

### Running the Detector

To simply start the detector and use the built-in heuristic rules (and custom model if you've trained one):

```bash
python detector.py
```
- Stand in front of your webcam and perform a hand seal.
- Press **'q'** in the video window to quit.

## 🧠 Training Your Own Jutsus

Want to add a new jutsu or improve the detection of existing ones?

**1. Collect Data:**
```bash
python collect_data.py
```
- Enter the name of the jutsu you want to record (e.g., "Water Dragon").
- Press **'r'** to start a 10-second countdown. Get into position!
- The script will record 300 frames of your hand seal and save the data to `gestures.csv`.
- *Tip: Don't forget to record an "Open Palm" or "None" class so the model knows when you aren't doing a jutsu.*

**2. Train the Model:**
```bash
python train_model.py
```
- This script processes `gestures.csv` and generates `gesture_model.pkl`.

**3. Test It Out:**
- Run `detector.py` again. It will automatically load your new `gesture_model.pkl` and start recognizing your custom jutsus!

## 🎮 How to perform the default seals

- **Shadow Clone (Cross Seal):** Cross your index fingers with all other fingers curled.
- **Tiger Seal (Fireball):** Clasp your hands together, extending both index fingers and thumbs straight up.
- **Rasengan:** Hold both hands out as if holding an invisible ball between your palms.
- **Chidori:** Wrap one hand tightly around the wrist of your other hand.

---
*Believe it!* 🍥
