# 🚦 Traffic Sign Recognition

This is a Python-based project for recognizing **Vietnamese traffic signs** using deep learning and computer vision. It includes a trained model (`model.h5`), a Flask web app (`app.py`), and camera-based real-time detection scripts.

---

## 📁 Project Structure

```
.
├── static/                       # Static assets (CSS, JS, images)
├── templates/                   # HTML templates for web interface
├── app.py                       # Flask web application
├── main.py                      # Main script to run recognition
├── Traffic_Sign_Recognition.ipynb # Jupyter notebook for training/evaluation
├── model.h5                     # Trained Keras model
├── labels.csv                   # Class labels of traffic signs
├── camera-detect-stop.py       # Script to detect STOP sign from camera
├── camera-detect-vietnamese.py # Detect various Vietnamese traffic signs
├── image-test-for-local-app-web.rar # Test images for local web app
├── ARIAL.TTF                    # Font used for image annotation
```

---

## 🚀 Features

- ✅ Real-time webcam detection of traffic signs
- ✅ Upload & test images via Flask web interface
- ✅ Pre-trained deep learning model (`model.h5`)
- ✅ Support for Vietnamese traffic signs
- ✅ Custom scripts for camera detection

---

## 🧠 Model Information

- Framework: TensorFlow / Keras
- Model file: `model.h5`
- Input: Resized traffic sign image (e.g., 30x30)
- Output: Predicted traffic sign label from `labels.csv`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Install Dependencies

If `requirements.txt` is not available:

```bash
pip install flask tensorflow opencv-python numpy
```

### 3. Run the Web Application

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:8080
```

### 4. Run Real-time Detection with Webcam

Detect STOP signs:

```bash
python camera-detect-stop.py
```

Detect multiple Vietnamese signs:

```bash
python camera-detect-vietnamese.py
```

---

## 🧪 Testing

To test with sample images:

1. Extract `image-test-for-local-app-web.rar`
2. Use the web app to upload and detect traffic signs.

---

## 📊 Labels

Class labels are listed in `labels.csv`. Here's an example snippet:

```csv
0,Cấm đi ngược chiều
1,Dừng lại
2,Rẽ trái
...
```

---


## 📄 License

This project is intended for **educational and research purposes only**.
