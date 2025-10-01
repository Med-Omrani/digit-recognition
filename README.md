# 🖥️ MNIST Digit Recognition with TensorFlow Lite on Raspberry Pi

This project runs a **handwritten digit recognition model (MNIST)** on a Raspberry Pi using **TensorFlow Lite**.  
You give the program an image of a digit (0–9), and it predicts the digit.

---

## ✅ What you need
- Raspberry Pi 4 (or 3) with Python 3 installed
- `mnist_model.tflite` (the trained model)
- Python libraries (see below)

---

---

## ⚙️ Setup & Run

### 1 — Clone the repository
```bash
git clone https://github.com/Med-Omrani/mnist_digit_recognition.git
cd mnist_digit_recognition

## 📦 Install dependencies
Install the required Python packages:

```bash
python3 -m venv mnist-env
source mnist-env/bin/activate
pip install -r requirements.txt
 
3 — Run the prediction
python src/predict_tflite.py images/image1.png