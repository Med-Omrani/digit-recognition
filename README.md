# Digit Recognition

*TensorFlow Lite on a Raspberry Pi CPU*

A small, complete **on-device inference** example: a handwritten digit image goes in,
the predicted class 0–9 comes out, running on a Raspberry Pi CPU through TensorFlow
Lite — no cloud call, no GPU.

---

## Why this exists

It is the smallest end-to-end path through the edge-AI workflow: train → convert to
`.tflite` → run on ARM. The same conversion and inference pattern is what makes
larger models deployable on a Pi or a microcontroller, so this repo keeps the example
deliberately small and readable.

---

## Layout

```
model/            mnist_model.tflite — the converted model
src/              predict_tflite.py — loads the model and classifies one image
images/           sample digit images for testing
requirements.txt  Python dependencies
```

---

## Run it

```bash
git clone https://github.com/Med-Omrani/digit-recognition.git
cd digit-recognition

python3 -m venv mnist-env
source mnist-env/bin/activate
pip install -r requirements.txt

python src/predict_tflite.py images/image1.png
```

The script prints the predicted digit. Images should be a single digit on a plain
background; they are normalised to the 28×28 input the model expects.

---

## Notes

- Inference runs on the Pi's ARM CPU via the TFLite interpreter — expect a few
  milliseconds per image for this model size.
- This is a learning project, not a benchmark: the model is a compact CNN trained on
  standard MNIST, so it is strong on clean handwritten digits and weak on anything
  photographed under real lighting.
- To retrain or change the architecture, export a new `.tflite` and drop it into
  `model/`.

**Stack:** Python · TensorFlow Lite · NumPy · Raspberry Pi

---

## License

MIT — see `LICENSE`.
