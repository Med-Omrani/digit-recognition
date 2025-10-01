import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
import sys
import matplotlib.pyplot as plt

MODEL_PATH = "/home/pi/mnist_model.tflite"

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded.")

def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img = ImageOps.invert(img)  # invert colors if background is white
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1), img

if len(sys.argv) < 2:
    print("Usage: python predict_tflite.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
input_data, img = preprocess_image(image_path)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_digit = np.argmax(output_data)

# Show image
plt.imshow(img, cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()

print("Predicted digit:", predicted_digit)
