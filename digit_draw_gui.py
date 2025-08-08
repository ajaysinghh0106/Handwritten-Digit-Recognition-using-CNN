# digit_draw_gui.py

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import os

# Optional: Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load trained model
model = tf.keras.models.load_model("cnn_model.keras")

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Recognize", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.label = tk.Label(root, text="Draw a digit and click Recognize", font=("Arial", 14))
        self.label.pack(pady=10)

    def paint(self, event):
        x1, y1 = event.x - 12, event.y - 12
        x2, y2 = event.x + 12, event.y + 12
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a digit and click Recognize")

    def predict(self):
        # Process image
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

        # Predict
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.label.config(text=f"Prediction: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
