# Facial Expression Recognition using CNN

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** to recognize facial expressions from images. The model is trained on a dataset of facial expressions and achieves **62% validation accuracy**. It can classify images into different emotion categories like Happy, Sad, Angry, etc.

## 🚀 Features
- Uses a **CNN model** for emotion classification.
- Supports real-time predictions.
- Implements **data augmentation** to improve generalization.
- Trained on a labeled dataset of facial expressions.

## 🏗️ Model Architecture
- **Convolutional Layers** with ReLU activation.
- **MaxPooling** to reduce spatial dimensions.
- **Dropout layers** to prevent overfitting.
- **Dense layers** with Softmax for classification.

## 📊 Dataset
The model is trained on the **FER-2013 dataset**, which contains grayscale images of different facial expressions.
- Classes: **Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust**.
- Image Size: **48x48 pixels**.
- Source: [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## 🔧 Installation
Clone the repository:
```bash
git clone https://github.com/your-username/facial-expression-recognition.git
cd facial-expression-recognition
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model
To train the CNN model, run:
```bash
python train.py
```
- The model will be saved as `facial_expression_model.h5`.

## 🎭 Making Predictions
To use the trained model for predictions:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("facial_expression_model.h5")

def predict_expression(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction)
```

## 📁 Project Structure
```
├── dataset/                # Dataset folder
├── models/                 # Saved models
├── train.py                # Training script
├── predict.py              # Inference script
├── requirements.txt        # Required dependencies
├── README.md               # Project documentation
```

## 🏆 Results
- **Training Accuracy:** ~70%
- **Validation Accuracy:** ~62%
- Further improvements can be made using **transfer learning** or **hyperparameter tuning**.

## 🤖 Future Enhancements
- Improve accuracy using **ResNet** or **EfficientNet**.
- Deploy model using **Flask or FastAPI**.
- Convert model to **TensorFlow Lite** for mobile applications.

## 🤝 Contributing
Pull requests are welcome! Feel free to **fork** this repository and submit improvements.

## 📜 License
This project is **open-source** under the MIT License.

---
