
# 🐾 Cat vs Dog Image Classifier

## 📘 Overview
This project is a **Convolutional Neural Network (CNN)** based image classifier that identifies whether a given image is of a **cat** or a **dog**.  
It is trained on the popular **Kaggle Cats and Dogs dataset** and implemented in **Python using TensorFlow and Keras**.

---

## 🚀 Features
- 🧠 Deep Learning model (CNN) for binary image classification  
- 📊 High accuracy on validation and test datasets  
- 🔄 Data preprocessing and augmentation using `ImageDataGenerator`  
- 💾 Model saving and loading (`.h5` format)  
- 📈 Training visualization (loss and accuracy graphs)  
- 💡 Easy to run on Google Colab or local Jupyter Notebook  

---

## 🧩 Model Architecture
The CNN model includes:
- **Convolutional Layers:** Extract image features  
- **MaxPooling Layers:** Reduce spatial size  
- **Dropout Layers:** Prevent overfitting  
- **Dense Layers:** Fully connected layers for classification  

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Activations:** ReLU, Sigmoid  

---

## 📁 Dataset
Dataset: [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

- Total images: 25,000 (12,500 cats + 12,500 dogs)  
- Split: 80% training, 20% validation  
- Images resized to 128x128 before training  

---

## ⚙️ How to Run on Google Colab

1. **Open in Google Colab**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aniket-k-13/cat-dog-classifier/blob/main/cat-dog-classifier.ipynb)

2. **Upload dataset to Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Run all cells
The model will train and display accuracy/loss plots.

Predict on new images

python
Copy code
model.predict(new_image)
📊 Results
Metric	Value
Training Accuracy	~98%
Validation Accuracy	~96%
Loss (val)	~0.08

Performance may vary slightly depending on dataset and preprocessing.

💾 Model Saving & Loading
python
Copy code
# Save model
model.save('cat_dog_model.h5')

# Load model
from tensorflow.keras.models import load_model
model = load_model('cat_dog_model.h5')

🧑‍💻 Author
Aniket Khandare
🔗 LinkedIn https://www.linkedin.com/in/aniket-khandare-18b822329
