
# 👁️ Eye Classification Using ResNet50

This project is a Deep Learning–powered **Eye Image Classification System** designed to validate whether uploaded eye images are suitable for medical analysis. The model classifies an input image into **five classes**, grouped under **Positive** and **Negative** categories.

## ✅ Positive Classes (Valid Image)
- **Open Eye**
- **Blur Eye**

## ❌ Negative Classes (Invalid Image)
- **Closed Eye**
- **Entire Face**
- **Random Object**

If a negative class is detected, the system automatically triggers a workflow to **send a message to the patient** requesting a clear and valid eye image.

The model is built using **ResNet50** (transfer learning) and incorporates modern Deep Learning techniques such as data augmentation, fine-tuning, softmax multi-class classification, and image preprocessing.

---

# 🚀 Features
- 📸 **Image Upload & Classification**
- 🧠 **ResNet50-based Deep Learning Model**
- ⚠️ **Invalid Image Detection & Patient Notification**
- 🔧 **Clean Inference Pipeline**
- 📊 **Supports 5-Class Classification**

---

# 🧠 Model Workflow

```
Input Image
    ↓
Resize → Normalize → Preprocess
    ↓
ResNet50 Backbone (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (5-Class Softmax)
    ↓
Prediction: Positive / Negative
```

---

# 📂 Dataset Structure

```
dataset/
│── open_eye/
│── blur_eye/
│── closed_eye/
│── entire_face/
└── random_object/
```

---

# 📦 Installation & Dependencies

## Requirements
```
Python 3.10+
TensorFlow 2.12+ / PyTorch 2.0+
Keras 2.12+
NumPy 1.24+
OpenCV 4.8+
scikit-learn 1.3+
Pandas 2.0+
Matplotlib 3.7+
```

## Install Dependencies
```
pip install -r requirements.txt
```

---

# ▶️ How to Run the Code

## **1️⃣ Clone the Repository**
```
git clone <your_repo_url>
cd EyeClassificationUsingResNet50
```

## **2️⃣ Run the Notebook**
```
jupyter notebook "EyeClassificationUsingResNet50.ipynb"
```

## **3️⃣ Run Inference**
```python
from model import predict_image

result = predict_image("test.jpg")
print(result)
```

**Output Example**
```
Class: closed_eye
Status: Negative – please upload a clear eye image.
```

---

# 📁 Recommended Folder Structure

```
EyeClassification/
│── model/
│    ├── resnet50_model.h5
│    ├── train.py
│    └── predict.py
│
│── dataset/
│── notebooks/
│    └── EyeClassification.ipynb
│
│── app/
│    └── inference_api.py
│
│── README.md
└── requirements.txt
```

---

# 🎯 Future Improvements
- Add Grad-CAM heatmaps  
- Deploy using FastAPI + AWS Lambda  
- Mobile app integration  
- Enhanced blur detection  
- Add eye segmentation CNN  

---

# 🏁 Conclusion

This project demonstrates the practical use of **ResNet50, transfer learning, and image classification** to automate medical image validation. The workflow ensures data quality by filtering invalid images and prompting patients to provide clearer inputs.

