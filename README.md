# 🩺 Automated Skin Lesion Detection using Deep Learning

**An AI-Powered Solution for Early Diagnosis of Malignant and Benign Skin Lesions**

---

## 🚀 Project Overview

Skin cancer remains one of the deadliest forms of cancer worldwide, with millions of cases diagnosed annually. Early detection is **crucial** for effective treatment and improved patient outcomes. This project aims to provide a robust, high-accuracy classifier using **deep learning techniques** to distinguish between **malignant** (cancerous) and **benign** (non-cancerous) skin lesions.

Our solution leverages a combination of **Convolutional Neural Networks (CNN)** and advanced preprocessing techniques to analyze skin images and assist healthcare professionals in making timely diagnoses.

---

## 📊 Key Features

- **High Accuracy Classification**: Efficiently distinguishes between malignant and benign lesions.
- **Preprocessing with CLAHE**: Enhanced contrast for clearer feature extraction.
- **Data Augmentation**: Increases model robustness with techniques like rotation, zoom, and flipping.
- **User-Friendly Interface**: Designed for integration with clinical decision support systems.

---

## 🛠️ Tech Stack

- **Python** 🐍
- **TensorFlow / Keras** 🤖
- **OpenCV** 📸
- **VGG16 Model** 🧠

---

## 🗂️ Dataset

The dataset used for this project was sourced from **Kaggle** and is based on the **ISIC Archive**. It consists of **3263 images** of skin lesions, categorized into **malignant** and **benign** classes:

- **Training Set**: 2609 images
- **Testing Set**: 654 images
- **Image Size**: 224x224 pixels

### Data Augmentation Techniques:
- **Rescaling**: Normalizes pixel values
- **Rotation, Shear, Zoom**: Enhances variability
- **Flipping**: Improves model generalization

---

## ⚙️ Model Architecture

Our deep learning model utilizes a **VGG16 base** for feature extraction, combined with:

- **Dense Layers**: Two dense layers with 512 neurons for complex pattern recognition
- **Batch Normalization**: Stabilizes training and improves performance
- **Dropout Regularization**: Reduces overfitting with a 50% dropout rate
- **Sigmoid Activation**: Final classification layer for binary output (benign vs malignant)

---

## 🧠 Training Process

The model was trained for **100 epochs** using the **Adam optimizer** with a learning rate of 0.001. Key training strategies included:

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Reduction**: Automatically reduces learning rate when validation loss plateaus

---

## 🏆 Results

Despite challenges like overfitting and prolonged training times, the model achieved:

- **Training Accuracy**: 85%
- **Validation Accuracy**: 86%
- **Test Accuracy**: 74%

The final model demonstrates promising potential for assisting in clinical diagnostics.

---

## 📈 Performance Visualization

Our analysis includes detailed plots of **training and validation accuracy** trends, as well as loss curves over epochs, highlighting the model's learning progress.

---

## 💡 Future Enhancements

- Explore other CNN architectures like **ResNet**, **DenseNet**, and **InceptionNet** for improved accuracy.
- Expand the dataset with more diverse and real-world clinical images.
- Integrate the model into a **web-based application** for easier accessibility by healthcare professionals.

---

## 🤝 Contributing

We welcome contributions to enhance the model and expand its capabilities. If you have ideas for improving the project, please feel free to submit a **pull request** or **open an issue**.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions, suggestions, or feedback, please reach out to the project team:

- Damir Tassybayev
- Zhansaya Orazbay
- Dias Nursultan
- Artur Kadyrzhanov
- **Supervisor**: Prof. Daniele Pannone

---

## 🌐 References

- [Kaggle Skin Cancer Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
- [ISIC Archive](https://www.isic-archive.com)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)

Together, we can advance the fight against skin cancer and help save lives with the power of **artificial intelligence**.
