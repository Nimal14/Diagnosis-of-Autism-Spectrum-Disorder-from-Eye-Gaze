## **Overview**  
This project implements a **Convolutional Neural Network (CNN)** for binary classification to detect autism traits based on image data.  
- **Frameworks Used:** TensorFlow/Keras for model building, ImageDataGenerator for preprocessing  
- **Dataset Split:** Training, Validation, and Test sets  

---

## **Dataset**  
- **Training Set:** `/content/DATASET_PRJ/train`  
- **Test Set:** `/content/DATASET_PRJ/test`  
- **Preprocessing:** Images resized to **224×224 pixels** before feeding into the model  

---

## **Model Architecture**  
The CNN model is designed with convolutional, pooling, and fully connected layers:  

- **Convolutional Layers:** 4 Conv2D layers with ReLU activation + Batch Normalization  
- **Pooling Layers:** MaxPooling2D for dimensionality reduction  
- **Dropout Layers:** Added to prevent overfitting  
- **Global Average Pooling:** Reduces feature map dimensions  
- **Dense Layers:** Fully connected layers with ReLU activation  
- **Output Layer:** Single neuron with **Sigmoid activation** (binary classification)  

---

## **Training & Evaluation**  
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  
- **Learning Rate Adjustment:** ReduceLROnPlateau  
- **Visualization:** Training/Validation accuracy & loss curves  

---

## **Key Metrics**  
The model evaluates performance using:  
- **Accuracy**  
- **Precision**  
- **Recall (Sensitivity)**  
- **Specificity**  
- **F1-Score**  
- **Confusion Matrix**  

---

## **Conclusion**  
This project successfully implements a CNN-based model for autism classification with **promising accuracy**.  

**Future Improvements:**  
- Data augmentation  
- Hyperparameter tuning  
- Transfer learning  
