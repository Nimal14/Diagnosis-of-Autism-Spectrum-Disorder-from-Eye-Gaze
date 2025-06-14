# Diagnosis-of-Autism-Spectrum-Disorder-from-Eye-Gaze
**Overview**

This project implements a Convolutional Neural Network (CNN) for binary classification to detect autism traits based on image data. It uses TensorFlow/Keras for model building and ImageDataGenerator for data preprocessing. The dataset is split into training, validation, and test sets.

**Dataset**

**Train Dataset**: /content/DATASET_PRJ/train

**Test Dataset**: /content/DATASET_PRJ/test

Images are resized to 224x224 pixels before feeding into the model.

**Model Architecture**

The CNN model consists of multiple convolutional and pooling layers followed by fully connected layers:

**Convolutional Layers**: 4 Conv2D layers with ReLU activation and Batch Normalization.

**Pooling Layers**: MaxPooling2D layers for dimensionality reduction.

**Dropout Layers**: Used to prevent overfitting.

**Global Average Pooling**: Reduces feature map dimensions.

**Dense Layers**: Fully connected layers with ReLU activation.

**Output Layer**: A single neuron with sigmoid activation for binary classification.

**Training and Evaluation**

The model is trained using Adam optimizer with binary cross-entropy loss.

ReduceLROnPlateau is used to adjust the learning rate dynamically.

Training and validation accuracy/loss are plotted for better visualization.

**Key Metrics**

The model evaluates performance using:

Accuracy

Precision

Recall (Sensitivity)

Specificity

F1-Score

Confusion Matrix

**Conclusion**

This project successfully implements a CNN-based model for autism classification with promising accuracy. Future improvements can include data augmentation, hyperparameter tuning, and transfer learning to enhance performance.
