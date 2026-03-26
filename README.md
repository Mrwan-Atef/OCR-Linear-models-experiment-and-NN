# OCR Linear Models and Neural Networks Experiment

## 📌 Project Overview
[cite_start]This project tackles the recognition of handwritten digits formulated as a multi-class classification problem, where each input image is assigned a label from the set {0, 1, ..., 9}[cite: 10]. 

[cite_start]The primary objective is to evaluate and compare different machine learning approaches under a unified preprocessing pipeline[cite: 27]. [cite_start]By doing so, the project analyzes how each model's theoretical assumptions align with the characteristics of image-based data [cite: 28][cite_start], highlighting the trade-offs between simplicity, interpretability, and predictive performance[cite: 29].

## 📊 Dataset
[cite_start]This work uses the **DIDA handwritten digit dataset (10k version)**[cite: 3].
* [cite_start]Contains approximately 10,000 labeled images of handwritten digits (0–9)[cite: 3].
* [cite_start]The raw images are RGB with varying spatial resolutions[cite: 4, 5, 6].

## ⚙️ Unified Preprocessing Pipeline (Feature Engineering)
[cite_start]Because raw images are not directly compatible with classical ML models [cite: 46][cite_start], all models in this study share a strict, unified preprocessing pipeline to ensure fair comparison[cite: 47].

1. [cite_start]**Image Resizing:** Images are resized to a fixed resolution of 28x28 pixels [cite: 50, 51] [cite_start]to provide a balance between information preservation and computational efficiency[cite: 52].
2. [cite_start]**Grayscale Conversion:** Images are converted from RGB to grayscale [cite: 59][cite_start], reducing input features by a factor of three [cite: 60] [cite_start]and focusing the model on shape and stroke intensity[cite: 61].
3. [cite_start]**Flattening:** The 2-D spatial arrangement is removed [cite: 69][cite_start], transforming the image into a 784-dimensional feature vector [cite: 64, 65] [cite_start]to enable the use of linear models and neural networks[cite: 67].
4. [cite_start]**Feature Scaling:** Standardization (zero mean, unit variance) is applied before training [cite: 77] [cite_start]to ensure stable, fast convergence for SGD-based models [cite: 82] [cite_start]and to prevent activation function saturation in networks[cite: 83].

[cite_start]*Note: The dataset is split into training, validation, and test sets [cite: 86] using stratified sampling to preserve class distributions[cite: 87].*

## 🧠 Investigated Models & Theoretical Analysis

### 1. Linear Regression (SGDRegressor)
* [cite_start]**Direct Regression:** Fails completely with a 13.5% accuracy (essentially random guessing)[cite: 114]. [cite_start]It predicts continuous values [cite: 112] [cite_start]and cannot capture the categorical nature of the problem [cite: 116][cite_start], suffering from very high bias[cite: 115].
* [cite_start]**One-Vs-All (OVA):** Fitting 10 separate binary regressors fixes the categorical target issue [cite: 118, 119, 123][cite_start], jumping the accuracy to 72.85%[cite: 126]. [cite_start]However, it remains limited because it cannot capture complex non-linear patterns like curved strokes[cite: 145].

### 2. Logistic Regression (SGDClassifier)
* [cite_start]Outputs class probabilities rather than numeric labels [cite: 164][cite_start], avoiding the target-representation problem[cite: 183].
* [cite_start]Utilizing cross-entropy loss focuses the model on correctly separating classes[cite: 165, 184], improving decision boundaries over standard Linear Regression. 
* [cite_start]Achieves a **75.90% test accuracy** [cite: 177] [cite_start]with stable, low-variance generalization[cite: 180, 181].

### 3. Gaussian Naive Bayes
* [cite_start]Serves as a highly computationally efficient baseline, training in just 4.45 seconds[cite: 226].
* [cite_start]Performs poorly with a **51.40% test accuracy**[cite: 226].
* [cite_start]**Why it fails:** It relies on a strong feature independence assumption[cite: 217]. [cite_start]In flattened images, adjacent pixels are highly correlated (forming strokes and edges) [cite: 224][cite_start], completely violating this assumption [cite: 223] [cite_start]and resulting in misestimated probabilities[cite: 234].

### 4. Multilayer Perceptron (MLP)
* [cite_start]A feedforward neural network that learns hierarchical, non-linear mappings from input features to class probabilities[cite: 258], making it highly suitable for image data.
* [cite_start]**Architecture:** 784 input neurons [cite: 276][cite_start], two hidden layers (512 and 256 units) [cite: 277, 308][cite_start], and a 10-neuron Softmax output layer[cite: 278].
* [cite_start]**Activation Function:** ReLU was chosen over Sigmoid and Tanh because it avoids the vanishing gradient problem [cite: 305][cite_start], allowing the model to converge the fastest (~80 iterations) [cite: 302] [cite_start]while achieving the best accuracy[cite: 302].
* [cite_start]Achieves the highest performance with an **89.10% test accuracy**[cite: 310].

## 🏆 Final Results Summary

| Model | Key Assumptions | Performance | Bias | Variance | Computational Cost | Final Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct Linear Regression** | [cite_start]Linear relationship, continuous output [cite: 337] | [cite_start]Very poor (≈13.5%) [cite: 337] | [cite_start]Very High [cite: 337] | [cite_start]Low [cite: 337] | [cite_start]Low [cite: 337] | [cite_start]Not suitable for classification [cite: 337] |
| **Linear Regression + OVA** | [cite_start]Linear separability per class [cite: 337] | [cite_start]Moderate (≈72.9%) [cite: 337] | [cite_start]High [cite: 337] | [cite_start]Low [cite: 337] | [cite_start]Medium [cite: 337] | [cite_start]Improves baseline but limited [cite: 337] |
| **Logistic Regression (OVA)** | [cite_start]Linear decision boundaries, probabilistic outputs [cite: 337] | [cite_start]Good (≈75.9%) [cite: 337] | [cite_start]Medium [cite: 337] | [cite_start]Low [cite: 337] | [cite_start]Medium [cite: 337] | [cite_start]Strong linear classifier [cite: 337] |
| **Gaussian Naive Bayes** | [cite_start]Pixel independence, Gaussian distribution [cite: 337] | [cite_start]Low (≈51.4%) [cite: 337] | [cite_start]High [cite: 337] | [cite_start]Very Low [cite: 337] | [cite_start]Very Low [cite: 337] | [cite_start]Fast but over simplistic [cite: 337] |
| **MLP (ReLU)** | [cite_start]Minimal assumptions, non-linear modeling [cite: 337] | [cite_start]**Best (≈89.1%)** [cite: 337] | [cite_start]Low [cite: 337] | [cite_start]Moderate [cite: 337] | [cite_start]Very High [cite: 337] | [cite_start]Best overall model [cite: 337] |

## 🚀 Conclusion
[cite_start]Model assumptions strictly dictate performance on image-based tasks[cite: 341]. [cite_start]Linear and probabilistic models suffer from restrictive assumptions [cite: 342] [cite_start]that do not align with the spatial and non-linear nature of images[cite: 342]. [cite_start]The MLP successfully relaxes these assumptions to learn complex hierarchical representations [cite: 343][cite_start], delivering the best overall performance at the cost of higher computational resources[cite: 343].
