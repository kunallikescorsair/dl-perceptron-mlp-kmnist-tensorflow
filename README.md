⚠️ Academic Integrity Notice:
This repository is shared publicly for learning and demonstration purposes only. Unauthorized copying or direct submission of this work for academic credit is strictly prohibited and may be considered academic misconduct.

# 🧠 Neural Networks – Perceptron and Custom MLPs on KMNIST

This project explores building a Perceptron from scratch and training multiple fully connected neural network models on the KMNIST dataset.  
Developed as part of the **Deep Learning (94691)** course for the **Master of Data Science and Innovation** at **UTS**.

## 📄 Project Overview
- **Part A:** Implemented a multi-class Perceptron from scratch using only NumPy.
- **Part B:** Trained three custom MLP architectures in PyTorch to classify handwritten Hiragana characters from the KMNIST dataset.
- Evaluated the effect of model depth, neuron counts, and dropout rates on accuracy and generalization.

## 📚 Dataset
- **KMNIST Dataset:** 70,000 grayscale images (28×28) of 10 Hiragana characters.
- **Preprocessing:** Flattened images, normalized pixel values, and encoded class labels.

## 🛠️ Methods
- **Perceptron (Part A):** Manually implemented forward and backward propagation.
- **MLPs (Part B Experiments):**
  - Experiment 1: 2-layer MLP (256→128) with 20% dropout.
  - Experiment 2: 3-layer deeper MLP (512→256→128) with higher dropout.
  - Experiment 3: Minimal 2-layer MLP (128→64) with strong 50% dropout.
- **Training:** Adam optimizer, CrossEntropyLoss, fixed hyperparameters (learning rate = 0.001, batch size = 128, epochs = 500).

## 🔥 Key Results
- **Best Test Accuracy:** 92.51% (Experiment 2 – Deeper Architecture).
- Deeper networks improved generalization; excessive dropout in smaller models caused underfitting.
- Loss curves and confusion matrices indicated stable training without overfitting spikes.

## 📈 Insights
- Depth and moderate dropout improve MLP performance on complex handwriting datasets.
- Heavy regularization severely limits model capacity on image tasks without convolutional layers.

## 📌 Technologies Used
- Python (NumPy, PyTorch)
- Jupyter Notebook
- Matplotlib for visualization

## 🚀 Future Improvements
- Extend to CNN architectures for spatial feature extraction.
- Apply data augmentation to improve robustness.
- Implement early stopping and hyperparameter tuning.

## 📜 Acknowledgements
This project was completed individually for academic purposes under the **Master of Data Science and Innovation** at **UTS**.  
All code and analysis are original work by **Kunal Gurung**.
