# pytorch-blob-neural-network
Multi-class classification using PyTorch: Synthetic blob dataset generation, neural network architecture, model evaluation, and decision boundary visualization.




# PyTorch Multi-Class Classification 


###  Overview
This project demonstrates a fundamental multi-class classification neural network implemented in **PyTorch**. Using `scikit-learn`'s `make_blobs` function, a 2D synthetic dataset with 4 classes is generated. The project covers data preprocessing, building a linear neural network stack, training with Cross-Entropy Loss and SGD optimizer, and visualizing decision boundaries.

###  Features
* **Dataset:** 1,000 synthetic data samples with 4 distinct classes (`scikit-learn.datasets.make_blobs`).
* **Model Architecture:** Fully connected Neural Network with ReLU activation functions (`nn.Sequential`).
* **Loss Function & Optimizer:** `nn.CrossEntropyLoss` and `torch.optim.SGD`.
* **Device Agnostic:** Supports both CPU and CUDA (GPU) execution.
* **Visualization:** Custom decision boundary visualization using `matplotlib`.
* **Evaluation:** Model accuracy evaluated via standard accuracy metrics and `torchmetrics`.
