# Probabilistic Artificial Intelligence – ETH Zürich

This repository contains the programming assignments and projects for the **Probabilistic Artificial Intelligence** course at ETH Zürich. 

The implementations focus on fundamental concepts in probabilistic machine learning, specifically targeting uncertainty quantification, Bayesian inference, and sequential decision-making. All tasks are implemented in Python.

## Repository Structure & Tasks

### [Task 1: Gaussian Process Regression](./task1_handout_d3d63876)
**Objective:** Scale spatial inference and predictions using Gaussian Processes (GPs).
* **Description:** This task involves implementing a GP regression model to solve a large-scale, real-world regression problem (such as predicting environmental pollution levels). Because exact GP inference is computationally intractable for massive datasets O(N^3), the implementation relies on scalable approximation techniques—specifically using inducing points and the Nyström approximation—to ensure efficient computation without sacrificing the quality of the predictive uncertainty.

### [Task 2: Bayesian Neural Networks (BNNs)](./task2_handout_e14a688d)
**Objective:** Classification with reliable uncertainty estimation.
* **Description:** Standard neural networks lack the ability to express "I don't know." This project bridges that gap by implementing a Bayesian Neural Network. By applying a probabilistic interpretation to the network weights, the model is trained using a loss function that balances a Cross-Entropy term (for predictive accuracy) and a Kullback–Leibler (KL) divergence term (to regularize the learned posterior). The model successfully outputs probability distributions over classes, making it highly robust when evaluating out-of-distribution or rotated image data (e.g., MNIST variants).

### [Task 3: Constrained Bayesian Optimization](./task3_handout)
**Objective:** Black-box function optimization under strict constraints.
* **Description:** This task revolves around tuning hyperparameters for an expensive, black-box system where some regions of the search space are "unsafe" or violate specific constraints. The project implements a Bayesian Optimization loop utilizing a surrogate GP model. It actively balances exploration and exploitation by maximizing acquisition functions like Expected Improvement (EI) and Upper Confidence Bound (UCB), while simultaneously modeling the probability of satisfying the constraints.

## Technologies & Libraries
* **Language:** Python 3.x
* **Core Libraries:** PyTorch / TensorFlow, NumPy, SciPy, Scikit-Learn
* **Concepts:** Gaussian Processes, Variational Inference, Bayesian Optimization, Expected Improvement, Nyström Approximation.

## How to Run
Navigate into the respective task folder and follow the specific instructions provided within the notebooks or Python scripts. Most tasks contain a main script (e.g., `solution.py`) that can be executed directly to train the model and generate the predictions required for the grading system.

```bash
# Example for Task 1
cd task1_handout_d3d63876
python solution.py
