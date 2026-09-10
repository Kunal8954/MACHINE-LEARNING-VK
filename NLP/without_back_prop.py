import numpy as np

# -------------------------
# Sigmoid Function
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# Input
# -------------------------
X = np.array([[1, 2]])

# -------------------------
# Fixed Weights
# -------------------------
W = np.array([
    [0.5],
    [0.3]
])

# -------------------------
# Bias
# -------------------------
b = np.array([[0.1]])

# -------------------------
# Forward Propagation
# -------------------------

z = np.dot(X, W) + b

prediction = sigmoid(z)

print("Weighted Sum (z):")
print(z)

print()

print("Prediction:")
print(prediction)