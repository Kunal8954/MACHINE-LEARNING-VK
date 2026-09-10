import numpy as np

# ----------------------------
# Activation Function
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ----------------------------
# Dataset
# ----------------------------
X = np.array([[1, 2]])      # One training example
y = np.array([[1]])         # Expected output

# ----------------------------
# Fixed Weights and Biases
# ----------------------------
W1 = np.array([
    [0.5, 0.2],
    [0.3, 0.4]
])

b1 = np.array([[0.1, 0.1]])

W2 = np.array([
    [0.6],
    [0.7]
])

b2 = np.array([[0.2]])

learning_rate = 0.1
epochs = 10

# ----------------------------
# Training
# ----------------------------
for epoch in range(epochs):

    # -------- Forward Propagation --------

    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, W2) + b2
    prediction = sigmoid(output_input)

    # -------- Loss --------

    error = y - prediction
    loss = np.mean(error ** 2)

    # -------- Backpropagation --------

    d_output = error * sigmoid_derivative(prediction)

    hidden_error = np.dot(d_output, W2.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    # -------- Weight Update --------

    W2 += learning_rate * np.dot(hidden_output.T, d_output)
    b2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    W1 += learning_rate * np.dot(X.T, d_hidden)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # -------- Print Everything --------

    print("=" * 50)
    print(f"Epoch {epoch + 1}")

    print("\nHidden Input:")
    print(hidden_input)

    print("\nHidden Output:")
    print(hidden_output)

    print("\nOutput Input:")
    print(output_input)

    print("\nPrediction:")
    print(prediction)

    print("\nLoss:")
    print(loss)

    print("\nUpdated W1:")
    print(W1)

    print("\nUpdated W2:")
    print(W2)

print("=" * 50)
print("Training Completed")
print("Final Prediction:")
print(prediction)


