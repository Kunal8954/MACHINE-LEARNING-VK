import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(output):
    return output*(1-output)

X=np.array([[1,2]])
y=np.array([[1]])

W=np.array([[0.5],[0.3]])
b=np.array([[0.1]])

learning_rate=0.1

for epoch in range(1000):

    # Forward
    z=np.dot(X,W)+b
    prediction=sigmoid(z)

    # Loss
    error=y-prediction
    loss=np.mean(error**2)

    # dL/dŷ
    dL_dyhat = -2 * (y - prediction)

    # dŷ/dz
    dyhat_dz = prediction * (1 - prediction)

    # δ
    delta = dL_dyhat * dyhat_dz

    # dL/dW
    dL_dW = np.dot(X.T, delta)  # that X.T is actually a dz_dw

    # dL/db
    dL_db = np.sum(delta, axis=0, keepdims=True)

    # Gradient Descent
    W -= learning_rate * dL_dW
    b -= learning_rate * dL_db

    if epoch%100==0:
        print(epoch, prediction, loss)

print("\nFinal Prediction")
print(prediction)
print("\nWeights")
print(W)
print("\nBias")
print(b)
