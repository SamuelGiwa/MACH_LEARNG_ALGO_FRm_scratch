import numpy as np

# Sigmoid function
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

# Stochastic Gradient Descent (SGD) update function
def DeltaSGD(W, X, D):
    alpha = 0.9  # Learning rate
    N = X.shape[0]  

    for k in range(N):
        x = X[k, :].reshape(-1, 1) 
        d = D[k]  

        v = np.dot(W, x)  # Dot product to compute the weighted sum
        y = sigmoid(v)    # Applying sigmoid to get output

        e = d - y  # Error term
        delta = y * (1 - y) * e  # applying the Delta rule

        dW = alpha * delta * x  # Update rule

        # Update weights
        W[0] = W[0] + dW[0, 0]
        W[1] = W[1] + dW[1, 0]
        W[2] = W[2] + dW[2, 0]

    return W

# Input data (X) and desired output (D)
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([0, 0, 1, 1])

# Initialize weights randomly between -1 and 1
W = 2 * np.random.rand(3) - 1

# Training loop
for epoch in range(10000):
    W = DeltaSGD(W, X, D)

# Inference loop
N = 4  # Number of samples to test
for k in range(N):
    x = X[k, :].reshape(-1, 1)  # Input vector
    v = np.dot(W, x)  # Weighted sum
    y = sigmoid(v)  # Output
    print(f"Sample {k + 1}: Predicted Output = {y[0]}")
