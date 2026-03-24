import numpy as np


class SoftmaxRegressionScratch:
    """
    Multi-class classification using Softmax + Gradient Descent
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        m, n = X.shape

        # 🔥 IMPORTANT FIX (convert labels to integer)
        y = y.astype(int)

        self.classes = len(np.unique(y))

        self.w = np.zeros((n, self.classes))
        self.b = np.zeros(self.classes)

        # One-hot encoding
        y_onehot = np.eye(self.classes)[y]

        self.losses = []

        for epoch in range(self.epochs):

            # Forward pass
            z = np.dot(X, self.w) + self.b
            probs = self.softmax(z)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
            self.losses.append(loss)

            # Backward pass
            dz = probs - y_onehot
            dw = (1 / m) * np.dot(X.T, dz)
            db = (1 / m) * np.sum(dz, axis=0)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                print(f"[Classification] Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)