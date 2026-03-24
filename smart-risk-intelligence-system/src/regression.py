print("USING CORRECT regression.py FILE")
import numpy as np


class LinearRegressionScratch:
    """
    Simple Linear Regression using Gradient Descent
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape

        # initialize parameters
        self.w = np.zeros(n)
        self.b = 0

        self.losses = []

        for epoch in range(self.epochs):

            # prediction
            y_pred = np.dot(X, self.w) + self.b

            # error
            error = y_pred - y

            # gradients
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # loss (MSE)
            loss = np.mean(error ** 2)
            self.losses.append(loss)

            # optional debug print
            if epoch % 200 == 0:
                print(f"[Regression] Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.w) + self.b