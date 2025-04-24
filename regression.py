import numpy as np
import matplotlib.pyplot as plt


# Logistic Regression implementation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_proba(w, X):
    return sigmoid(X.dot(w))


def compute_loss(w, X, y, lam):
    predictions = predict_proba(w, X)
    epsilon = 1e-15  # To avoid log(0) errors
    log_loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    reg_term = 0.5 * lam / X.shape[0] * np.sum(w ** 2)
    return log_loss + reg_term


def train_logistic_regression(w_init, X, y, lam=0.0001, lr=0.05, epochs=500, tol=1e-6):
    N, d = X.shape
    w = w_init.copy()
    loss_hist = [compute_loss(w, X, y, lam)]

    for epoch in range(epochs):
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi, yi = X[i], y[i]
            ai = sigmoid(xi.dot(w))
            w -= lr * ((ai - yi) * xi + lam * w)

        current_loss = compute_loss(w, X, y, lam)
        loss_hist.append(current_loss)

        if np.linalg.norm(loss_hist[-2] - current_loss) < tol:
            break

    return w, loss_hist


# Data preparation
def prepare_data(seed=2):
    np.random.seed(seed)
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

    X_with_bias = np.c_[X, np.ones(X.shape[0])]
    return X_with_bias, X, y


# Plot the logistic regression result
def plot_logistic_regression(X_raw, y, w):
    plt.figure(figsize=(8, 6))

    # Plot dataset
    plt.plot(X_raw[y == 0], y[y == 0], 'ro', markersize=6, label="Fail (y=0)")
    plt.plot(X_raw[y == 1], y[y == 1], 'bs', markersize=6, label="Pass (y=1)")

    # Draw prediction probability curve
    xx = np.linspace(0, 6, 500)
    yy = sigmoid(w[0] * xx + w[1])
    plt.plot(xx, yy, 'g-', linewidth=2, label="Probability Curve")

    # Threshold line at probability = 0.5
    threshold = -w[1] / w[0]
    plt.plot(threshold, 0.5, 'y^', markersize=10, label="Decision Threshold (0.5 probability)")
    plt.axhline(y=0.5, color='grey', linestyle='--')  # visualize threshold line horizontally

    plt.title("Logistic Regression Prediction: Exam Pass or Fail")
    plt.xlabel("Studying Hours")
    plt.ylabel("Predicted Pass Probability")
    plt.axis([-1, 7, -0.2, 1.2])
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()


# Main execution block
def main():
    X, X_raw, y = prepare_data()
    w_init = np.random.randn(X.shape[1])

    w, loss_hist = train_logistic_regression(w_init, X, y)

    print('Optimized weights:', w)
    print('Final Loss:', loss_hist[-1])

    plot_logistic_regression(X_raw, y, w)


if __name__ == "__main__":
    main()
