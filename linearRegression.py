import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    independent_path = 'data/linearX.csv'
    dependent_path = 'data/linearY.csv'

    X = pd.read_csv(independent_path, header=None).values
    Y = pd.read_csv(dependent_path, header=None).values
    Y = Y.reshape(-1, 1)
    return X, Y

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

def add_intercept(X):
    return np.c_[np.ones(X.shape[0]), X]

def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - Y) ** 2)
    return cost

def gradient_descent(X, Y, theta, alpha, num_iters):
    m = len(Y)
    cost_history = []

    for i in range(num_iters):
        prediction = X.dot(theta)
        error = prediction - Y
        theta -= (alpha / m) * X.T.dot(error)

        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < 1e-6:
            break

    return theta, cost_history

X, Y = load_data()
X = normalize_data(X)
X = add_intercept(X)

theta_initial = np.zeros((X.shape[1], 1))
alpha = 0.5
num_iters = 1000

theta_optimal, cost_history = gradient_descent(X, Y, theta_initial, alpha, num_iters)

plt.plot(cost_history[:50])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function vs Iterations')
plt.show()

plt.scatter(X[:, 1], Y, color='blue', label='Data')
plt.plot(X[:, 1], X.dot(theta_optimal), color='red', label='Fitted line')
plt.xlabel('Predictor')
plt.ylabel('Response')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

for lr in [0.005, 0.5, 5]:
    theta_optimal_lr, cost_history_lr = gradient_descent(X, Y, theta_initial, lr, num_iters)
    plt.plot(cost_history_lr[:50], label=f'lr={lr}')

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function for Different Learning Rates')
plt.legend()
plt.show()
