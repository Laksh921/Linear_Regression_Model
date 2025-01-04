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

# Plot the first figure (Cost Function vs Iterations)
plt.plot(cost_history[:50])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function vs Iterations')
plt.show()

# Plot the second figure (Data + Fitted Line)
plt.scatter(X[:, 1], Y, color='blue', label='Data')
plt.plot(X[:, 1], X.dot(theta_optimal), color='red', label='Fitted line')
plt.xlabel('Predictor')
plt.ylabel('Response')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Trying three different learning rates and plotting each result separately
learning_rates = [0.005, 0.5, 5]

# Create subplots for each learning rate (Cost Function vs Iterations)
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

for i, lr in enumerate(learning_rates):
    # Run gradient descent for the current learning rate
    theta_optimal_lr, cost_history_lr = gradient_descent(X, Y, theta_initial, lr, num_iters=1000)

    # Plot cost history for each learning rate on the respective subplot
    axs[i].plot(cost_history_lr[:50])  # Plot only the first 50 iterations
    axs[i].set_title(f'Cost Function for lr={lr}')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Cost')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# Print optimal parameters for the learning rates (You can include them in the report)
for lr in learning_rates:
    theta_optimal_lr, cost_history_lr = gradient_descent(X, Y, theta_initial, lr, num_iters=1000)
    print(f"Optimal parameters for lr={lr}:")
    print(f"θ₀ = {theta_optimal_lr[0]}")
    print(f"θ₁ = {theta_optimal_lr[1]}")
    print(f"Final cost function value: {cost_history_lr[-1]}\n")


def batch_gradient_descent(X, Y, theta, alpha, num_iters):
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


def stochastic_gradient_descent(X, Y, theta, alpha, num_iters):
    m = len(Y)
    cost_history = []

    for i in range(num_iters):
        for j in range(m):
            rand_index = np.random.randint(m)
            X_rand = X[rand_index:rand_index + 1]
            Y_rand = Y[rand_index:rand_index + 1]
            prediction = X_rand.dot(theta)
            error = prediction - Y_rand
            theta -= alpha * X_rand.T.dot(error)

        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < 1e-6:
            break

    return theta, cost_history


def mini_batch_gradient_descent(X, Y, theta, alpha, num_iters, batch_size=32):
    m = len(Y)
    cost_history = []
    for i in range(num_iters):
        # Shuffle the data for each iteration
        shuffle_index = np.random.permutation(m)
        X_shuffled = X[shuffle_index]
        Y_shuffled = Y[shuffle_index]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j + batch_size]
            Y_batch = Y_shuffled[j:j + batch_size]

            prediction = X_batch.dot(theta)
            error = prediction - Y_batch
            theta -= (alpha / batch_size) * X_batch.T.dot(error)

        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < 1e-6:
            break

    return theta, cost_history


# Load and preprocess data
X, Y = load_data()
X = normalize_data(X)
X = add_intercept(X)

theta_initial = np.zeros((X.shape[1], 1))
alpha = 0.5
num_iters = 1000

# Run Batch Gradient Descent
theta_optimal_batch, cost_history_batch = batch_gradient_descent(X, Y, theta_initial, alpha, num_iters)

# Run Stochastic Gradient Descent
theta_optimal_sgd, cost_history_sgd = stochastic_gradient_descent(X, Y, theta_initial, alpha, num_iters)

# Run Mini-Batch Gradient Descent
theta_optimal_mini_batch, cost_history_mini_batch = mini_batch_gradient_descent(X, Y, theta_initial, alpha, num_iters,
                                                                                batch_size=32)

# Plot the cost function for each method independently
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Batch Gradient Descent Cost
axs[0].plot(cost_history_batch[:50])  # Plot first 50 iterations
axs[0].set_title('Cost Function for Batch Gradient Descent')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Cost')

# Stochastic Gradient Descent Cost
axs[1].plot(cost_history_sgd[:50])  # Plot first 50 iterations
axs[1].set_title('Cost Function for Stochastic Gradient Descent')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Cost')

# Mini-Batch Gradient Descent Cost
axs[2].plot(cost_history_mini_batch[:50])  # Plot first 50 iterations
axs[2].set_title('Cost Function for Mini-Batch Gradient Descent')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Cost')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# Print optimal parameters and final cost function values for each method
print("Batch Gradient Descent:")
print(f"Optimal parameters θ₀ = {theta_optimal_batch[0]}, θ₁ = {theta_optimal_batch[1]}")
print(f"Final cost: {cost_history_batch[-1]}\n")

print("Stochastic Gradient Descent:")
print(f"Optimal parameters θ₀ = {theta_optimal_sgd[0]}, θ₁ = {theta_optimal_sgd[1]}")
print(f"Final cost: {cost_history_sgd[-1]}\n")

print("Mini-Batch Gradient Descent:")
print(f"Optimal parameters θ₀ = {theta_optimal_mini_batch[0]}, θ₁ = {theta_optimal_mini_batch[1]}")
print(f"Final cost: {cost_history_mini_batch[-1]}\n")
