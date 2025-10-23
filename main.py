import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
x_raw = np.linspace(-50, 50, 200).reshape(-1, 1)
x = (x_raw - np.mean(x_raw)) / np.std(x_raw)

y = 5*x + 1 + np.random.randn(200, 1) * 0.3
# nn_quadratic_equation_predict
# y = 2*x**2 - 3*x + 1 + np.random.randn(200, 1) * 0.3
# nn_cubic_equation_predict
# y = x**3 - 4*x**2 + 2*x + 0.5 + np.random.randn(200, 1) * 0.3
# nn_quartic_equation_predict
# y = x**4 - 2*x**3 + 3*x**2 - x + 1 + np.random.randn(200, 1) * 0.3
# nn_quintic_equation_predict
# y = x**5 - 3*x**4 - 5*x**3 + 2*x**2 - x + 1 + np.random.randn(200, 1) * 0.3

# structure
layer_sizes = [1, 64, 64, 1]
lr = 0.01

# initialize
weights = []
biases = []
for i in range(len(layer_sizes) - 1):
    w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
    b = np.zeros((1, layer_sizes[i+1]))
    weights.append(w)
    biases.append(b)

# 激活函數 tanh
def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

loss = 1
epoch = 0

while loss>0.1:
    # forward
    activations = [x]
    zs = []
    a = x
    for w, b in zip(weights[:-1], biases[:-1]):
        z = a.dot(w) + b
        zs.append(z)
        a = tanh(z)
        activations.append(a)
    
    z = activations[-1].dot(weights[-1]) + biases[-1]
    zs.append(z)
    y_pred = z
    
    loss = np.mean((y_pred - y)**2)
    
    # backward
    dy = 2 * (y_pred - y) / len(y)
    deltas = [dy]
    
    for i in reversed(range(len(layer_sizes) - 2)):
        dz = deltas[-1].dot(weights[i+1].T) * tanh_derivative(zs[i])
        deltas.append(dz)
    deltas.reverse()
    
    for i in range(len(weights)):
        dw = activations[i].T.dot(deltas[i])
        db = np.sum(deltas[i], axis=0, keepdims=True)
        weights[i] -= lr * dw
        biases[i] -= lr * db
    
    lr = 0.01 * (0.99 ** (epoch // 100))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss={loss:.5f}")


# 預測
a = x
for w, b in zip(weights[:-1], biases[:-1]):
    a = tanh(a.dot(w) + b)
y_pred = a.dot(weights[-1]) + biases[-1]

# render
plt.figure(figsize=(8,5))
plt.scatter(x_raw, y, s=10, label="True data", alpha=0.6)
plt.plot(x_raw, y_pred, color="red", linewidth=2, label="NN prediction")
plt.xlabel("x (original scale)")
plt.ylabel("y")
plt.legend()
plt.title("Neural Network Fit (x ∈ [-50, 50])")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("nn_predict.png", dpi=300)