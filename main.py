import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
x = np.linspace(-2, 2, 200).reshape(-1, 1)

y = 5*x + 1 + np.random.randn(200, 1) * 0.3

# 二次方程式 
# y2 = 2*x**2 - 3*x + 1 + np.random.randn(200, 1) * 0.3
# 三次方程式 
# y3 = x**3 - 4*x**2 + 2*x + 0.5 + np.random.randn(200, 1) * 0.3
# 四次方程式
# y4 = x**4 - 2*x**3 + 3*x**2 - x + 1 + np.random.randn(200, 1) * 0.3
# 五次方程式 
# y5 = x**5 - 3*x**4 - 5*x**3 + 2*x**2 - x + 1 + np.random.randn(200, 1) * 0.3

# structure
layer_sizes = [1, 32, 1]
lr = 0.01

# initialize
weights = []
biases = []
for i in range(len(layer_sizes) - 1):
    w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
    b = np.random.randn(1, layer_sizes[i+1]) * 0.01
    weights.append(w)
    biases.append(b)

# 激活函數
# 這裡我用 tanh
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

# render
plt.figure(figsize=(8,5))
plt.scatter(x, y, s=10, label="True data", alpha=0.6)

# 預測
a = x
for w, b in zip(weights[:-1], biases[:-1]):
    a = tanh(a.dot(w) + b)
y_pred = a.dot(weights[-1]) + biases[-1]

plt.plot(x, y_pred, color="red", linewidth=2, label="NN prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("NNs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("nn_predict.png", dpi=300)
