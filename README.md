# 程式碼解釋
1. 建立資料
```python=
x_raw = np.linspace(-50, 50, 200).reshape(-1, 1)
x = (x_raw - np.mean(x_raw)) / np.std(x_raw)
y = 5*x + 1 + np.random.randn(200, 1) * 0.3

```
- `x_raw`：原始輸入，範圍從 -50 到 50，總共 200 個點
- `x`：標準化後的輸入，使平均值為 0、標準差為 1這樣神經網路訓練更穩定，不會出現梯度爆炸或消失
- `y`：目標輸出（真實資料）線性函數 5*x + 1
- 加上高斯噪聲 `np.random.randn(...) * 0.3` 模擬真實資料的隨機性

2. 定義神經網路結構與學習率
```python=
layer_sizes = [1, 64, 64, 1]
lr = 0.01
```

- `[1, 64, 64, 1]` 表示1個輸入神經元（`x`）-> 兩層隱藏層，每層64個神經元 -> 1個輸出神經元（`y_pred`）
- `lr`：初始學習率（learning rate）

3. 初始化權重與偏置
```python=
weights = []
biases = []
for i in range(len(layer_sizes) - 1):
    w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
    b = np.zeros((1, layer_sizes[i+1]))
    weights.append(w)
    biases.append(b)
```

- `weights`：每層的權重矩陣，大小 (上一層神經元數, 下一層神經元數)
- `biases`：每層的偏置向量，初始化為 0
- 權重初始化方式為 **He初始化**，適合 tanh 或 ReLU 激活函數，公式為 `np.sqrt(2 / fan_in)`，讓輸入方差保持穩定

4. 定義激活函數
```python=
def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2
```
- `tanh(x)`：雙曲正切函數，輸出範圍 (-1, 1)
- `tanh_derivative(x)`：tanh 的導數，用於反向傳播計算梯度

5. 訓練迴圈
```python=
loss = 1
epoch = 0

while loss>0.1:
    ...
```

5-1. 前向傳播（Forward Pass）
```python=
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
```

- `z = a.dot(w) + b`：線性組合（加權和 + 偏置）
- `a = tanh(z)`：激活函數，加入非線性
- 最後一層不加激活函數（因為是回歸問題）
- `activations`：儲存每層的輸出，用於反向傳播
- `zs`：儲存每層的線性輸入（`z`）

5-2. 計算損失（Loss）
```python=
loss = np.mean((y_pred - y)**2)
```

- 使用高中教的均方差 $MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$ 計算 Loss

5-3. 反向傳播（Backward Pass）
```python=
dy = 2 * (y_pred - y) / len(y)
deltas = [dy]

for i in reversed(range(len(layer_sizes) - 2)):
    dz = deltas[-1].dot(weights[i+1].T) * tanh_derivative(zs[i])
    deltas.append(dz)
deltas.reverse()
```
- 計算每層的 誤差項 `delta`
- 從輸出層往輸入層反傳
- 使用鏈式法則乘上 `tanh_derivative(z)`

5-4. 更新權重、偏置跟學習率
```python=
for i in range(len(weights)):
    dw = activations[i].T.dot(deltas[i])
    db = np.sum(deltas[i], axis=0, keepdims=True)
    weights[i] -= lr * dw
    biases[i] -= lr * db
    
lr = 0.01 * (0.99 ** (epoch // 100))
```

- 計算每層權重梯度 dw 和偏置梯度 db
- 使用梯度下降更新權重與偏置
- 每 100 個 epoch，學習率衰減 1%，讓訓練後期更穩定

6. 預測（Forward Pass）
```python=
a = x
for w, b in zip(weights[:-1], biases[:-1]):
    a = tanh(a.dot(w) + b)
y_pred = a.dot(weights[-1]) + biases[-1]
```
- 使用訓練好的權重做預測
- 與訓練時前向傳播相同

7. 畫圖
```python=
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
```
