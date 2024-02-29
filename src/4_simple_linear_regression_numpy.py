import numpy as np

train_x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
     [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
     [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
train_y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
     [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
     [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

model_weight = 0.0
model_bias = 0.0
learning_rate = 0.005

for epoch in range(10000):
    model_out = model_weight * train_x + model_bias

    model_loss = ((train_y - model_out) ** 2).mean()

    model_weight = model_weight - learning_rate * ((model_out - train_y) * train_x).mean()
    model_bias = model_bias - learning_rate * (model_out - train_y).mean()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Weight : {model_weight:.3f}, Bias : {model_bias:.3f}, Cost : {model_loss:.3f}")
