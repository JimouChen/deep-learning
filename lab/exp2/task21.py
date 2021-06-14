import torch.utils.data
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
loss_value = []


# 画出loss曲线图
def draw_loss(loss_counter: list):
    train_counter = [_ for _ in range(len(loss_counter))]
    plt.figure()
    plt.plot(train_counter, loss_counter, color='red', label='loss')
    plt.xlabel('times od training')
    plt.ylabel('loss value')
    plt.title('Loss Curve')
    plt.show()


# 构建RNN模型
class RNNNet(nn.Module):

    def __init__(self):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)

        out = out.view(-1, hidden_size)
        out = self.liner(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


model = RNNNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 1e-2)
hidden_prev = torch.zeros(1, 1, hidden_size)

# 训练模型
for iter in range(6000):

    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        loss_value.append(loss)
        print("Iter: {} loss: {} ".format(iter, loss))

        prediction = []
        Input = x[:, 0, :]
        for _ in range(x.shape[1]):
            Input = Input.view(1, 1, 1)
            (pred, hidden_prev) = model(Input, hidden_prev)
            Input = pred
            prediction.append(pred.detach().numpy().ravel()[0])

        x = x.data.numpy().ravel()
        y = y.data.numpy()
        plt.scatter(time_steps[:-1], x.ravel(), s=90)
        plt.plot(time_steps[:-1], x.ravel(), label='real sin')
        plt.scatter(time_steps[1:], prediction, label='predict sin')
        plt.title('Iter Times: {}'.format(iter))
        plt.legend(loc='upper right')

        plt.show()

start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)  # 标准的sin
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

prediction = []
Input = x[:, 0, :]
for _ in range(x.shape[1]):
    Input = Input.view(1, 1, 1)
    (pred, hidden_prev) = model(Input, hidden_prev)
    Input = pred
    prediction.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], prediction)
plt.show()
draw_loss(loss_value)
