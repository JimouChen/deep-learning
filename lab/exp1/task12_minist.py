import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 512  # 设置批次大小
EPOCHS = 20  # 总共训练批次
# 如果有GPU就使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses = []  # 保存loss便于画图
test_losses = []

# 下载和加载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1037,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1037,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28，定义两个卷积层和两个全连接层
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)  # 加入relu函数
        out = F.max_pool2d(out, 2, 2)  # 使用最大池化
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)  # 1 * 2000，改变尺寸
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)  # 加入softmax转化为概率输出
        return out


# 生成模型和优化器
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)
        loss = F.nll_loss(output, target)  # 计算loss
        loss.backward()  # 计算梯度
        optimizer.step()  # 修改权值
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())  # 记录训练的loss值


# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(float('{:.4f}'.format(test_loss)))

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


# 画出loss变化曲线
def draw_loss():
    train_counter = [i * 12000 for i in range(1, len(train_losses) + 1)]
    test_counter = [i * 36000 for i in range(0, len(test_losses))]
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red', s=15)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('times of training')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# 最后开始训练和测试
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

draw_loss()
print(test_losses)
print(list(test_losses))
print(train_losses)
