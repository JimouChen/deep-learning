from multiprocessing.dummy import freeze_support
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import copy
import torch.nn.functional as F

MINI_BATCH = 8  # 数据集的图片数量很大，无法一次性加载所有数据，所以一次加载一个mini-batch的图片
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU可用则使用GPU
train_losses = []  # 记录训练时的loss变化

# ToTensor(): 将ndarrray格式的图像转换为Tensor张量
# Normalize(mean, std) mean：每个通道颜色平均值，这里的平均值为0.5，私人数据集自己计算；std：每个通道颜色标准偏差，(原始数据 - mean) / std 得到归一化后的数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练数据加载
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=MINI_BATCH, shuffle=True, num_workers=4)
# 测试数据加载
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=MINI_BATCH, shuffle=False, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层：3通道到6通道，卷积5*5
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层：6通道到16通道，卷积5*5

        self.pool = nn.MaxPool2d(2, 2)  # 池化层，在2*2窗口上进行下采样

        # 三个全连接层 ：16*5*5 -> 120 -> 84 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义数据流向
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 加入relu激活函数
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 变换数据维度为 1*(16*5*5)，-1表示根据后面推测
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, criterion, optimizer, epochs):
    since = time.time()

    best_acc = 0.0  # 记录模型测试时的最高准确率
    best_model_wts = copy.deepcopy(model.state_dict())  # 记录模型测试出的最佳参数

    for epoch in range(epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # 训练模型
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播，计算损失
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每1000批图片打印训练数据
            if (i != 0) and (i % 1000 == 0):
                print('step: {:d},  loss: {:.3f}'.format(i, running_loss / 1000))
                train_losses.append('{:.3f}'.format(running_loss / 1000))  # 记录loss
                running_loss = 0.0

        # 每个epoch以测试数据的整体准确率为标准测试一下模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        if acc > best_acc:  # 当前准确率更高时更新
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('最高准确率: {}%'.format(100 * best_acc))

    # 返回测试出的最佳模型
    model.load_state_dict(best_model_wts)
    return model


# 画出训练时的loss变化曲线
def draw_loss():
    train_counter = [i * 1000 for i in range(1, len(train_losses) + 1)]
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('times of training')
    plt.ylabel('negative log likelihood loss')
    plt.show()


net = Net()
net.to(DEVICE)

# 使用分类交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器使用随机梯度下降SGD，并设置学习率和角动量
optimizer = optim.SGD(net.parameters(), lr=0.0018, momentum=0.9)

# 训练20个epoch
if __name__ == '__main__':
    freeze_support()
    net = train(net, criterion, optimizer, 20)
    draw_loss()
# 保存模型参数
# torch.save(net.state_dict(), 'net_dict.pt')
