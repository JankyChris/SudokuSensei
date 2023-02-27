import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__ == "__main__":
    batch_size = 256
    learning_rate = 0.01
    n_epochs = 24

    """
    train_dataset = torchvision.datasets.MNIST("./mnist_data", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST("./mnist_data", train=False, transform=transforms.ToTensor(), download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    """
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dataset = ImageFolder("./printed_digit_dataset/assets", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    data=iter(train_loader)
    samples, labels=next(data)
    print(f"number of samples {samples.shape}")
    print(f"number of labels {labels.shape}")
    plt.figure(figsize=(10,8))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(samples[i][0],cmap='gray')
    plt.show()

    network = Net()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [ {batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                train_losses.append(loss.item())
                train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                torch.save(network.state_dict(), './model_results/model.pth')
                torch.save(optimizer.state_dict(), './model_results/optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()