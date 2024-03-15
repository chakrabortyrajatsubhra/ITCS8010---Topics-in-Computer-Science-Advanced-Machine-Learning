import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Data Preparation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# Create a validation split
validation_split = 0.2
dataset_size = len(trainset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=train_sampler, num_workers=2)
valloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=val_sampler, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
# Define depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# Model Architecture with increased size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Increase the number of filters in the convolutional layers
        self.conv1 = DepthwiseSeparableConv(3, 128, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(128, 256, 3, padding=1)
        self.conv3 = DepthwiseSeparableConv(256, 512, 3, padding=1)
        self.conv4 = DepthwiseSeparableConv(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adding one more fully connected layer and more neurons
        self.fc1 = nn.Linear(512 * 2 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)  # Flatten the layer
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

net = Net()
net.to(device)
# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
epoch_count = []
training_losses = []
validation_losses = []
validation_accuracies = []
for epoch in range(10):
    net.train()  # Set model to training mode
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Compute validation loss and accuracy
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    net.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Append metrics for this epoch
    epoch_count.append(epoch + 1)
    training_losses.append(running_loss / len(trainloader))
    validation_losses.append(val_loss / len(valloader))
    validation_accuracies.append(100 * val_correct / val_total)

    # Print metrics for this epoch
    print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(trainloader)}, Val Loss: {val_loss / len(valloader)}, Val Accuracy: {100 * val_correct / val_total}%')    


    
plt.figure(figsize=(12, 6))

# Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_losses, label='Training Loss')
plt.plot(epoch_count, validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epoch_count, validation_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
# Evaluation
net.eval()  # Set the model to evaluation mode
correct = 0
total = 0
y_true = []
y_pred = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


print(f'Accuracy: {100 * correct / total}%')
print(f'Precision: {precision_score(y_true, y_pred, average="macro")}')
print(f'Recall: {recall_score(y_true, y_pred, average="macro")}')
print(f'F1-score: {f1_score(y_true, y_pred, average="macro")}')

# Calculate the number of parameters for Net
net_params = sum(p.numel() for p in net.parameters())
print(f'Number of parameters (Net): {net_params}')    