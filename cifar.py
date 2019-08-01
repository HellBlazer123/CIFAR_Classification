
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import os.path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
parser.add_argument('--ckptroot', type=str, default="../checkpoint/checkpoint.t7", help='path to checkpoint')

parser.add_argument('--alpha', type=float, default=0.001, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size_train', type=int, default=128, help='training set input batch size')
parser.add_argument('--batch_size_test', type=int, default=64, help='test set input batch size')

parser.add_argument('--resume', type=bool, default=False, help='whether training from ckpt')
parser.add_argument('--is_gpu', type=bool, default=True, help='whether training using GPU')

args = parser.parse_args()

print(">>> Preparing Data ...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4384, 0.4812, 0.4555), (0.2093, 0.1874, 0.2014)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4384, 0.4812, 0.4555), (0.2093, 0.1874, 0.2014)),
])

print(">>> Preparing CIFAR10 dataset ...")

trainset = torchvision.datasets.CIFAR10( root = args.dataroot, train = True, download = True, transform=transform_train )
trainloader = torch.utils.data.DataLoader( trainset, batch_size = args.batch_size_train, shuffle = True, num_workers = 2 )

testset = torchvision.datasets.CIFAR10(root = args.dataroot, train = False, download = True, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size_test, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

kernel_size = 3
padding = 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, sample):
        sample = self.conv_layer(sample)
        sample = sample.view(x.size(0), -1)
        sample = self.fc_layer(sample)
        return sample


print(">>> Initializing CNN model ...")

start_epoch = 0

if args.resume:
    print('>>> Resuming from checkpoint ...')
    assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.ckptroot)
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    print('>>> Building new CNN model ...')
    net = CNN()

if args.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), alpha=args.alpha, weight_decay=args.decay)

def calculate_accuracy(loader, is_gpu):
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        temp, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


print(">>> Start training ...")

for epoch in range(start_epoch, args.epochs + start_epoch):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        if args.is_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if epoch > 16:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000
        optimizer.step()

        running_loss += loss.data[0]

    running_loss /= len(trainloader)

    train_accuracy = calculate_accuracy(trainloader, args.is_gpu)
    test_accuracy = calculate_accuracy(testloader, args.is_gpu)

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
        epoch+1, running_loss, train_accuracy, test_accuracy))

    if epoch % 50 == 0:
        print('>>> Creating checkpoint for generated model ...')
        state = {
            'net': net.module if args.is_gpu else net,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '../checkpoint/checkpoint.t7')

print('>>> Training Completed ...')
