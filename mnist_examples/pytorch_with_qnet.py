from __future__ import print_function

import datetime
import logging
import os
import sys
import time

sys.path.append("../quantum-neural-network")

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchsummary
from matplotlib import pyplot as plt

from qnet import QNet

test_accuracy_list = []
training_accuracy_list = []
batches_list = []
parameters_list = []
dhs_list = []


def create_output_model_path(args, version=0):
    if args.quantum:
        model_path = os.path.join('results',
                                  'MNIST-quantum_{}-backend_{}-classes_{}-ansatz_{}-netwidth_{}-nlayers_{}-nsweeps_{}'
                                  '-activation_{}-shots_{}-samples_{}-bsize_{}-optimiser_{}-lr_{}-batchnorm_{}'
                                  '-tepochs_{}-loginterval_{}_{}'.format(
                                      args.quantum, args.q_backend, args.classes, args.q_ansatz, args.width,
                                      args.layers, args.q_sweeps, args.activation, args.shots, args.samples_per_class,
                                      args.batch_size, args.optimiser, args.lr, args.batchnorm, args.epochs,
                                      args.log_interval, version))
    else:
        model_path = os.path.join('results',
                                  'MNIST-quantum_{}-classes_{}-netwidth_{}-nlayers_{}-samples_{}-'
                                  'bsize_{}-optimiser_{}-lr_{}-batchnorm_{}-tepochs_{}-loginterval_{}_{}'.format(
                                      args.quantum, args.classes, args.width, args.layers,
                                      args.samples_per_class, args.batch_size, args.optimiser, args.lr, args.batchnorm,
                                      args.epochs,
                                      args.log_interval, version))

    if os.path.exists(model_path + ".npy"):
        return create_output_model_path(args, version=version + 1)
    else:
        return model_path


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, args.width)
        self.bn1d = nn.BatchNorm1d(args.width)
        self.test_network = nn.ModuleList()

        if args.quantum:
            self.test_network.append(QNet(args.width, args.encoding, args.q_ansatz, args.layers, args.q_sweeps,
                                          args.activation, args.shots, args.q_backend, save_statevectors=True))
        else:
            for i in range(args.layers):
                self.test_network.append(nn.Linear(args.width, args.width, bias=True))

        self.fc2 = nn.Linear(args.width, args.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        if self.args.batchnorm:
            x = self.bn1d(x)
        x = np.pi * torch.sigmoid(x)
        for f in self.test_network:
            x = f(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, test_loader, model_path):
    model.train()
    log_start_time = time.time()
    batches_per_epoch = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        model.test_network[0].qnn.statevectors = []
        correct = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            seen_images = ((batch_idx + 1) * train_loader.batch_size)
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tTime: {:.3f}s'.format(
                epoch, seen_images, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(),
                time.time() - log_start_time))

            # Report the training accuracy
            percentage_accuracy = 100. * correct / len(data)
            training_accuracy_list.append(percentage_accuracy)
            logging.info('Training set accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(data), percentage_accuracy))

            batches_list.append((epoch - 1) * batches_per_epoch + batch_idx)
            parameters_list.append(list(model.test_network[0].parameters())[0].detach().numpy().flatten().tolist())

            if args.q_backend == 'statevector_simulator':
                statevectors = np.array(model.test_network[0].qnn.statevectors)

                labels = np.array(target)

                class_0_statevectors = statevectors[labels == 0]
                class_1_statevectors = statevectors[labels != 0]

                rho = np.mean([np.outer(vector, np.conj(vector)) for vector in class_0_statevectors], axis=0)
                sigma = np.mean([np.outer(vector, np.conj(vector)) for vector in class_1_statevectors], axis=0)

                dhs = np.trace(np.linalg.matrix_power((rho - sigma), 2))
                dhs_list.append(dhs.real)

            test(model, device, test_loader)

            output = [batches_list, test_accuracy_list, parameters_list, training_accuracy_list, dhs_list]
            if args.quantum:
                gradients = model.test_network[0].qnn.gradients
                output.append(gradients)

            np.save(model_path, np.array(output))
            log_start_time = time.time()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    percentage_accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy_list.append(percentage_accuracy)
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), percentage_accuracy))
    print(test_accuracy_list)
    print(batches_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--samples-per-class', default=500, type=int,
                        help='Number of training images per class in the training set (default: 500)')
    parser.add_argument('--optimiser', type=str, default='sgd',
                        help='Optimiser to use (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--submission_time', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        metavar='N',
                        help='Timestamp at submission')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the results of the run')

    parser.add_argument('--batchnorm', action='store_true', default=False,
                        help='If enabled, apply BatchNorm1d to the input of the pre-quantum Sigmoid.')

    parser.add_argument('-q', '--quantum', dest='quantum', action='store_true',
                        help='If enabled, use a minimised version of ResNet-18 with QNet as the final layer')
    parser.add_argument('--q_backend', type=str, default='qasm_simulator',
                        help='Type of backend simulator to run quantum circuits on (default: qasm_simulator)')

    parser.add_argument('-w', '--width', type=int, default=8,
                        help='Width of the test network (default: 8). If quantum, this is the number of qubits.')
    parser.add_argument('--classes', type=int, default=8,
                        help='Number of MNIST classses.')

    parser.add_argument('--encoding', type=str, default='vector',
                        help='Data encoding method (default: vector)')
    parser.add_argument('--q_ansatz', type=str, default='abbas',
                        help='Variational ansatz method (default: abbas)')
    parser.add_argument('--q_sweeps', type=int, default=1,
                        help='Number of ansatz sweeeps.')
    parser.add_argument('--activation', type=str, default='null',
                        help='Quantum layer activation function type (default: null)')
    parser.add_argument('--shots', type=int, default=100,
                        help='Number of shots for quantum circuit evaulations.')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of test network layers.')

    args = parser.parse_args()

    # Create the file where results will be saved
    model_path = create_output_model_path(args)
    np.save(model_path, [])

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_trainset = datasets.MNIST('./datasets', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    no_training_samples = args.samples_per_class
    num_classes = args.classes

    train_labels = mnist_trainset.targets.numpy()
    train_idx = np.concatenate(
        [np.where(train_labels == digit)[0][0:no_training_samples] for digit in range(num_classes)])
    mnist_trainset.targets = train_labels[train_idx]
    mnist_trainset.data = mnist_trainset.data[train_idx]

    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    no_test_samples = 500

    mnist_testset = datasets.MNIST('./datasets', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_labels = mnist_testset.targets.numpy()
    test_idx = np.concatenate([np.where(test_labels == digit)[0][0:no_test_samples] for digit in range(num_classes)])
    mnist_testset.targets = test_labels[test_idx]
    mnist_testset.data = mnist_testset.data[test_idx]

    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args).to(device)
    print(args)
    print(torchsummary.summary(model, (1, 28, 28)))

    if args.optimiser == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimiser == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimiser == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimiser choice not implemented yet')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, test_loader, model_path)
        # test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    if args.plot:
        ax = plt.subplot(111)
        plt.plot(batches_list, test_accuracy_list, "--o")
        plt.xlabel('Training batches', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.show()


if __name__ == '__main__':
    main()
