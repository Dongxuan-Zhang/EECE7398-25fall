import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))

        x = x.view(x.size(0), -1)  # Flatten operation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Create model instance
model = CNN(num_classes=10, in_channels=3)

# Add L2 regularization
weight_decay = 1e-5  # L2 regularization coefficient
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)


def train(dataset):
    # Set dataset and hyperparameters
    if dataset == 'mnist':
        num_classes = 10
        # Data augmentation for MNIST (even though it's grayscale, we can still apply some transformations)
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Apply random affine transformations
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        in_channels = 1  # MNIST is grayscale
    elif dataset == 'cifar':
        num_classes = 10
        # Data augmentation for CIFAR10
        transform_train = transforms.Compose([
            transforms.RandAugment(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        in_channels = 3  # CIFAR is color
    else:
        raise ValueError("Unknown dataset")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    device = get_device()
    print(f"Device: {device}")

    # Move model to appropriate device
    model = CNN(num_classes=num_classes).to(device)
    if in_channels == 1:
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / 500}")


        # Evaluate test set accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in torch.utils.data.DataLoader(testset, batch_size=1):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Testing Accuracy: {accuracy:.2f}%')

    # Save model
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model.state_dict(), f'model/{dataset}_cnn.pth')
    print('Finished Training')

# CIFAR-10 类别名称
cifar10_classes = {
    0: 'Plane',
    1: 'Car',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

def infer_dataset_from_filename(image_path):
    if 'mnist' in image_path.lower():
        return 'mnist'
    elif 'cifar' in image_path.lower():
        return 'cifar'
    else:
        raise ValueError("Unknown dataset")

def visualize_conv_layer(model, image, dataset):
    # get conv layer output
    conv_output = model.conv1(image)
    
    # create a 8x4 subplot layout
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle(f'First CONV Layer Output for {dataset.upper()}', fontsize=16)

    # iterate over 32 filters
    for i, ax in enumerate(axes.flat):
        if i < conv_output.size(1):  # ensure we have enough channels
            # get current filter output
            filter_output = conv_output[0, i].detach().cpu().numpy()
            
            # display image
            im = ax.imshow(filter_output, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')

    # save image
    plt.tight_layout()
    plt.savefig(f'CONV_rslt_{dataset}.png')
    plt.close()

def test(image_path):
    # Infer dataset from filename
    dataset = infer_dataset_from_filename(image_path)

    # Select model file and parameters based on dataset
    if dataset == 'mnist':
        model_path = 'model/mnist_cnn.pth'
        num_classes = 10
        in_channels = 1
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    elif dataset == 'cifar':
        model_path = 'model/cifar_cnn.pth'
        num_classes = 10
        in_channels = 3
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"can't find model file: {model_path}")

    # load model
    device = get_device()
    model = CNN(num_classes=num_classes).to(device)
    if in_channels == 1:
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2).to(device)
    
    # load model weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Execute inference
    with torch.no_grad():
        # get conv layer output
        conv_output = model.conv1(image)
        
        # continue forward propagation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # get prediction result
    if dataset == 'cifar':
        predicted_class = cifar10_classes[predicted.item()]
    else:
        predicted_class = predicted.item()

    print(f"Using {dataset} model to test image: {image_path}")
    print(f"Predicted class: {predicted_class}")

    # Visualize conv layer output
    visualize_conv_layer(model, image, dataset)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'test'], help='Command to execute: train or test')
    parser.add_argument('--mnist', action='store_const', const='mnist', dest='dataset', help='Use MNIST dataset')
    parser.add_argument('--cifar', action='store_const', const='cifar', dest='dataset', help='Use CIFAR dataset')
    parser.add_argument('image_path', nargs='?', help='Path to the image for testing')
    args = parser.parse_args()

    if args.command == 'train':
        if args.dataset:
            train(args.dataset)
        else:
            print('Please specify dataset: --mnist or --cifar')
    elif args.command == 'test':
        if args.image_path:
            test(args.image_path)
        else:
            print('Please specify the image path for testing')
