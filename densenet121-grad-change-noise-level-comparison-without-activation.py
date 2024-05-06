import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random


# Define NoisyReLU activation function
class NoisyReLU(nn.Module):
    def __init__(self, noise_level=0.1):
        super(NoisyReLU, self).__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return F.relu(x + noise)


# Function to inject constant noise into the gradients
def inject_faults(parameters, noise_level=0.01):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_level
                # print(param.grad.shape)
                param.grad += noise

# Function to inject a vector of a noise distribution into the gradients
def inject_faults_vector(parameters, noise_level=0.01):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                # Define your desired mean and standard deviation for the noise
                desired_mean = 1  # Mean of the noise
                desired_std = 0.5  # Standard deviation of the noise

                # Generate noise with standard normal distribution and scale it
                noise_level = torch.randn_like(param.grad) * desired_std + desired_mean

                noise = torch.randn_like(param.grad) + noise_level
                # print(param.grad.shape)
                param.grad += noise


# Load the DenseNet model and adjust for CIFAR100
model = torchvision.models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to replace activation functions in the model
def replace_activations(model, old, new):
    for name, module in model.named_children():
        if isinstance(module, old):
            setattr(model, name, new)
        else:
            replace_activations(module, old, new)

model.train()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Create a list to hold the images and labels
image_label_list = []

# Randomly select 100 images and their labels
for _ in range(100):
    img, label = random.choice(dataset)
    image_label_list.append((img, label))

my_dict = {}

N = 10

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

nv_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]

fi_counts = []

for nv in nv_levels:

    change_counter = 0
    image_counter = 0

    for _ in range(100):

        img, label = image_label_list[image_counter]

        img = img.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long, device=device)

        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 100)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(img)
            _, original_predicted = torch.max(outputs, 1)

        model.train()
        optimizer.zero_grad()
        outputs = model(img)
        loss = nn.CrossEntropyLoss()(outputs, label)
        loss.backward()
        inject_faults_vector(model.parameters(), noise_level=nv)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(img)
            _, new_predicted = torch.max(outputs, 1)

        if original_predicted.item() != new_predicted.item():
            change_counter += 1

        image_counter+=1

    if (my_dict.__contains__(nv)):
        curr = my_dict.get(nv)
        curr += change_counter
        my_dict[nv] = curr
    else:
        my_dict[nv] = change_counter

    print("CURRENT DICT", my_dict)

for key in my_dict:
    my_dict[key] /= N

fi_counts = list(my_dict.items())

import matplotlib.pyplot as plt

data = fi_counts

# Unpacking the data into x and y coordinates
x = [point[0] for point in data]
y = [point[1] for point in data]

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o')
plt.title('Noise Levels vs Misclassifications for grad change without activation')
plt.xlabel('Noise Level')
plt.ylabel('Misclassifications')
plt.grid(True)
plt.show()

plt.savefig('plot-noise-levels-grad-change-without-activation.jpg', format='jpg')
