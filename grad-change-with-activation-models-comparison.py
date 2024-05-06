import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class NoisyReLU(nn.Module):
    def __init__(self, noise_level=0.9):
        super(NoisyReLU, self).__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return torch.relu(x + noise)

# Function to inject constant noise into the gradients
def inject_faults(parameters, noise_level=0.01):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_level
                param.grad += noise

# Function to inject a vector of a noise distribution into the gradients
def inject_faults_vector(parameters, noise_level=0.01):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:

                desired_mean = 1  # Mean of the noise
                desired_std = 0.5  # Standard deviation of the noise

                # Generate noise with standard normal distribution and scale it
                noise_level = torch.randn_like(param.grad) * desired_std + desired_mean

                noise = torch.randn_like(param.grad) + noise_level
                param.grad += noise



# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformation for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Loading the dataset
dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# A list to hold the images and labels
image_label_list = []

my_dict = {}

N = 10

for _ in range(100):
    img, label = random.choice(dataset)
    image_label_list.append((img, label))

model_names = ['resnet18', 'vgg16', 'densenet121']


# Function to replace activation functions
def replace_activations(model, old, new):
    for name, module in model.named_children():
        if isinstance(module, old):
            setattr(model, name, new)
        else:
            replace_activations(module, old, new)


fi_counts = []

noise_level = (0.3 + 0.4 + 0.5) / 3


for _ in range(N):
    for model_name in model_names:

        change_counter = 0
        counter = 0
        image_counter = 0
        for _ in range(100):


            counter += 1
            img, label = image_label_list[image_counter]

            img = img.unsqueeze(0).to(device)
            label = torch.tensor([label], dtype=torch.long, device=device)

            if (model_name == 'resnet18'):
                model = torchvision.models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, 100)
            elif (model_name == 'vgg16'):
                model = torchvision.models.vgg16(pretrained=True)
                # Replace the final fully connected layer
                num_features = model.classifier[6].in_features  # Access the in_features of the last layer in classifier
                model.classifier[6] = nn.Linear(num_features, 100)  # CIFAR-100 has 100 classes
            elif (model_name == 'densenet121'):
                model = torchvision.models.densenet121(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, 100)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
            inject_faults_vector(model.parameters(), noise_level=noise_level)
            optimizer.step()

            replace_activations(model, nn.ReLU, NoisyReLU(noise_level=0.9))

            model.eval()
            with torch.no_grad():
                outputs = model(img)
                _, new_predicted = torch.max(outputs, 1)

            if original_predicted.item() != new_predicted.item():
                change_counter += 1

            image_counter += 1

        if (my_dict.__contains__(model_name)):
            curr = my_dict.get(model_name)
            curr += change_counter
            my_dict[model_name] = curr
        else:
            my_dict[model_name] = change_counter

        print("CURRENT DICT", my_dict)

for key in my_dict:
    my_dict[key] /= N

fi_counts = list(my_dict.items())

data = fi_counts

# Unpacking the data into x and y coordinates
x = [point[0] for point in data]
y = [point[1] for point in data]

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o')
plt.title('Models vs Misclassifications grad change with activation')
plt.xlabel('Model')
plt.ylabel('Misclassifications')
plt.grid(True)
plt.show()

plt.savefig('plot-gradchange-with-activation-model-comparison.jpg', format='jpg')