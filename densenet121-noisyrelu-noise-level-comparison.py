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

model = torchvision.models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 100)

model.to(device)
model.eval()

# Function to replace activation functions
def replace_activations(model, old, new):
    for name, module in model.named_children():
        if isinstance(module, old):
            setattr(model, name, new)
        else:
            replace_activations(module, old, new)


noise_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
fi_counts = []

nv_avg = []
nv_each = []

# A list to hold the images and labels
image_label_list = []

# Randomly select 100 images and their labels
for _ in range(100):
    img, label = random.choice(dataset)
    image_label_list.append((img, label))

my_dict = {}

N = 10

for _ in range(N):
    for nv in noise_values:

        change_count = 0
        counter = 0
        image_counter = 0

        for _ in range(100):

            counter += 1

            img, label = image_label_list[image_counter]

            img = img.unsqueeze(0).to(device)

            # Predict the label with the original model
            with torch.no_grad():
                outputs = model(img)
                _, original_predicted = torch.max(outputs, 1)
                original_class = dataset.classes[original_predicted.item()]

            # Replace ReLU with NoisyReLU
            replace_activations(model, nn.ReLU, NoisyReLU(noise_level=nv))

            # Predict the label with the modified model
            with torch.no_grad():
                outputs = model(img)
                _, new_predicted = torch.max(outputs, 1)
                new_class = dataset.classes[new_predicted.item()]

            # Check if the original and new predictions are different
            if original_predicted != new_predicted:
                change_count += 1

            image_counter += 1

        if (my_dict.__contains__(nv)):
            curr = my_dict.get(nv)
            curr += change_count
            my_dict[nv] = curr
        else:
            my_dict[nv] = change_count

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
plt.title('Noise Levels vs Misclassifications for DenseNet121 model')
plt.xlabel('Noise Level')
plt.ylabel('Misclassifications')
plt.grid(True)
plt.show()

plt.savefig('plot-noise-levels-densenet121.png', format='png')