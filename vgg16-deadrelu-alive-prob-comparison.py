import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class DeadReLU(nn.Module):
    def __init__(self, alive_probability=0.9):
        super(DeadReLU, self).__init__()
        self.alive_probability = alive_probability

    def forward(self, x):
        random_mask = (torch.rand_like(x) < self.alive_probability).float()
        return torch.relu(x) * random_mask


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

model = torchvision.models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features  # Access the in_features of the last layer in classifier
model.classifier[6] = nn.Linear(num_features, 100)  # CIFAR-100 has 100 classes

model.to(device)
model.eval()


# Function to replace activation functions
def replace_activations(model, old, new):
    for name, module in model.named_children():
        if isinstance(module, old):
            setattr(model, name, new)
        else:
            replace_activations(module, old, new)


alive_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fi_counts = []

# A list to hold the images and labels
image_label_list = []

# Randomly select 100 images and their labels
for _ in range(100):
    img, label = random.choice(dataset)
    image_label_list.append((img, label))

my_dict = {}

N = 10

for _ in range(N):
    for ap in alive_probs:

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

            # Replace ReLU with DeadReLU
            replace_activations(model, nn.ReLU, DeadReLU(alive_probability=ap))

            # Predict the label with the modified model
            with torch.no_grad():
                outputs = model(img)
                _, new_predicted = torch.max(outputs, 1)
                new_class = dataset.classes[new_predicted.item()]

            # Check if the original and new predictions are different
            if original_predicted != new_predicted:
                change_count += 1

            image_counter += 1

        if (my_dict.__contains__(ap)):
            curr = my_dict.get(ap)
            curr += change_count
            my_dict[ap] = curr
        else:
            my_dict[ap] = change_count

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
plt.title('Alive Probabilities vs Misclassifications for VGG16 model with Dead ReLU')
plt.xlabel('Alive Probability')
plt.ylabel('Misclassifications')
plt.grid(True)
plt.show()

plt.savefig('plot-alive-probs-vgg16.png', format='png')