import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image    
import matplotlib.pyplot as plt




class Classifier(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.classifier(x)
    




if __name__ == "__main__":
    
    # device = "mps"
    
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    #     batch_size=128, shuffle=True
    # )
    
    # classifier = Classifier().to(device)
    
    # optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    
    # epochs = 10
    
    # for epoch in range(epochs):
    #     for x, y in train_loader:
    #         x = x.view(x.size(0), -1).to(device)
    #         y = y.to(device)
            
    #         optimizer.zero_grad()
    #         logits = classifier(x)
    #         loss = criterion(logits, y)
    #         loss.backward()
    #         optimizer.step()
            
    #         print(f"Epoch: {epoch}, Loss: {loss.item()}")
            
    # torch.save(classifier.state_dict(), 'classifier.pth')
    
    classifier = Classifier()

    model = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(model)
    classifier.eval()

    path = 'samples'
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Assurez-vous que l'image est en niveaux de gris
        transforms.Resize((28, 28)),  # Redimensionnez l'image à 28x28 pixels
        transforms.ToTensor(),  # Convertissez l'image en tenseur
        transforms.Normalize((0.5,), (0.5,))  # Normalisez l'image
    ])
    
    l = []
    
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path).convert('L')  # Lisez l'image en tant qu'image PIL et convertissez-la en niveaux de gris
        img = transform(img)  # Appliquez les transformations
        img = img.flatten().unsqueeze(0)  # Aplatissez et ajoutez une dimension batch
        logits = classifier(img)
        print(f"Image: {i}, Predicted Class: {logits.argmax().item()}")
        l.append(logits.argmax().item())
        
        

    
    plt.hist(l, bins=10)
    plt.xlabel('Classes')
    plt.ylabel('Nombre de samples')
    plt.title('Distribution des classes')
    plt.show()
        
        