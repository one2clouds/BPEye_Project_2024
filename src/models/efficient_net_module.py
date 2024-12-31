import torch 
import torch.nn as nn 
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import pathlib
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import EfficientNet_B0_Weights



class Efficient_Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(classes))
        
    def forward(self, x:torch.Tensor, num_classes) -> torch.Tensor:
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        # self.model = self.model.to(device)
        return self.model(x)
    

if __name__== "__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/train/'
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    root=pathlib.Path(train_path)
    classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

    model = Efficient_Net(classes=classes)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for inputs,labels in tqdm(loader):
        inputs = inputs.to(device)
        print(inputs.shape)
        outputs = nn.Sigmoid()(model(inputs))
        print(outputs)
        break     


