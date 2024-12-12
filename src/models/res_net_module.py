import torch.nn as nn 
import torchvision
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import pathlib 
from tqdm import tqdm 
import torch.optim as optim
from torchvision.models import ResNet50_Weights


class Res_Net(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))
        
    def forward(self, x:torch.Tensor, num_classes) -> torch.Tensor:
        # SLOWER for every forward call
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
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

    model = Res_Net(classes=classes)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for inputs,labels in tqdm(loader):
        inputs = inputs.to(device)
        outputs = nn.Sigmoid()(model(inputs))
        print(outputs)
        break   