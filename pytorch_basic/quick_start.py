import os
import torch

from torch import nn
#Dataset은 샘플과 정답을 저장하고, DataLoader는 Dataset을 순회 가능한 객체로 감싼다.
from torch.utils.data import DataLoader
# CIFAR, COCO 등과 같은 다양한 비전 데이터에 대한 데이터셋을 포함 
from torchvision import datasets
#샘플과 정답을 변경하기 위한 transform과 target_transform의 두 인자를 포함
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)


test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

batch_size = 64
# 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch), 샘플링(sampling), 섞기(shuffle) 
# 및 다중 프로세스로 데이터 불러오기(multiprocess data loading)를 지원
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in test_dataloader:
    print("Shape of X [N, C, H, W]: ",X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        # super 연산자 : 부모 클래스로 부터 상속받은 필드나 메소드를 자식 클래스에서 참조하는 데 사용하는 참조 변수
        super(NeuralNetwork, self).__init__()
        # nn.Flatten : 연속 된 범위의 Dim을 텐서로 평평하게 만듬
        self.flatten=nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    




model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr  = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X ,y )in enumerate(dataloader):
            X,y= X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            # pytorch 에서는 gradients 값들을 추후에 backward를 해줄 때 계속 더해주기 때문에 backpropagation을 하기 전에 gradient의 값을 0으로 
            # 만들어주고 시작해야 함.
            optimizer.zero_grad()
            loss.backward()
            # method를 통해 argument로 전달받은 parameter를 업데이트 한다.
            optimizer.step()
            
            if batch % 100 ==0:
                loss , current = loss.item(), batch * len(X)
                print(f"loss : {loss :>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)== y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n") 

epochs = 20
for t in range(epochs):
    print(f"Epochs {t+1}\n------------------------")
    train(train_dataloader,model, loss_fn ,optimizer)
    test(test_dataloader, model, loss_fn)

print("Done")

torch.save(model.state_dict(),"model.pth")
print("Saved Pytorch model")


classes = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",]

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

x,y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    pred= model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')