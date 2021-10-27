import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import  ToTensor, Lambda, Compose

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
    



model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)


x,y = test_data[i][0], test_data[i][1]
with torch.no_grad():
    pred= model(x)
    #pred.argmax : 예측 값의 최대 값
    predicted, actual = classes[pred[i].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')