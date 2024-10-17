import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

# 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) # 평균 0.5, 표준편차 0.5 
])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 로지스틱 회귀 모델 정의
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28*28, 10) # 입력 784, 출력 10

    def forward(self, x):
        x = x.view(-1, 28*28)   # 28 x 28 이미지를 784차원 벡터로 변환
        outputs = self.linear(x)
        return outputs 

model = LogisticRegressionModel()

# 손실 함수와 최적화 함수 설정
criterion = nn.CrossEntropyLoss()   # 소프트맥스 포함
optimizer = optim.SGD(model.parameters(), lr = 0.01)    # 학습률 0.01

# 모델 학습(학습 반복 횟수 10)
num_epochs = 10

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass: 예측값 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: 그래디언트 초기화 및 역전파 
        optimizer.zero_grad()   # 이전 그래디언트 초기화
        loss.backward() # 그래디언트 계산
        optimizer.step()    # 파라미터 업데이트

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 평가
with torch.no_grad():   # 그래디언트 계산 비활성화 
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # 총 레이블 수 
        correct += (predicted == labels).sum().item()   # 맞춘 개수
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')