import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 전처리 및 로드 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# MNIST 데이터셋에서 0, 1만 필터링
def filter_01(dataset):
    targets = dataset.targets 
    mask = (targets == 0) | (targets == 1)
    dataset.targets = targets[mask]
    dataset.data = dataset.data[mask]
    return dataset 

# 필터링된 데이터셋 로드
train_dataset = filter_01(datasets.MNIST(
    root='./data', train=True, transform=transform, download = True
))
test_dataset = filter_01(datasets.MNIST(
    root='./data', train=False, transform=transform
))

# 데이터 로더
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) # MNIST와 같은 작은 데이터셋에서 batch size로 64가 주로 사용. 학습 안정성 유지 
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False) # 테스트 속도 높이기 위해 batch size를 1000으로 설정(테스트 시에는 가중치 업데이트 X)

# 로지스틱 회귀 모델
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28*28, 2)   # 출력 0, 1 총 2개 

    def forward(self, x):
        x = x.view(-1, 28*28)   # 28 x 28 이미지를 784차원 벡터로 변환
        outputs = self.linear(x)
        return outputs

# 모델  
model = LogisticRegression()

# 손실 함수와 최적화 함수 
criterion = nn.CrossEntropyLoss()   # 소프트맥스 포함
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 학습률 0.01

# 학습 반복 횟수
num_epochs = 5 

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass: 예측값 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: 그래디언트 초기화 및 역전파
        optimizer.zero_grad()
        loss.backward() # 그래디언트 계산
        optimizer.step() # 파라미터 업데이트 

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 평가
with torch.no_grad():
    correct_0 = 0   # 올바른 예측 수 
    correct_1 = 0
    total_0 = 0     # 레이블 총 개수
    total_1 = 0

    for images, labels in test_loader:
        outputs = model(images) # 예측값 계산
        _, predicted = torch.max(outputs.data, 1)   # 가장 높은 값의 인덱스가 예측 클래스

        # 정확도 계산
        total_0 += (labels==0).sum().item()
        total_1 += (labels==1).sum().item()
        correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
        correct_1 += ((predicted == 1) & (labels == 1)).sum().item()

        accuracy_0 = 100*(correct_0/total_0)
        accuracy_1 = 100*(correct_1/total_1)

    print(f'Accuracy for class 0: {accuracy_0:.2f}%')
    print(f'Accuracy for class 1: {accuracy_1:.2f}%')

# 가중치 저장
torch.save(model.state_dict(), 'logistic_regression.pth')