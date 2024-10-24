import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image 
from torchvision import transforms

# Sobel 필터 사용하는 convolution layer 
class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]])
        
        # convolution layer 설정
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, bias=False)

        # sobel 필터를 레이어의 가중치로 설정
        self.conv_x.weight = nn.Parameter(sobel_x.unsqueeze(0).unsqueeze(0))
        self.conv_y.weight = nn.Parameter(sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges 

# 이미지를 텐서로 변환
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # 흑백 변환
    transform = transforms.ToTensor()  # 텐서 변환
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가 
    return image_tensor

# 정규화 후 시각화 
def normalize_and_plot(original, edge_detected): 
    edge_detected = edge_detected.squeeze().detach().numpy()
    edge_detected = (edge_detected - edge_detected.min()) / (edge_detected.max() - edge_detected.min())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.squeeze(), cmap='gray')
    axs[0].set_title('oroginal image')  
    axs[0].axis('off')

    axs[1].imshow(edge_detected, cmap='gray')
    axs[1].set_title('edge detected image')
    axs[1].axis('off')

    plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    image_path = "./image.png"  # 이미지 경로 설정
    image = load_image(image_path)

    sobel_layer = SobelLayer()  # SobelLayer 인스턴스 생성
    edge_detected = sobel_layer(image)  # Sobel 연산 적용

    # 결과 시각화
    normalize_and_plot(image, edge_detected)
