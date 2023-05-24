import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

import sys
sys.path.append("./")

from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def processing_data(data_path, height, width, batch_size, test_split=0.1):
    mtcnn_transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化处理
    ])

    # 加载数据集并进行划分
    dataset = ImageFolder(data_path, transform=mtcnn_transforms)
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建DataLoader
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader

data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=128)


mtcnn_model = FaceDetector()

model = MobileNetV1(classes=2).to(device)

epochs = 70

optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',  factor=0.7, patience=2)

criterion = nn.CrossEntropyLoss()    


import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

best_loss = 1e9
best_acc = 0
best_model_weights = copy.deepcopy(model.state_dict())
best_model_weights_acc = copy.deepcopy(model.state_dict())
loss_list = []  # 存储损失函数值

for epoch in range(epochs):
    model.train()

    for batch_idx, (images, labels) in tqdm(enumerate(train_data_loader, 1)):
        images = images.to(device)
        labels = labels.to(device)

        # 使用MTCNN检测人脸并裁剪出人脸区域
        faces = []
        for image in images:
            # 将张量image转换为PIL类型
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image.cpu())

            detect_face_img = mtcnn_model.draw_bboxes(pil_image)
            faces.append(detect_face_img)

        # 转换每个face为张量类型
        transform = ToTensor()
        faces = [transform(face) for face in faces]
        faces = torch.stack(faces).to(device)
        #labels = labels.repeat(len(faces))
        
        # 使用MobileNet进行口罩检测
        outputs = model(faces)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss

        loss_list.append(loss.item())

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for images, labels in valid_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            faces = []
            for image in images:

                # 将张量image转换为PIL类型
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(image.cpu())

                detect_face_img = mtcnn_model.draw_bboxes(pil_image)
                faces.append(detect_face_img)

            # 转换每个face为张量类型
            transform = ToTensor()
            faces = [transform(face) for face in faces]
            faces = torch.stack(faces).to(device)

            outputs = model(faces)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = 100.0 * total_correct / total_samples
        print('Validation Accuracy: {:.2f}%'.format(accuracy))

        if accuracy > best_acc:
            best_model_weights_acc = copy.deepcopy(model.state_dict())
            best_acc = accuracy

    # 更新学习率
    scheduler.step(accuracy)

print('Finish Training.')

torch.save(best_model_weights, './results/model.pth')
torch.save(best_model_weights_acc, './results/model_acc.pth')