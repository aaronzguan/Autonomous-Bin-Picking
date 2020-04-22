"""
Train a network to predict if current container has objects or not
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import cv2
from torchvision import models
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import MultiStepLR
from glob import glob


class object_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(object_dataset, self).__init__()
        root_path = '/home/aaronguan/Desktop/16662-RobotAutonomy/Project_RLBench/EmptyContainer/dataset/'
        # root_path = '/home/ubuntu/autonomy/dataset/'
        positive_img_path = glob(root_path + 'contain_object/*')
        negative_img_path = glob(root_path + 'no_object/*')
        self.image_path = positive_img_path + negative_img_path
        self.label = [1] * len(positive_img_path) + [0] * len(negative_img_path)

    def __getitem__(self, index):
        image = cv2.imread(self.image_path[index])
        label = np.asarray(self.label[index]).reshape(1, 1)
        return ToTensor()(image), torch.LongTensor(label)

    def __len__(self):
        return len(self.image_path)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class object_detector_cnn(nn.Module):
    def __init__(self):
        super(object_detector_cnn, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet18.children()))[:-1]
        self.classifier = nn.Sequential(*[Flatten(),
                                          nn.Linear(512, 2)])

    def forward(self, image):
        features = self.backbone(image)
        output = self.classifier(features)
        return output


def check_container_empty(model_path, image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = object_detector_cnn()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    output = model(ToTensor()(image).unsqueeze(0))
    _, label_pred = torch.max(output, 1)
    label_pred = label_pred.detach().numpy()[0]
    return not label_pred


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = object_detector_cnn()

    train_dataset = object_dataset()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    best_acc = 0
    for epoch in range(40):
        model.to(device)
        model.train()
        running_batch = 0
        running_loss = 0.0
        running_corrects = 0
        for i, (image, label) in enumerate(train_dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, label_pred = torch.max(output, 1)

            running_batch += label.size(0)
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(label_pred == label.view(-1)).item()

            if ((i+1)% 10 == 0):
                print("[{},{}] loss: {}, accuracy: {}".format(epoch, i+1, running_loss/running_batch, running_corrects/running_batch))

        cur_acc = running_corrects/running_batch
        if best_acc < cur_acc:
            torch.save(model.cpu().state_dict(), "best_model_{}.pth".format(epoch))
            print("Model Saved at {} epoch!".format(epoch))
            best_acc = cur_acc

