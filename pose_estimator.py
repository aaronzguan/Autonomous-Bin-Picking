import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torchvision import models
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import MultiStepLR


class pose_dataset(torch.utils.data.Dataset):
    def __init__(self, training=True):
        super(pose_dataset, self).__init__()
        # self.root_path = '/home/aaronguan/Desktop/16662-RobotAutonomy/Project_RLBench/EmptyContainer/dataset'
        self.root_path = '/home/ubuntu/autonomy/dataset'
        if training:
            with open('dataset/train_data.txt') as f:
                image_path = f.readlines()
            self.image_path = image_path
            self.pose = np.load('dataset/train_pose_data.npy')
        else:
            with open('dataset/test_data.txt') as f:
                image_path = f.readlines()
            self.image_path = image_path
            self.pose = np.load('dataset/test_pose_data.npy')

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root_path, self.image_path[index].replace('\n', '')))
        pose = self.pose[index]
        return torch.from_numpy(image).float().permute(2, 0, 1), torch.from_numpy(pose).float()

    def __len__(self):
        return len(self.image_path)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PoseCNN(nn.Module):
    def __init__(self):
        super(PoseCNN, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet18.children()))[:-1]
        self.classifier = nn.Sequential(*[Flatten(),
                                          nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                          nn.Linear(256, 9)])

    def forward(self, image):
        features = self.backbone(image)
        output = self.classifier(features)
        return output


def validate(model, loader):
    model.eval()
    running_batch, running_loss = 0, 0.0
    with torch.no_grad():
        for i, (image, target_pose) in enumerate(loader):
            image, target_pose = image.to(device), target_pose.to(device)
            pred_pose = model(image)
            loss = criterion(pred_pose, target_pose)

            running_batch += len(target_pose)
            running_loss += loss.item()

        running_loss /= running_batch
    return running_loss


def estimate_pose(model_path, image):
    image *= 255
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    poseCNN = PoseCNN()
    poseCNN.load_state_dict(torch.load(model_path))
    poseCNN.to(device)
    poseCNN.eval()
    pose = poseCNN(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()).detach().numpy()[0]
    return pose


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PoseCNN()

    train_dataset = pose_dataset(training=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    test_dataset = pose_dataset(training=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    best_loss = 1.0
    for epoch in range(40):
        model.to(device)
        model.train()
        for i, (image, target_pose) in enumerate(train_dataloader):
            image, target_pose = image.to(device), target_pose.to(device)
            pred_pose = model(image)
            loss = criterion(pred_pose, target_pose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i+1)% 10 == 0):
                print("[{},{}] loss: {}".format(epoch, i+1, loss.item()))

        val_loss = validate(model, test_dataloader)
        print('Epoch: {}, loss: {}'.format(epoch, val_loss))
        if val_loss < best_loss:
            torch.save(model.cpu().state_dict(), "best_model_{}.pth".format(epoch))
            best_loss = val_loss
            print("Model Saved at {} epoch!".format(epoch))

