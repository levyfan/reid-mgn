#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms import functional

from __init__ import DEVICE, cmc, mean_ap
from market1501 import Market1501

root = os.path.dirname(os.path.realpath(__file__)) + '/../Market-1501-v15.09.15'


class IDE(nn.Module):

    def __init__(self, num_classes):
        super(IDE, self).__init__()

        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

        nn.init.kaiming_normal_(self.classifier[0].weight, mode='fan_out')
        nn.init.constant_(self.classifier[0].bias, 0.)

        nn.init.normal_(self.classifier[1].weight, mean=1., std=0.02)
        nn.init.constant_(self.classifier[1].bias, 0.)

        nn.init.normal_(self.classifier[4].weight, std=0.001)
        nn.init.constant_(self.classifier[4].bias, 0.)

    def forward(self, x):
        """
        :param x: input image of (N, C, H, W)
        :return: (feature of N*2048, label predict of N*num_classes)
        """
        x = self.backbone(x)
        x = x.squeeze()

        y = self.classifier(x)
        return x, y


def run():
    batch_size = 32

    train_transform = transforms.Compose([
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_flip_transform = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Market1501(root + '/bounding_box_train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    query_dataset = Market1501(root + '/query', transform=test_transform)
    query_flip_dataset = Market1501(root + '/query', transform=test_flip_transform)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    query_flip_loader = DataLoader(query_flip_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Market1501(root + '/bounding_box_test', transform=test_transform)
    test_flip_dataset = Market1501(root + '/bounding_box_test', transform=test_flip_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_flip_loader = DataLoader(test_flip_dataset, batch_size=batch_size, shuffle=False)

    ide = IDE(num_classes=len(train_dataset.unique_ids)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    params = [
        {'params': ide.backbone.parameters(), 'lr': 0.01},
        {'params': ide.classifier.parameters(), 'lr': 0.1},
    ]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    epochs = 50
    for epoch in range(epochs):
        ide.train()
        scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = ide(inputs)
            loss = criterion(outputs[1], labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print('%d/%d - %d/%d - loss: %f' % (epoch, epochs, i, len(train_loader), loss.item()))
        print('epoch: %d/%d - loss: %f' % (epoch, epochs, running_loss / len(train_loader)))

        if epoch % 10 == 9:
            ide.eval()

            query = np.concatenate([ide(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                    for inputs, _ in query_loader])
            query_flip = np.concatenate([ide(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                         for inputs, _ in query_flip_loader])

            test = np.concatenate([ide(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                   for inputs, _ in test_loader])
            test_flip = np.concatenate([ide(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                        for inputs, _ in test_flip_loader])

            # dist = cdist((query + query_flip) / 2., (test + test_flip) / 2.)
            dist = cdist(normalize(query + query_flip), normalize(test + test_flip))
            r = cmc(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras)
            print('epoch[%d]: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, m_ap, r[0], r[2], r[4], r[9]))


if __name__ == '__main__':
    run()
