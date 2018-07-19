#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import copy
import multiprocessing
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import resnet50, Bottleneck
from torchvision.transforms import functional

from __init__ import cmc, mean_ap, DEVICE
from market1501 import Market1501, RandomIdSampler
from triplet import TripletSemihardLoss

parser = argparse.ArgumentParser(description='Multiple Granularity Network')
parser.add_argument('--model', choices=['mgn', 'p1_single', 'p2_single', 'p3_single'], default='mgn')
parser.add_argument('--root', default=os.path.dirname(os.path.realpath(__file__)) + '/../Market-1501-v15.09.15')
parser.add_argument('--workers', default=multiprocessing.cpu_count() / 2, type=int)
args = parser.parse_args()
print('')
print(args)
print('')


class MGN(nn.Module):
    """
    @ARTICLE{2018arXiv180401438W,
        author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
        title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1804.01438},
        primaryClass = "cs.CV",
        keywords = {Computer Science - Computer Vision and Pattern Recognition},
        year = 2018,
        month = apr,
        adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """

    def __init__(self, num_classes):
        super(MGN, self).__init__()

        resnet = resnet50(pretrained=True)

        # backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3[0],  # res_conv4_1
        )

        # 3. Multiple Granularity Network 3.1. Network Architecture: The difference is that we employ no down-sampling
        # operations in res_conv5_1 block.

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # mgn part-2
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        # Figure 3: Notice that the 1 × 1 convolutions for dimension reduction and fully connected layers for identity
        # prediction in each branch DO NOT share weights with each other.

        # 4. Experiment 4.1 Implementation: Notice that different branches in the network are all initialized with the
        # same pretrained weights of the corresponding layers after the res conv4 1 block.

        # conv1 reduce
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)
        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        predict = []
        triplet_losses = []
        softmax_losses = []

        if args.model in {'mgn', 'p1_single'}:
            p1 = self.p1(x)

            zg_p1 = self.maxpool_zg_p1(p1)  # z_g^G
            fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)  # f_g^G, L_triplet^G
            l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))  # L_softmax^G

            predict.append(fg_p1)
            triplet_losses.append(fg_p1)
            softmax_losses.append(l_p1)

        if args.model in {'mgn', 'p2_single'}:
            p2 = self.p2(x)

            zg_p2 = self.maxpool_zg_p2(p2)  # z_g^P2
            fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)  # f_g^P2, L_triplet^P2
            l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P2

            zp2 = self.maxpool_zp2(p2)
            z0_p2 = zp2[:, :, 0:1, :]  # z_p0^P2
            z1_p2 = zp2[:, :, 1:2, :]  # z_p1^P2
            f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)  # f_p0^P2
            f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)  # f_p1^P2
            l0_p2 = self.fc_id_256_1_0(f0_p2)  # L_softmax0^P2
            l1_p2 = self.fc_id_256_1_1(f1_p2)  # L_softmax1^P2

            predict.extend([fg_p2, f0_p2, f1_p2])
            triplet_losses.append(fg_p2)
            softmax_losses.extend([l_p2, l0_p2, l1_p2])

        if args.model in {'mgn', 'p3_single'}:
            p3 = self.p3(x)

            zg_p3 = self.maxpool_zg_p3(p3)  # z_g^P3
            fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)  # f_g^P3, L_triplet^P3
            l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P3

            zp3 = self.maxpool_zp3(p3)
            z0_p3 = zp3[:, :, 0:1, :]  # z_p0^P3
            z1_p3 = zp3[:, :, 1:2, :]  # z_p1^P3
            z2_p3 = zp3[:, :, 2:3, :]  # z_p2^P3
            f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)  # f_p0^P3
            f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)  # f_p1^P3
            f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)  # f_p2^P3
            l0_p3 = self.fc_id_256_2_0(f0_p3)  # L_softmax0^P3
            l1_p3 = self.fc_id_256_2_1(f1_p3)  # L_softmax1^P3
            l2_p3 = self.fc_id_256_2_2(f2_p3)  # L_softmax2^P3

            predict.extend([fg_p3, f0_p3, f1_p3, f2_p3])
            triplet_losses.append(fg_p3)
            softmax_losses.extend([l_p3, l0_p3, l1_p3, l2_p3])

        # 3. Multiple Granularity Network 3.1. Network Architecture: During testing phases, to obtain the most powerful
        # discrimination, all the features reduced to 256-dim are concatenated as the final feature.
        predict = torch.cat(predict, dim=1)
        return predict, triplet_losses, softmax_losses


def run():
    batch_id = 16
    batch_image = 4
    batch_test = 32

    train_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Market1501(args.root + '/bounding_box_train', transform=train_transform)
    train_loader = dataloader.DataLoader(train_dataset,
                                         sampler=RandomIdSampler(train_dataset, batch_image=batch_image),
                                         batch_size=batch_id * batch_image,
                                         num_workers=args.workers)

    test_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_flip_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query_dataset = Market1501(args.root + '/query', transform=test_transform)
    query_flip_dataset = Market1501(args.root + '/query', transform=test_flip_transform)
    query_loader = dataloader.DataLoader(query_dataset, batch_size=batch_test, num_workers=args.workers)
    query_flip_loader = dataloader.DataLoader(query_flip_dataset, batch_size=batch_test, num_workers=args.workers)

    test_dataset = Market1501(args.root + '/bounding_box_test', transform=test_transform)
    test_flip_dataset = Market1501(args.root + '/bounding_box_test', transform=test_flip_transform)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=batch_test, num_workers=args.workers)
    test_flip_loader = dataloader.DataLoader(test_flip_dataset, batch_size=batch_test, num_workers=args.workers)

    mgn = MGN(num_classes=len(train_dataset.unique_ids)).to(DEVICE)

    cross_entropy_loss = nn.CrossEntropyLoss()
    triplet_semihard_loss = TripletSemihardLoss(margin=1.2)

    optimizer = optim.SGD(mgn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60], gamma=0.1)

    epochs = 80
    for epoch in range(epochs):
        mgn.train()
        scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = mgn(inputs)
            losses = [triplet_semihard_loss(output, labels) for output in outputs[1]] + \
                     [cross_entropy_loss(output, labels) for output in outputs[2]]
            loss = sum(losses) / len(losses)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print('%d/%d - %d/%d - loss: %f' % (epoch + 1, epochs, i, len(train_loader), loss.item()))
        print('epoch: %d/%d - loss: %f' % (epoch + 1, epochs, running_loss / len(train_loader)))

        if epoch % 10 == 9:
            mgn.eval()

            query = np.concatenate([mgn(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                    for inputs, _ in query_loader])
            query_flip = np.concatenate([mgn(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                         for inputs, _ in query_flip_loader])

            test = np.concatenate([mgn(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                   for inputs, _ in test_loader])
            test_flip = np.concatenate([mgn(inputs.to(DEVICE))[0].detach().cpu().numpy()
                                        for inputs, _ in test_flip_loader])

            dist = cdist((query + query_flip) / 2., (test + test_flip) / 2.)
            # dist = cdist(normalize(query + query_flip), normalize(test + test_flip))
            r = cmc(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras)
            print('epoch[%d]: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, m_ap, r[0], r[2], r[4], r[9]))


if __name__ == '__main__':
    run()
