#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import unittest

import torch
from scipy.spatial.distance import cdist

from triplet import TripletSemihardLoss


def triplet_semihard_loss(input, target, margin):
    """
    :param input: :math:`(N, C)` where `C = number of channels`
    :param target: :math:`(N)`
    :param margin:
    :return: scalar.
    """
    dist = cdist(input, input)

    pos_max = np.zeros(target.shape[0])
    neg_min = np.zeros(target.shape[0])
    for i in range(input.shape[0]):
        anchor = target[i]
        postives = dist[i, target == anchor]
        negtives = dist[i, target != anchor]

        pos_max[i] = np.max(postives)
        neg_min[i] = np.min(negtives)

    # loss(x, y) = max(0, -y * (x1 - x2) + margin)
    loss = np.maximum(0, margin - (neg_min - pos_max))
    return np.mean(loss)


class TripletSemihardLossTest(unittest.TestCase):

    def setUp(self):
        self.criterion = TripletSemihardLoss(margin=1.)

    def test_forward(self):
        input = np.random.randn(12, 100)
        target = np.random.randint(4, size=(12,))

        loss = triplet_semihard_loss(input, target, margin=1.)
        loss_ = self.criterion(torch.from_numpy(input), torch.from_numpy(target)).item()
        self.assertAlmostEqual(loss, loss_, delta=1e-6)
