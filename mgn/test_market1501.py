#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

from mock import patch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from market1501 import Market1501, RandomIdSampler

root = os.path.dirname(os.path.realpath(__file__)) + '/../Market-1501-v15.09.15'


class Market1501Test(unittest.TestCase):
    def test_id(self):
        self.assertEqual(-1, Market1501.id('-1_c1s1_000401_03.jpg'))

    def test_camera(self):
        self.assertEqual(1, Market1501.camera('-1_c1s1_000401_03.jpg'))

    def test_train(self):
        train = Market1501(root + '/bounding_box_train')
        self.assertEqual(12936, len(train.imgs))
        self.assertEqual(12936, len(train.ids))
        self.assertEqual(12936, len(train.cameras))
        self.assertEqual(751, len(train._id2label))

        self.assertTrue(train.imgs[0].endswith('0002_c1s1_000451_03.jpg'))
        self.assertEqual(2, train.ids[0])
        self.assertEqual(1, train.cameras[0])
        self.assertEqual(0, train._id2label[2])
        _, target = train[0]
        self.assertEqual(0, target)

        self.assertTrue(train.imgs[46].endswith('0007_c1s6_028546_01.jpg'))
        self.assertEqual(7, train.ids[46])
        self.assertEqual(1, train.cameras[46])
        self.assertEqual(1, train._id2label[7])
        _, target = train[46]
        self.assertEqual(1, target)

    def test_query(self):
        query = Market1501(root + '/query')
        self.assertEqual(3368, len(query.imgs))
        self.assertEqual(3368, len(query.ids))
        self.assertEqual(3368, len(query.cameras))
        self.assertEqual(750, len(query._id2label))

        self.assertTrue(query.imgs[0].endswith('0001_c1s1_001051_00.jpg'))
        self.assertEqual(1, query.ids[0])
        self.assertEqual(1, query.cameras[0])
        self.assertEqual(0, query._id2label[1])
        _, target = query[0]
        self.assertEqual(0, target)

        self.assertTrue(query.imgs[6].endswith('0003_c1s6_015971_00.jpg'))
        self.assertEqual(3, query.ids[6])
        self.assertEqual(1, query.cameras[6])
        self.assertEqual(1, query._id2label[3])
        _, target = query[6]
        self.assertEqual(1, target)

    def test_test(self):
        test = Market1501(root + '/bounding_box_test')
        self.assertEqual(15913, len(test.imgs))
        self.assertEqual(15913, len(test.ids))
        self.assertEqual(15913, len(test.cameras))
        self.assertEqual(751, len(test._id2label))

        self.assertTrue(test.imgs[0].endswith('0000_c1s1_000151_01.jpg'))
        self.assertEqual(0, test.ids[0])
        self.assertEqual(1, test.cameras[0])
        self.assertEqual(0, test._id2label[0])
        img, target = test[0]
        self.assertEqual(0, target)

        self.assertTrue(test.imgs[2798].endswith('0001_c1s1_001051_03.jpg'))
        self.assertEqual(1, test.ids[2798])
        self.assertEqual(1, test.cameras[2798])
        self.assertEqual(1, test._id2label[1])
        img, target = test[2798]
        self.assertEqual(1, target)


class RandomIdSamplerTest(unittest.TestCase):
    def setUp(self):
        self.batch_id = 4
        self.batch_image = 16
        self.data_source = Market1501(root + '/bounding_box_train', transform=ToTensor())
        self.sampler = RandomIdSampler(self.data_source, batch_image=self.batch_image)
        self.data_loader = DataLoader(self.data_source,
                                      sampler=self.sampler, batch_size=self.batch_id * self.batch_image)

    @patch('random.shuffle', lambda x: x)
    @patch('random.sample', lambda population, k: population[:k])
    def test_sampler(self):
        imgs = [img for img in self.sampler]
        self.assertEqual(range(16), imgs[:16])
        self.assertEqual(range(46, 53) + range(46, 53) + range(46, 48), imgs[16:32])

    @patch('random.shuffle', lambda x: x)
    @patch('random.sample', lambda population, k: population[:k])
    def test_data_loader(self):
        it = self.data_loader.__iter__()

        _, target = next(it)
        self.assertEqual([0] * 16 + [1] * 16 + [2] * 16 + [3] * 16, target.numpy().tolist())

        _, target = next(it)
        self.assertEqual([4] * 16 + [5] * 16 + [6] * 16 + [7] * 16, target.numpy().tolist())
