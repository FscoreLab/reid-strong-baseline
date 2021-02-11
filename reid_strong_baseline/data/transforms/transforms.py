# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
import skimage.transform
import numpy as np
import cv2

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=128, max_side=256):  # 608
        image = np.asarray(sample)

        rows, cols, cns = image.shape

        smallest_side = cols #min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # scale_x = min_side / rows
        # scale_y = min_side / cols

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = rows # max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 256 - rows  #  32 - rows % 32 if rows % 32 else 0
        pad_h = 128 - cols #  32 - cols % 32 if cols % 32 else 0
        #
        # new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # assert pad_w >= 0 and pad_h >= 0
        # if pad_h < 0 or pad_w < 0:
        #     print(pad_w, pad_h, image.shape)
        # print(image.shape, new_image.shape, (rows, cols), new_image[:rows, :cols, :].shape)
        # new_image[:rows, :cols, :] = image.astype(np.float32)

        if pad_h or pad_w:
            # print(image.shape)
            new_image = cv2.copyMakeBorder(image, 0, pad_w,
                                           0,
                                           pad_h,
                                           cv2.BORDER_REFLECT101)
        else:
            new_image = image

        # print(new_image.shape, pad_w, pad_h, rows, cols)
        return new_image.astype(np.float32)