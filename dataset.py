import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import os.path
import re
import filecmp

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    # random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []

        for i in seq:
            index = int(filepath[char_len - 7:char_len - 4]) - i
            file_name = filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png'

            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png').convert('RGB'),
                               scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath, 'im' + str(nFrames) + '.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC) for j in reversed(seq)]

    return target, input, neigbor


def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames / 2)  # 准备往左右拿 future frames
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        # target = target.resize((int(target.size[0] / 2), int(target.size[1] / 2)), Image.BICUBIC)
        # 从这里的 input 和 target 的定义可以看出，input 是 输入图片下采样而来
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []
        if nFrames % 2 == 0:
            seq = [x for x in range(-tt, tt) if x != 0]
        else:
            # 如果要 7 帧，那么就应该是 [-3, -2, -1, 1, 2, 3]
            seq = [x for x in range(-tt, tt + 1) if x != 0]
        # random.shuffle(seq) #if random sequence
        # 个人觉得没有必要shuffle，除非是在ablation study
        for i in seq:
            index1 = int(filepath[char_len - 7:char_len - 4]) + i
            file_name1 = filepath[0:char_len - 7] + '{0:03d}'.format(index1) + '.png'

            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

    else: # 仅用一个other_dataset来区分是否为vimeo的dataset也太粗糙了，这个方法只适用于作者机器上的dataset，不适用其他的dataset，此vimeo dataset非彼vimeo dataset
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = []
        split_result = re.split(r'[\\|/]', filepath)
        target_index = int(split_result[-1][2:-4])
        seq = [x for x in range(-tt, tt + nFrames % 2) if x != 0]
        # random.shuffle(seq) #if random sequence
        for j in seq:
            neighbour_index = target_index + j
            neighbour_file_name = 'im' + str(neighbour_index).zfill(5) + '.png'
            split_result[-1] = neighbour_file_name
            neighbour_file_path = '/'.join(split_result)

            if os.path.exists(neighbour_file_path):
                temp = modcrop(Image.open(neighbour_file_path).convert('RGB'), scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

            #neigbor.append(modcrop(Image.open(neighbour_file_path).convert('RGB'), scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC))
            
    return target, input, neigbor


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                         nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # flow = rescale_flow(flow,0,1)
    return flow


def rescale_flow(x, max_range, min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range - min_range) / (max_val - min_val) * (x - max_val) + max_range


def modcrop(img, modulo):
    # 对图片进行裁切
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [line.rstrip() for line in open(join(image_dir, file_list))][0:1270] # max: [0:25402]
        self.image_filenames = [join(image_dir, x) for x in self.image_filenames]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                     self.other_dataset)
        else:
            print('You should not be able to see this: DatasetFromFolder.__getitem__.future_frame == False')
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                              self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input, target, neigbor, self.patch_size, self.upscale_factor,
                                                  self.nFrames)

        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)

        flow = [get_flow(input, j) for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        # image_dir-default: './Vid4';
        # nFrames-default: 7;
        # upscale_factor-default: 4;
        # file_list-default: 'foliage.txt'
        # other_dataset-default: True 这个为True，是为了避开vimeo的数据集
        # future_frame-default: True
        # transform-default: Transform to tensor

        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                     self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                              self.other_dataset)

        flow = [get_flow(input, j) for j in neigbor]
        print('You should not be able to see this: DatasetFromFolderTest.__getitem__')

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)
