import torch
import os
import cv2
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
# cfrom skimage  import io
class MyDataset(Dataset):

    def __init__(self,cfg, img_list, label_list):

        # self.transform = transform
        self.data_path = cfg['datadir']

        self.label2idx = {'dog': 0,
             'elephant': 1,
             'giraffe': 2,
             'guitar': 3,
             'horse': 4,
             'house': 5,
             'person': 6}

        labels = []

        for img, label in zip(img_list, label_list):
            img_path = os.path.join(cfg['datadir'],'train', label,img)
            label = self.label2idx[label]
            tmp = {'img':img_path, 'label':label}
            labels.append(tmp)
        self.labels =labels

        self.transform = transforms.Compose([
            Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

        self.train_transforms=transforms.Compose([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=(1,1.1)),
                transforms.ColorJitter(contrast=0.1),
                transforms.ColorJitter(saturation=0.1),
            ]),
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10,fill=255),
                transforms.RandomCrop((224,224)), #이미지 사이즈 설정
            ]),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])



        # for idx, file_type in enumerate(os.listdir(self.data_dir)):
        #     label_path = os.path.join(self.data_dir, file_type)
        #     for img in os.listdir(label_path):
        #         img_path=  os.path.join(label_path ,img)
        #         label = self.label2idx[file_type]
        #         label = {'img': img_path, 'label': label}
        #         labels.append(label)


        self.labels = labels
        self.toTensor = transforms.ToTensor()

    def __len__(self):
         return len(self.labels)
    def __getitem__(self, idx):

        #1. 데이터 불러오기
        label = self.labels[idx]['label']
        image = Image.open(self.labels[idx]['img'])

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data = {'input':image, 'label':label}

        #augmentation
        # data= self.transform(data)
        # image, label = data['input'], data['label']

        image= self.train_transforms(image)




        # image = self.transform(image=np.array(image))['image']

        return image, label


def data_gather(cfg):

    imgs = []
    labels = []
    f = open(cfg['train_file'], 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')  # 줄 끝의 줄 바꿈 문자를 제거한다.
        imgs.append(line[0])
        labels.append(line[1])

    f.close()

    return imgs, labels




def train_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Pad(4),
        transforms.RandomCrop(img_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # set normalize
    return transform

# 여러 transform 함수 구현

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.tensor(label), 'input': torch.tensor(input).float()}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean)/self.std

        data = {'label': label, 'input': input}

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
