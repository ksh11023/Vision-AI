import os
from torch.utils.data import Dataset
from PIL import Image

class LotteDataset(Dataset):
  def __init__(self, data_root, train_mode, transform=None):
    super(LotteDataset, self).__init__()
    self.train_mode=train_mode
    self.transform = transform
    self.label2idx = {'dog': 0,
             'elephant': 1,
             'giraffe': 2,
             'guitar': 3,
             'horse': 4,
             'house': 5,
             'person': 6}

    if self.train_mode==False:

        self.img_list= []
        img_list = []
        for file in os.listdir(data_root):
            file_root = os.path.join(data_root, file)
            for data in os.listdir(file_root):
                data_path = os.path.join(file_root, data)
                img_list.append(data_path)
        self.img_list = sorted(img_list)

    else: #학습할 때
        self.img_list = []
        self.label_list=[]
        img_list = []
        for file in os.listdir(data_root):
            file_root = os.path.join(data_root, file)
            for data in os.listdir(file_root):
                data_path = os.path.join(file_root, data)
                img_list.append(data_path)
        self.img_list = sorted(img_list)

        for label in self.img_list:
            label = label.split('/')[-2]
            self.label_list.append(self.label2idx[label])

        print()


  def __len__(self):
      return len(self.img_list)

  def __getitem__(self, index):
    img_path = self.img_list[index]

    #Image, Label Loading
    if self.train_mode:
        label = self.label_list[index]

    img = Image.open(img_path)

    #Augmentation
    if self.transform:
      img = self.transform(img)

    if self.train_mode:
      return img,label
    else:
      return img

if __name__=='__main__':
    data_root ='/Users/sungheui/PycharmProjects/basic/dataset/train'

    lot = LotteDataset(data_root, True)
