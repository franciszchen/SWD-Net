import os
import time
import pickle
import lmdb
import numpy as np
import random
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))

def get_loader(lmdb_dir, batch_size, stage, num_workers, bi='bilinear'):
    """
    This is to generate input_data(96*96) and output_target(192*192)
    No upsample
    For final_up network
    """

    transform_data = transforms.ToTensor()
    transform_target = transforms.ToTensor()

    if stage == 'train':
        dataset = LMDB_Dataset(lmdb_dir,transform_data, transform_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=False)
    else:
        dataset = LMDB_Dataset(lmdb_dir,transform_data, transform_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=False)
    return dataloader

class LMDB_Dataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_dir, transform_data=None, transform_target=None):

        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(lmdb_dir, max_readers=10, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        cache_file = os.path.join(lmdb_dir, 'cache')# should add cache_dir
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file,'rb'))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key,_ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file,'wb'))

        self.transform_data = transform_data
        self.transform_target = transform_target


    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            dictbuf_bytes = txn.get(self.keys[index])

        dictbuf = pickle.loads(dictbuf_bytes)# imageDict has two keys 'image' 'label'
        data_buf = dictbuf['data']
        target_buf = dictbuf['target']

        img_data = BytesIO()
        img_data.write(data_buf)
        img_data.seek(0)
        image_data = np.asarray(Image.open(img_data))
        img_target = BytesIO()
        img_target.write(target_buf)
        img_target.seek(0)
        image_target = np.asarray(Image.open(img_target))

        image_data = image_data.astype(np.float32) / 255.0
        image_data = pil2tensor(image_data, np.float32)
        image_target = image_target.astype(np.float32) / 255.0
        image_target = pil2tensor(image_target, np.float32)

        return image_data, image_target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__+'('+self.lmdb_dir+')'
















