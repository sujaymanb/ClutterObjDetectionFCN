import torch
import torch.utils.data as data
import glob
import os
import utils
import numpy as np

# TODO: validation
class SuctionData(data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # load file list
        #print(os.path.join(root_dir, 'data', self.mode, 'color', '*'))
        self.color_list = glob.glob(os.path.join(root_dir, 'data', self.mode, 'color', '*.png'))
        self.depth_list = glob.glob(os.path.join(root_dir, 'data', self.mode, 'depth', '*.png'))
        self.label_list = glob.glob(os.path.join(root_dir, 'data', self.mode, 'label', '*.png'))

        print(len(self.color_list),len(self.depth_list),len(self.label_list))


    def __len__(self):
        return len(self.color_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color = utils.rgb2array(self.color_list[idx],hwc=False)
        depth = utils.depth2array(self.depth_list[idx],hwc=False)
        if self.mode == 'train':
            label = utils.label2array(self.label_list[idx],hwc=False)
            sample = {'color': torch.from_numpy(color),
                      'depth': torch.from_numpy(depth),
                      'label': torch.from_numpy(label.astype(np.float32))}
        else:
            sample = {'color': torch.from_numpy(color),
                      'depth': torch.from_numpy(depth)}

        if self.transform:
            sample = self.transform(sample)

        return sample