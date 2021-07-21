import os
from cv2 import data
import numpy as np
from rich import print
from PIL import ImageFile

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from constants import NORM_MEAN, NORM_STD

DATA_PATH = '/home/a/anshu/Groups/processed/Groups/'  # './data/'
NUM_WORKERS = 12  # no. of subprocesses to use for data loading
torch.manual_seed(0)

# FIX: Truncated Image Error
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, x_context, x_target, x_pose, y_age, y_gender, transform):
        super(BaseDataset, self).__init__()
        self.x_context = x_context
        self.x_target = x_target
        self.x_pose = x_pose

        self.y_age = y_age
        self.y_gender = y_gender

        self.transform = transform
        self.norm = transforms.Normalize(NORM_MEAN, NORM_STD)

    def __len__(self):
        return len(self.y_age)

    def __getitem__(self, index):
        context = self.x_context[index]
        target = self.x_target[index]
        pose = torch.tensor(self.x_pose[index], dtype=torch.float32)

        age_label = self.y_age[index]
        gender_label = self.y_gender[index]

        if self.transform:
            target = self.transform(target)
            context = self.transform(context)

        if self.norm:
            context = self.norm(context)
            target = self.norm(target)

        data = {
            'target': target,
            'context': context,
            'pose': pose,
            'labels': {
                'age': age_label,
                'gender': gender_label
            }
        }
        return data


def load_data(batch_size, ob_face_region=None, ob_people=None, datatype='train'):
    '''
    Args:
        batch_size (int):
        ob_face_region (str): {none, eye, lower, face, head}
        ob_people (str): {none, target, all}
        datatype (str): {train, test}
    Return:
        dataloader (Dataloader): 
    '''
    # assert datatype in ['train', 'test']
    # assert ob_face_region in [None, 'eye', 'lower', 'face', 'head']
    # assert ob_people in [None, 'AO', 'TO']

    # if ob_people and not ob_face_region:
    #     ob_face_region = "face"  # setting default ob_face_region as face

    # if ob_face_region and not ob_people:
    #     ob_people = "AO"  # setting default ob_people as AO -- All Obfuscated

    # print("Loading {} data for:\n Obfuscated Face Region: {} \n Obfuscated People: {}".format(
    #     datatype.upper(), ob_face_region, ob_people))

    # obfus_path = None
    # intact_path = None
    # if datatype == 'test' or ob_face_region == None or ob_people == "TO":  # testing is done on intact faces
    #     intact_path = DATA_PATH
    #     # TODO: repalce with the ideal scenario
    #     # intact_path = DATA_PATH + "intact/"
    # if datatype == 'train' and ob_face_region:
    #     obfus_path = DATA_PATH + "privacy/{}/".format(ob_face_region)
    #     # TODO
    #     # obfus_path = DATA_PATH + "privacy/{}/".format(ob_face_region)

    if datatype == 'train':
        data_context = np.load(os.path.join(DATA_PATH, 'context_arr.npy'))
        data_target = np.load(os.path.join(DATA_PATH, 'target_arr.npy'))
        data_age = np.load(os.path.join(DATA_PATH, 'age_arr.npy'))
        data_gender = np.load(os.path.join(DATA_PATH, 'gender_arr.npy'))
        data_pose = np.load(os.path.join(DATA_PATH, 'pose_arr.npy'))


        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(
        ), transforms.RandomResizedCrop(400), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor()])
    else:
        data_context = np.load(os.path.join(DATA_PATH, 't_context_arr.npy'))
        data_target = np.load(os.path.join(DATA_PATH, 't_target_arr.npy'))
        data_age = np.load(os.path.join(DATA_PATH, 't_age_arr.npy'))
        data_gender = np.load(os.path.join(DATA_PATH, 't_gender_arr.npy'))
        data_pose = np.load(os.path.join(DATA_PATH, 't_pose_arr.npy'))


        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    dataset = BaseDataset(data_context, data_target, data_pose, data_age,
                                data_gender, transform)


    print('{} data loaded of size: {}'.format(datatype, len(dataset)))
    
    # if facing batch-size use drop_last=True argument
    dataloader = DataLoader(dataset, pin_memory=True, batch_size=batch_size,
                            shuffle=True, num_workers=NUM_WORKERS)

    return dataloader
