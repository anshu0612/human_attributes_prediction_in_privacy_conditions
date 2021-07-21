import cv2
import json
import numpy as np
from rich import print
from PIL import ImageFile

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from constants import NORM_MEAN, NORM_STD, DPAC_AGE_LABEL_TO_IDX, DPAC_GENDER_LABEL_TO_IDX, \
    DPAC_EMOTION_LABEL_TO_IDX, IMG_HEIGHT, IMG_WIDTH

#TODO: Update the data path
DATA_PATH = '/home/a/anshu/TRAIN_DATA/'  # './data/'
NUM_WORKERS = 12  # no. of subprocesses to use for data loading
torch.manual_seed(0)

# FIX: Truncated Image Error
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, x_context, x_target, x_pose, y_age, y_gender, y_emotion, transform):
        super(BaseDataset, self).__init__()
        self.x_context = x_context
        self.x_target = x_target
        self.x_pose = x_pose

        self.y_age = y_age
        self.y_gender = y_gender
        self.y_emotion = y_emotion

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
        emotion_label = self.y_emotion[index]

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
                'gender': gender_label,
                'emotion': emotion_label
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
    assert datatype in ['train', 'test']
    assert ob_face_region in [None, 'eye', 'lower', 'face', 'head']
    assert ob_people in [None, 'AO', 'TO']

    if ob_people and not ob_face_region:
        ob_face_region = "face"  # setting default ob_face_region as face

    if ob_face_region and not ob_people:
        ob_people = "AO"  # setting default ob_people as AO -- All Obfuscated

    print("Loading {} data for:\n Obfuscated Face Region: {} \n Obfuscated People: {}".format(
        datatype.upper(), ob_face_region, ob_people))

    with open(DATA_PATH + "data.json",  encoding='utf-8') as f:
        all_data = json.load(f)

    with open(DATA_PATH + "splits.json",  encoding='utf-8') as f:
        splits_data = json.load(f)

    obfus_path = None
    intact_path = None
    if datatype == 'test' or ob_face_region == None or ob_people == "TO":  # testing is done on intact faces
        intact_path = DATA_PATH
        # TODO: repalce with the ideal scenario
        # intact_path = DATA_PATH + "intact/"
    if datatype == 'train' and ob_face_region:
        obfus_path = DATA_PATH + "privacy/{}/".format(ob_face_region)
        # TODO
        # obfus_path = DATA_PATH + "privacy/{}/".format(ob_face_region)

    # labels
    data_gender = []
    data_emotion = []
    data_age = []
    # inputs
    data_context = []
    data_target = []
    data_pose = []

    # TODO: temp
    count = 0
    for img_id in splits_data[datatype]:

        # TODO: Fix these errorneous test images
        if img_id == "6383_1_3" or img_id == "7074_2_1":
            continue

        if count > 4:
            break
        count += 1
        img_data = all_data[img_id]

        # adding the age, gender and emotion labels
        data_age.append(DPAC_AGE_LABEL_TO_IDX[img_data['attributes']['age']])
        data_gender.append(
            DPAC_GENDER_LABEL_TO_IDX[img_data['attributes']['gender']])
        data_emotion.append(
            DPAC_EMOTION_LABEL_TO_IDX[img_data['attributes']['emotion']])

        # context -- entire image
        if intact_path:
            context_img = cv2.cvtColor(cv2.imread(
                intact_path + "images/" + img_id + ".jpg"), cv2.COLOR_BGR2RGB)

            dup_context_img = context_img.copy()  # duplicating for cropping a target

        # bounding box of a target
        body_cor = img_data['body_bb']

        if obfus_path:
            obfus_context_img = cv2.cvtColor(cv2.imread(
                obfus_path + img_id + ".jpg"), cv2.COLOR_BGR2RGB)
            #TODO: "images/"
            dup_context_img = obfus_context_img.copy()

            if ob_people == "TO":  # set only obfuscated target in the image
                context_img[abs(body_cor[1]):abs(body_cor[3]), abs(body_cor[0]):abs(
                    body_cor[2])] = obfus_context_img[abs(body_cor[1]):abs(body_cor[3]), abs(body_cor[0]):abs(body_cor[2])]

            if ob_people == "AO":  # use obfuscated images with for all
                context_img = obfus_context_img

        context_img = cv2.resize(context_img, (IMG_HEIGHT, IMG_WIDTH))
        data_context.append(context_img)

        target_img = dup_context_img[abs(body_cor[1]):abs(
            body_cor[3]), abs(body_cor[0]):abs(body_cor[2])]
        target_img = cv2.resize(target_img, (IMG_HEIGHT, IMG_WIDTH))
        data_target.append(target_img)

        # TODO: Update path for pose
        pose = np.load(DATA_PATH + "pose/" + img_id + ".npy")
        # pose = np.einsum('kli->ikl', pose)
        pose = np.reshape(pose, (24, 8, 18))
        pose = cv2.resize(pose, (25, 25))
        pose = np.einsum('kli->ikl', pose)
        data_pose.append(pose)

    if datatype == 'train':
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(
        ), transforms.RandomResizedCrop(400), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()])

    dataset = BaseDataset(data_context, data_target, data_pose, data_age,
                          data_gender, data_emotion, transform)

    print('{} data loaded of size: {}'.format(datatype, len(dataset)))
    # if facing batch-size use drop_last=True argument
    dataloader = DataLoader(dataset, pin_memory=True, batch_size=batch_size,
                            shuffle=True, num_workers=NUM_WORKERS)

    return dataloader
