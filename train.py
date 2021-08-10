import os
import argparse
from tqdm import tqdm
from rich import print

import torch
import torch.nn as nn

from dataloader import load_data
from models.base import MultiOutputModel
from loss import MultiTaskLoss_DPAC

from constants import DPAC_ATT_CAT_COUNT

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def model_checkpoint_save(model, name, epoch):
    '''
    Args:
        model (MultiOutputModel)
        name (str): directory name to save the checkpoint in 
        epoch (int): file name of the checkpoint
    '''
    f = os.path.join(name, '{}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def train(num_epochs, batch_size, learning_rate, weight_decay, ob_face_region, ob_people, gpu_device):
    '''
    Args:
        num_epochs (int):
        batch_size (int):
        learning_rate (float):
        weight_decay (float): 
        ob_face_region (str):
        ob_people (str):
        gpu_device (int):
    '''

    device = torch.device("cuda:" + str(gpu_device)
                          if torch.cuda.is_available() else "cpu")
    print("cuda is_available:", torch.cuda.is_available())
    print("cuda current_device", torch.cuda.current_device())

    train_dataloader = load_data(
        batch_size, ob_face_region, ob_people, 'train')

    model = MultiOutputModel(device, n_age_cat=DPAC_ATT_CAT_COUNT['age'],
                             n_gender_cat=DPAC_ATT_CAT_COUNT['gender'], n_emotion_cat=DPAC_ATT_CAT_COUNT['emotion'])

    model.to(device)
    loss = MultiTaskLoss_DPAC()

    # TODO: check this later
    clip_value = 10
    nn.utils.clip_grad_value_(model.parameters(), clip_value)
    nn.utils.clip_grad_value_(loss.parameters(), clip_value)

    model_params = (list(loss.parameters()) + list(model.parameters()))

    optimizer = torch.optim.SGD(
        model_params, lr=learning_rate,  momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=8,  gamma=0.1)
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'max', patience=20)
    # TODO: Do you want to make dataset configurable
    dataset = 'dpac'
    cp_dir_name = 'cp_{}_{}_{}'.format(dataset, ob_face_region, ob_people)

    cp_save_dir = os.path.join(cp_dir_name)
    if not os.path.isdir(cp_save_dir):
        os.makedirs(cp_dir_name)

    print("Total model parameters: {}".format(sum(p.numel()
                                                  for p in model.parameters() if p.requires_grad)))

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0
        running_loss_age = 0
        running_loss_gender = 0
        running_loss_emotion = 0

        print("--------------- Current Epoch: {} / {} --------------------".format(epoch, num_epochs))
        print("Current Learning Rate: {}".format(
            optimizer.param_groups[0]['lr']))

        for batch in tqdm(train_dataloader):
            target_pose = batch['pose'].to(device)
            target = batch['target'].to(device)
            context = batch['context'].to(device)
            target_att_labels = batch['labels']
            target_att_labels = {t: target_att_labels[t].to(
                device) for t in target_att_labels}

            output = model(context, target, target_pose)

            loss_train, losses_train = loss(output, target_att_labels)

            batch_len = len(batch)
            running_loss += loss_train
            running_loss_age += losses_train['age']/batch_len
            running_loss_gender += losses_train['gender']/batch_len
            running_loss_emotion += losses_train['emotion']/batch_len

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        print("Running loss age: {}".format(running_loss_age))
        print("Running loss gender: {}".format(running_loss_gender))
        print("Running loss emotion: {}".format(running_loss_emotion))

        if epoch > 20:
            print("-------------Saving the Checkpoint-----------------------")
            model_checkpoint_save(model, cp_dir_name, epoch)

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=40,
                        help="number of epochs to train for")
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay -- value between [0, 1]')

    parser.add_argument('--ob_face_region', type=str,
                        default=None, help='face region to obfuscated')
    parser.add_argument('--ob_people', type=str, default=None,
                        help='TO: Target obfuscated, AO: All obfuscated')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='GPU device to train the model on')

    # TODO: Add support for the EMOTION, IoG and CAER-S datasets
    # parser.add_argument('--dataset', type=str, default="dpac")

    args = parser.parse_args()
    train(num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.lr, weight_decay=args.weight_decay,
          ob_face_region=args.ob_face_region, ob_people=args.ob_people, gpu_device=args.gpu_device)
