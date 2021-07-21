import torch
import warnings
import argparse
import numpy as np
from rich import print

from constants import IOG_ATT_CAT_COUNT
from dataloader_iog import load_data
from models.base_iog import MultiOutputModel
# from loss import MultiTaskLoss_IOG

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


def _net_output_to_predictions(output):
    '''
    '''
    _, predicted_age = output['age'].cpu().max(1)
    _, predicted_gender = output['gender'].cpu().max(1)
    return predicted_age.numpy().tolist(), predicted_gender.numpy().tolist()


def _model_checkpoint_load(model, name):
    '''
    '''
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))


def _calculate_metrics(target, output):
    '''
    '''
    predicted_age = output['age']
    gt_age = target['age']

    predicted_gender = output['gender']
    gt_gender = target['gender']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        recall_age = recall_score(
            y_true=gt_age, y_pred=predicted_age, average='weighted')
        precision_age = precision_score(
            y_true=gt_age, y_pred=predicted_age, average='weighted')
        f1_age = f1_score(y_true=gt_age, y_pred=predicted_age,
                          average='weighted')
        accuracy_age = accuracy_score(y_true=gt_age, y_pred=predicted_age)

        recall_gender = recall_score(
            y_true=gt_gender, y_pred=predicted_gender, average='weighted')
        precision_gender = precision_score(
            y_true=gt_gender, y_pred=predicted_gender, average='weighted')
        f1_gender = f1_score(
            y_true=gt_gender, y_pred=predicted_gender, average='weighted')
        accuracy_gender = accuracy_score(
            y_true=gt_gender, y_pred=predicted_gender)

    print("Accuracy Age: {:.4f}, Gender: {:.4f}".format(
        accuracy_age, accuracy_gender))
    print("Precision Age: {:.4f}, Gender: {:.4f}".format(
        precision_age,  precision_gender))
    print("Recall Age: {:.4f}, Gender: {:.4f}".format(
        recall_age, recall_gender))
    print("F1 Age: {:.4f}, Gender: {:.4f}".format(
        f1_age, f1_gender))


def test(checkpoint=None, gpu_device=0):
    '''
    '''
    device = torch.device("cuda:" + str(gpu_device)
                          if torch.cuda.is_available() else "cpu")
    # loss = MultiTaskLoss_IOG()

    model = MultiOutputModel(device, n_age_cat=IOG_ATT_CAT_COUNT['age'],
                             n_gender_cat=IOG_ATT_CAT_COUNT['gender'])
    model.to(device)

    test_dataloader = load_data(batch_size=16, datatype='test')

    if checkpoint is not None:
        _model_checkpoint_load(model, checkpoint)

    model.eval()

    age_predictions = []
    gender_predictions = []

    age_labels = []
    gender_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            target_pose = batch['pose'].to(device)
            target = batch['target'].to(device)
            context = batch['context'].to(device)
            target_att_labels = batch['labels']
            target_att_labels = {t: target_att_labels[t].to(
                device) for t in target_att_labels}

            output = model(context, target, target_pose)
            # _train, val_train_losses = loss(output, target_labels)
            (batch_age_predictions,
             batch_gender_predictions) = _net_output_to_predictions(output)

            age_labels.extend(target_att_labels['age'].cpu().numpy().tolist())
            gender_labels.extend(
                target_att_labels['gender'].cpu().numpy().tolist())

            age_predictions.extend(batch_age_predictions)
            gender_predictions.extend(batch_gender_predictions)

    target_dict = {"age": np.asarray(age_labels), "gender": np.asarray(
        gender_labels)}
    output_dict = {"age": np.asarray(age_predictions), "gender": np.asarray(
        gender_predictions)}

    _calculate_metrics(target_dict, output_dict)

    # model.train()


if __name__ == "__main__":
    # print("torch.cuda.device_count() ", torch.cuda.device_count())
    # print("cuda is_available:", torch.cuda.is_available())
    # print("cuda current_device", torch.cuda.current_device())
    # print("cuda get_device_name:", torch.cuda.get_device_name())
    # print("cuda memory_allocated:", torch.cuda.memory_allocated())
    # print("cuda memory_reserved:", torch.cuda.memory_reserved())

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_path', type=str, default="/home/a/anshu/context-aware-attributes-prediction/cp_dpac_face_T/29.pth",
                        help='Checkpoint to test on', required=True)
    # gpu device to use
    parser.add_argument('--gpu_device', type=int,
                        help='GPU device to train the model on')
    args = parser.parse_args()

    test(args.cp_path, args.gpu_device)
