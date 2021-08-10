from torch import nn


class MultiTaskLoss_DPAC(nn.Module):
    def __init__(self):
        super(MultiTaskLoss_DPAC, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, net_output, ground_truth):
        gender_loss = self.ce_loss(
            net_output['gender'], ground_truth['gender'])
        age_loss = self.ce_loss(net_output['age'], ground_truth['age'])
        emotion_loss = self.ce_loss(
            net_output['emotion'], ground_truth['emotion'])
        loss = gender_loss + age_loss + emotion_loss

        return loss, {'age': age_loss.item(), 'gender': gender_loss.item(), 'emotion': emotion_loss.item()}


class MultiTaskLoss_EMOTIC(nn.Module):
    def __init__(self):
        super(MultiTaskLoss_EMOTIC, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.emotion_criterion = nn.MultiLabelSoftMarginLoss(
            reduction='mean')  # for multi-label emotion

    def forward(self, net_output, ground_truth):
        gender_loss = self.ce_loss(
            net_output['gender'], ground_truth['gender'])
        age_loss = self.ce_loss(net_output['age'], ground_truth['age'])
        emotion_loss = self.emotion_criterion(
            net_output['emotion'], ground_truth['emotion'])

        loss = gender_loss + age_loss + emotion_loss
        return loss, {'age': age_loss.item(), 'gender': gender_loss.item(), 'emotion': emotion_loss.item()}


class MultiTaskLoss_IOG(nn.Module):
    def __init__(self):
        super(MultiTaskLoss_IOG, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, net_output, ground_truth):
        gender_loss = self.ce_loss(
            net_output['gender'], ground_truth['gender'])
        age_loss = self.ce_loss(net_output['age'], ground_truth['age'])
        loss = gender_loss + age_loss

        return loss, {'age': age_loss.item(), 'gender': gender_loss.item()}


class MultiTaskLoss_CAER(nn.Module):
    def __init__(self):
        super(MultiTaskLoss_CAER, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, net_output, ground_truth):
        emotion_loss = self.ce_loss(
            net_output['emotion'], ground_truth['emotion'])
        return emotion_loss
