import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
# from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
# from absl import logging

from .gat import VisualRelationshipStream
from .places import get_places_model
from .resnet import resnet34


class MultiOutputModel(nn.Module):
    def __init__(self, device: int, n_age_cat: int, n_gender_cat: int) -> None:
        super(MultiOutputModel, self).__init__()

        self.device = device
        # ---------- Situational Context Branch---------------------
        # Objects Stream
        self.objects_feat_extractor = models.resnet34(pretrained=True)
        self.objects_feat_extractor.fc = nn.Linear(512, 256)

        # Event Stream
        model_context = get_places_model(device)

        self.event_feat_extractor = nn.Sequential(*(list(model_context.children())[:-1]))
        self.num_places_scene = list(model_context.children())[-1].in_features

        # Relationship Stream
        self.relationship_feat_extractor = VisualRelationshipStream(device)
        self.gat_pool_max = nn.MaxPool1d(2)
        self.gat_pool_avg = nn.AvgPool1d(2)

        self.sc_pool = nn.MaxPool1d(2)
        self.sc_out = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # ------------- Target Branch ----------------------
        self.target_base = resnet34(
            pretrained=True,
            last_conv_stride=1,
            last_conv_dilation=1)

        self.pg_target_avg = nn.AvgPool1d(4)

        # --------- Output Layers -----------------
        feat_in_age_gen = 256
        feat_out_age_gen = 128
        feat_in_classifer_age_gen = 64

        self.age_out = nn.Sequential(
            nn.Linear(feat_in_age_gen, feat_out_age_gen),
            nn.BatchNorm1d(feat_out_age_gen),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_out_age_gen, feat_in_classifer_age_gen),
            nn.BatchNorm1d(feat_in_classifer_age_gen),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_in_classifer_age_gen, n_age_cat)
        )

        self.gender_out = nn.Sequential(
            nn.Linear(feat_in_age_gen, feat_out_age_gen),
            nn.BatchNorm1d(feat_out_age_gen),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_out_age_gen, feat_in_classifer_age_gen),
            nn.BatchNorm1d(feat_in_classifer_age_gen),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_in_classifer_age_gen, n_gender_cat)
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, context, target, pose_masks):

        batch = target.size()[0]

        # Objects Features
        obj_feat = self.objects_feat_extractor(context)
        obj_feat = F.relu(obj_feat.view(batch, -1))

        # Event Features
        event_feat = self.event_feat_extractor(context)
        event_feat = F.relu(event_feat.view(-1, self.num_places_scene))

        # Visual Relationship Features
        rel_feat = self.relationship_feat_extractor(context, target)
        rel_feat = rel_feat.view(batch, 1, -1)
        rel_feat_max = self.gat_pool_max(rel_feat).squeeze(1)
        rel_feat_avg = self.gat_pool_avg(rel_feat).squeeze(1)
        rel_feat = F.relu(
            torch.cat([rel_feat_max, rel_feat_avg], axis=1))

        # Target Features
        # Pose-guided feature extraction proposed in
        # Miao, Jiaxu, et al. "Pose-guided feature alignment for occluded person re-identification." ICCV 2019
        resnet_feat = self.target_base(target)
        pg_target_feat = torch.FloatTensor().to(self.device)
        for i in range(0, 18):  # There are 18 pose landmarks for an image
            mask = pose_masks[:, i, :, :]/255
            mask = torch.unsqueeze(mask, 1)
            mask = mask.expand_as(resnet_feat)
            pg_feature_ = mask*resnet_feat  # element-wise multiplication
            pg_feature_ = nn.AdaptiveAvgPool2d((1, 1))(pg_feature_)
            pg_feature_ = torch.squeeze(pg_feature_, dim=2)
            pg_target_feat = torch.cat((pg_target_feat, pg_feature_), 2)
        pg_target_feat = nn.AdaptiveMaxPool1d(1)(pg_target_feat)
        pg_target_feat = pg_target_feat.squeeze(2)
        pg_target_feat = pg_target_feat.unsqueeze(1)
        pg_target_feat = self.pg_target_avg(pg_target_feat).squeeze(1)

        # --------------- Classifiers -------------------------
        sc_feat = F.relu(
            torch.cat([obj_feat, rel_feat, event_feat], 1))

        # Reducing situational context features size for age and gender
        sc_feat = self.sc_pool(sc_feat.unsqueeze(1)).squeeze(1)
        sc_feat = self.sc_out(sc_feat)
        concat_feat = torch.cat([pg_target_feat, sc_feat], dim=1)

        out_age = self.age_out(concat_feat)
        out_gender = self.gender_out(concat_feat)

        return {
            'age': out_age,
            'gender': out_gender
        }
