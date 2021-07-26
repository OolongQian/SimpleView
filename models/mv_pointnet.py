"""
This is a simple fusion between pointnet and multiview CNN.
"""

import torch
import torch.nn as nn
from all_utils import DATASET_NUM_CLASS
from models.model_utils import Squeeze, BatchNormPoint
from models.mv_utils import PCViews
from pointnet_pyt.pointnet.model import PointNetfeat


class MVPointNet(nn.Module):
    def __init__(self, task, dataset, backbone,
                 feat_size):
        super(MVPointNet, self).__init__()

        # sucheng: MV backbone.
        assert task == 'cls'
        self.task = task
        self.num_class = DATASET_NUM_CLASS[dataset]
        # sucheng: should this dropout_p be standardized?
        self.dropout_p = 0.5
        self.feat_size = feat_size

        self.pc_views = PCViews()
        self.num_views = self.pc_views.num_views

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)

        self.img_model = nn.Sequential(*img_layers)

        # self.final_fc = MVFC(
        #     num_views=self.num_views,
        #     in_features=in_features,
        #     out_features=self.num_class,
        #     dropout_p=self.dropout_p)

        # sucheng: PointNet backbone.
        feature_transform = True
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_class)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # sucheng: MV head.
        #   this is slightly different from MV or pointnet.
        self.mv_head = nn.Sequential(
            BatchNormPoint(in_features),
            # dropout before concatenation so that each view drops features independently
            nn.Dropout(self.dropout_p),
            nn.Flatten())
        self.fuse_head = nn.Sequential(
            nn.Linear(in_features=in_features * self.num_views + 1024,
                      out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(in_features=in_features, out_features=self.num_class,
                      bias=True))

    def forward(self, pc):
        """
        :param pc:
        :return:
        """
        pc = pc.cuda()
        img = self.get_img(pc)
        # img shape (B * V, 1, H, W).
        mv_feat = self.img_model(img)
        mv_feat = torch.reshape(mv_feat, shape=(pc.shape[0], self.num_views, -1))
        # do some mysterious dropout.
        mv_feat = self.mv_head(mv_feat)
        mv_feat = torch.reshape(mv_feat, shape=(pc.shape[0], -1))

        # get pointnet feature.
        # PointNetfeat class gives global point feature.
        pc = pc.transpose(2, 1).float()
        pn_feat, _, _ = self.feat(pc)

        # then, consider the head.
        fuse_feat = torch.cat([mv_feat, pn_feat], dim=1)
        logit = self.fuse_head(fuse_feat)
        out = {'logit': logit}
        return out

    def get_img(self, pc):
        img = self.pc_views.get_img(pc)
        img = torch.tensor(img).float()
        img = img.to(next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]

        # we can see that this is actually a very small net.
        # the output global dimension is just 128.
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features


class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
            BatchNormPoint(in_features),
            # dropout before concatenation so that each view drops features independently
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(in_features=in_features * self.num_views,
                      out_features=in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=in_features, out_features=out_features,
                      bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        out = self.model(feat)
        return out
