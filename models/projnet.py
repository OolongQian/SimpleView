"""
in this work, we use a pointnet++ architecture, the fused feature is
    hierarchically projected to multiview image.
"""

import torch
from all_utils import DATASET_NUM_CLASS
import torch.nn as nn
import torch.nn.functional as F
from models.mv_utils import PCViews
from pointnet2.models.pointnet2_ssg_cls import Pointnet2SSG
from models.model_utils import Squeeze, BatchNormPoint


class ProjNet(torch.nn.Module):
    """
    This ProjNet should refer to the existing PointNet and CNN model architecture as an ablation.
    Therefore making our argument stronger.
    """

    def __init__(self, dataset, task,
                 backbone,  # resnet.
                 feat_size  # some hyperparam inside the feature size -> this seems to be only feasible for resnet.
                 ):
        super(ProjNet, self).__init__()
        self.task = task
        self.num_class = DATASET_NUM_CLASS[dataset]

        # pointnet++ architecture.
        self.pointnet2 = Pointnet2SSG(self.num_class, input_channels=0, use_xyz=True)

        # views.
        self.pc_views = PCViews()
        self.num_views = self.pc_views.num_views

        img_layers, in_features = self.get_img_layers(backbone, feat_size)
        # get some list of its feature.
        # each layer corresponds to some level of abstraction.
        self.img_layer0 = nn.ModuleList(img_layers[:4])
        self.img_layer1 = nn.ModuleList(img_layers[4:5])
        self.img_layer2 = nn.ModuleList(img_layers[5:6])
        self.img_layer3 = nn.ModuleList(img_layers[6:7])
        self.final_layer = nn.ModuleList(img_layers[7:])

        self.batch_tmp0 = nn.BatchNorm2d(4)
        self.proj_nn0 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.proj_nn1 = nn.Sequential(
            nn.Conv2d(129, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.proj_nn2 = nn.Sequential(
            nn.Conv2d(257, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.fuse_nn0 = nn.Sequential(
            nn.Conv2d(16 * 2, 16 * 2, kernel_size=1, stride=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.fuse_nn1 = nn.Sequential(
            nn.Conv2d(32 * 2, 32 * 2, kernel_size=1, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fuse_nn2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 2, kernel_size=1, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # final cnn layer.
        self.dropout_p = 0.5
        self.final_fc = MVFC(
            num_views=self.num_views,
            in_features=in_features,
            out_features=self.num_class,
            dropout_p=self.dropout_p)

    def forward(self, pc):
        """
        :param pc:
        :return:
        """
        # ok, the point cloud are in here.
        pc = pc.cuda()
        B = pc.shape[0]

        # (B, 1024, 3), None.
        xyz0, features0 = self.pointnet2._break_up_pc(pc)
        # (B, 512, 3), (B, 320, 512).
        xyz1, features1 = self.pointnet2.SA_modules[0](xyz0, features0)
        # (B, 128, 3), (B, 640, 128).
        xyz2, features2 = self.pointnet2.SA_modules[1](xyz1, features1)

        # hierarchical pointnet feature extraction and image construction.
        #   all in full resolution.
        img0 = self.get_img(pc)
        # img1 = self.get_img(xyz1)
        # img2 = self.get_img(xyz2)
        feat_img0 = self.get_featured_img(xyz0, xyz0 - torch.mean(xyz0, dim=1, keepdim=True))
        feat_img1 = self.get_featured_img(xyz1, features1.transpose(1, 2))
        feat_img2 = self.get_featured_img(xyz2, features2.transpose(1, 2))
        print(feat_img2.shape)
        exit(0)

        # sucheng: shit, the image quality is so fucking low.
        # img = img2
        # img = torch.reshape(img, shape=(B, self.num_views, img.shape[-3], img.shape[-2], img.shape[-1]))
        # from PIL import Image
        # import numpy as np
        # for i in range(self.num_views):
        #     this_img = img[0, i, 0, ...].detach().cpu().numpy()
        #     this_img = this_img / (np.max(this_img) - np.min(this_img))
        #     this_img -= np.min(this_img)
        #     this_img = Image.fromarray(np.uint8(this_img * 255), 'L')
        #     this_img.save(f'{i}.png')

        # build image feature pyramid.
        def list_nn_forward(nns, input):
            out = input
            for layer in nns:
                out = layer(out)
            return out

        # prepare for the fuse, use a linear layer to project pointnet feature in the same subspace of image feature.
        # print(feat_img0.shape, feat_img1.shape, feat_img2.shape)
        # exit(0)
        feat_img0 = self.proj_nn0(feat_img0)
        feat_img1 = self.proj_nn1(feat_img1)
        feat_img2 = self.proj_nn2(feat_img2)

        # (1, 128, 128).
        imfeat0 = img0
        # (16, 128, 128).
        imfeat1 = list_nn_forward(self.img_layer0, imfeat0)
        print(imfeat1.shape)
        fuse_feat1 = self.fuse_nn0(torch.cat([imfeat1, feat_img0], dim=1))
        print(fuse_feat1.shape)
        exit(0)
        # (32, 64, 64).
        imfeat2 = list_nn_forward(self.img_layer1, fuse_feat1)
        fuse_feat2 = self.fuse_nn1(torch.cat([imfeat2, F.max_pool2d(feat_img1, kernel_size=2)], dim=1))
        # (64, 32, 32).
        imfeat3 = list_nn_forward(self.img_layer2, fuse_feat2)
        fuse_feat3 = self.fuse_nn2(torch.cat([imfeat3, F.max_pool2d(feat_img2, kernel_size=4)], dim=1))
        imfeat4 = list_nn_forward(self.img_layer3, fuse_feat3)

        final_feat = list_nn_forward(self.final_layer, imfeat4)

        logit = self.final_fc(final_feat)
        out = {'logit': logit}
        return out

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
            feature_size=feat_size * 2,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
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

    def get_img(self, pc):
        img = self.pc_views.get_img(pc)
        img = torch.tensor(img).float()
        img = img.to(next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    def get_featured_img(self, pc, feat):
        feat_img = self.pc_views.get_featured_img(pc, feat)
        return feat_img


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
