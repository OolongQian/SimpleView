import torch
import torch.nn as nn
from dgcnn.pytorch.model import get_graph_feature
from models.model_utils import Squeeze, BatchNormPoint
from models.mv_utils import PCViews
import torch.nn.functional as F


class DgcnnAttentionFusion(nn.Module):

    def __init__(self, task, dataset, backbone, feat_size):
        super(DgcnnAttentionFusion, self).__init__()
        self.task = task
        self.dataset = dataset

        self.dropout_p = 0.5
        self.feat_size = feat_size

        self.pc_views = PCViews()
        self.num_views = self.pc_views.num_views

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size * 4)
        self.img_model = nn.Sequential(*img_layers)

        class Args:
            def __init__(self):
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.3
                self.leaky_relu = 1

        args = Args()
        self.edge_conv1 = EdgeConvBlock(args=args, in_channel=6, out_channel=64)
        self.edge_conv2 = EdgeConvBlock(args=args, in_channel=64 * 2, out_channel=64)
        self.edge_conv3 = EdgeConvBlock(args=args, in_channel=64 * 2, out_channel=64)

        self.fusion1 = AttentionFusionBlock(args=args, in_channel=64 * 2, out_channel=64, mv_in_channel=512)
        self.fusion2 = AttentionFusionBlock(args=args, in_channel=64 * 2, out_channel=128, mv_in_channel=256)

        self.mv_embed = nn.Linear(512, 256)
        self.pt_final = nn.Conv1d(128, 256, kernel_size=1, stride=1)

        self.final_fc = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 40)
        )

    def forward(self, pc, cls=None):
        B = pc.shape[0]

        pc = pc.to(next(self.parameters()).device)
        img = self.get_img(pc)
        mv_feat = self.img_model(img)

        pc = pc.permute(0, 2, 1).contiguous()
        pt_feat = pc
        pt_feat = self.edge_conv1(pt_feat)
        pt_feat = self.edge_conv2(pt_feat)
        pt_feat = self.edge_conv3(pt_feat)

        # do maxpooling over mv_feat.
        mv_feat = torch.reshape(mv_feat, shape=(B, self.num_views, -1))
        mv_feat = torch.max(mv_feat, dim=1)[0]

        # tile and repeat mv feat.
        num_points = pt_feat.shape[-1]
        pt_feat = self.fusion1(pt_feat, mv_feat)
        mv_feat = self.mv_embed(mv_feat)
        pt_feat = self.fusion2(pt_feat, mv_feat)

        pt_feat = self.pt_final(pt_feat)
        pt_feat = torch.max(pt_feat, dim=2)[0]

        final_fuse = torch.cat([pt_feat, mv_feat], dim=1)
        logit = self.final_fc(final_fuse)

        return {'logit': logit}

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


class EdgeConvBlock(nn.Module):
    """
    this code is adapted from DGCNN.
    """

    def __init__(self, args, in_channel, out_channel):
        super(EdgeConvBlock, self).__init__()
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                                  self.bn,
                                  act_mod(**act_mod_args))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class AttentionFusionBlock(EdgeConvBlock):
    def __init__(self, args, in_channel, out_channel, mv_in_channel):
        super(AttentionFusionBlock, self).__init__(args, in_channel, out_channel)
        self.attention_nn = nn.Conv1d(in_channel // 2 + mv_in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, pc_feat, view_feat):
        """
        pc_feat: (B, feat_size, num_points).
        view_feat: (B, view_feat).
        Args:
            pc_feat:
            view_feat:

        Returns:
        """
        # first
        # tile view feat.
        num_points = pc_feat.shape[-1]
        view_feat_tile = torch.unsqueeze(view_feat, dim=-1).repeat(1, 1, num_points)
        fuse_feat = torch.cat([pc_feat, view_feat_tile], dim=1)
        fuse_attention = self.attention_nn(fuse_feat)

        # quantize this attention, the range is [0, 1].
        fuse_attention = F.sigmoid(torch.log(torch.abs(fuse_attention) + 1e-10))
        # rescale it to [-1, 1].
        fuse_attention = fuse_attention * 2 - 1
        pc_feat = super(AttentionFusionBlock, self).forward(pc_feat)
        refined_pt_feat = pc_feat + pc_feat * fuse_attention
        return refined_pt_feat


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
