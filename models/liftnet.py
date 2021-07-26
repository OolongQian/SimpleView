import torch
from all_utils import DATASET_NUM_CLASS
import torch.nn as nn
from models.model_utils import Squeeze


class LiftNet(torch.nn.Module):
    """
    This LiftNet should refer to the existing PointNet and CNN model architecture as an ablation.
    Therefore making our argument stronger.
    """

    def __init__(self, dataset, task,
                 backbone,  # resnet.
                 feat_size  # some hyperparam inside the feature size -> this seems to be only feasible for resnet.
                 ):
        super(LiftNet, self).__init__()
        self.task = task
        num_class = DATASET_NUM_CLASS[dataset]
        img_layers, in_features = self.get_img_layers(backbone, feat_size)
        self.img_model = nn.Sequential(*img_layers)
        # get some list of its feature.
        # each layer corresponds to some level of abstraction.
        self.img_layer0 = list(self.img_model.children())[:4]
        self.img_layer1 = list(self.img_model.children())[4:5]
        self.img_layer2 = list(self.img_model.children())[5:6]
        self.img_layer3 = list(self.img_model.children())[6:7]
        self.final_layer = list(self.img_model.children())[7:]

    def forward(self, pc):
        """
        :param pc:
        :return:
        """
        # ok, the point cloud are in here.
        pc = pc.cuda()
        img = self.get_img(pc)
        print(pc.shape)
        exit(0)

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
