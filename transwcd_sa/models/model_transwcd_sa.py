import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer




class TransWCD_dual(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)


        # Difference Modules
        self.diff_c4 = conv_diff_d(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)

    def get_param_groups(self):

        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    ###
    def forward(self, x1, x2, cam_only=False, return_feat=False, ):
        _x1 = self.encoder(x1)
        _x2 = self.encoder(x2)

        _, _, _, _c4_1 = _x1
        _, _, _, _c4_2 = _x2

        ### difference module ###
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4

        cls_x4 = self.pooling(_c4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        _c4 = self.dropout(_c4)     #
        #pred_enc = self.linear_pred(_c4)
        ###
        if return_feat:
            return cls_x4, _c4
        return cls_x4


class TransWCD_single(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        #self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)


        # Difference Modules
        self.diff_c4 = conv_diff_s(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)

    def get_param_groups(self):

        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)


        return param_groups

    def forward(self, x1, x2, cam_only=False, return_feat=False):
        ### difference module ###
        # diff(A,B)
        # x = self.diff(torch.cat((x1, x2), dim=1))    #nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # _x = self.encoder(x)

        #  A-B
        x1 = x1[:, :, :, :]
        x2 = x2[:, :, :, :]
        x = torch.absolute(x1 - x2)
        _x = self.encoder(x)

        _, _, _, _c4 = _x

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4

        cls_x4 = self.pooling(_c4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        _c4 = self.dropout(_c4)     #
        #pred_enc = self.linear_pred(_c4)
        if return_feat:
            return cls_x4, _c4
        return cls_x4


if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    transwcd = TransWCD_dual('mit_b1', num_classes=2, embedding_dim=256, pretrained=True)
    transwcd._param_groups()
    dummy_input = torch.rand(2, 3, 256, 256)
    transwcd(dummy_input)


# conv 3*3
def conv_diff_d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

# conv 1*1 w/o relu
def conv_diff_s(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #nn.ReLU(),
    )


