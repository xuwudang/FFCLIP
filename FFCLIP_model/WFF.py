import torch
import torch.nn as nn
import cv2
import numpy as np

class WFF(nn.Module):
    def __init__(self, channel=256):
        super(WFF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_f1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c1 = nn.Sequential(nn.Conv2d(2 * channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def fusion(self, f1, f2, f_vec):
        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self, rgb, edge_feature):
        Fr = self.conv_r1(rgb)
        Fe = self.conv_f1(edge_feature)
        f = torch.cat([Fr, Fe], dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        Fo = self.fusion(Fr, Fe, f)
        return Fo


def extract_edge_feature(image_tensor):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edge_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)

    edge_tensor = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).float()
    return edge_tensor


if __name__ == '__main__':
    batch_size = 1
    channels = 256
    height = 224
    width = 224

    wff = WFF(channel=channels)
    rgb_input = torch.randn(batch_size, channels, height, width)
    edge_feature = extract_edge_feature(rgb_input[:, :3, :, :])
    edge_feature = edge_feature.repeat(1, channels, 1, 1)

    output = wff(rgb_input, edge_feature)

    print("RGB 输入形状:", rgb_input.shape)
    print("边缘特征 输入形状:", edge_feature.shape)
    print("输出形状:", output.shape)
