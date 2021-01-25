from models.unet import UNet3D
import torch.nn as nn
import torch.functional as F
from resnets.resnet import resnet10
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import multi_slice_viewer.multi_slice_viewer
import torch


class MyModel(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super(MyModel, self).__init__()
        self.num_classes = num_classes

        self.unet = UNet3D(in_channel=2, n_classes=1)  # nima sigmoida na koncu!
        self.classifier = resnet10(num_classes=self.num_classes, activation="softmax")
        self.activation = nn.Tanh()  # nn.Sigmoid()

    def forward(self, x):
        # for i in range(23, 32, 1):
        #    slice_to_show = i
        #    plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        #    plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        #    plt.show()
        #    plt.close("all")
        show_input = x.cpu().detach().numpy()
        x1 = self.unet(x)
        x1 = self.activation(x1) + 1.
        show_att = x1.cpu().detach().numpy()


        # plt.imshow(show_att[1,0,5])
        # plt.imshow(show_input[1, 0, 5], alpha=0.2, cmap="Greys")
        # plt.show()
        ones = torch.ones_like(x1)
        x1 = torch.cat([ones, x1], 1)  # concatenate along channel dimension, where

        x_input = x1 * x  # with modified PET
        x_ = x_input.cpu().detach().numpy()

        #Najprej poišči na PET
        multi_slice_viewer.multi_slice_viewer.seg_viewer(show_input[1, 0], show_input[1, 1], cmap_="jet")

        #Prikaži masko
        multi_slice_viewer.multi_slice_viewer.seg_viewer(show_input[1, 0], show_att[1, 0])

        #Maskiran PET
        multi_slice_viewer.multi_slice_viewer.seg_viewer(show_input[1, 0], x_[1, 1], cmap_="jet")
        # seg_viewer(x_[0,0], x_[0,1])#x_[0,1])
        # plt.imshow(show_input[0,0,0], cmap="Greys_r")
        # plt.imshow(x_[0,1,0], alpha=0.5)
        # plt.show()
        # slice_to_show = 5
        # plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        # plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        # plt.show()
        # plt.close("all")
        classified = self.classifier(x_input)
        return classified  # , ()


class MyModel2(nn.Module):  # TODO: make it tresholded :)
    def __init__(self, num_classes=5, **kwargs):
        super(MyModel2, self).__init__()
        self.num_classes = num_classes

        self.unet = UNet3D(in_channel=2, n_classes=1)  # nima sigmoida na koncu!
        self.classifier = resnet10(num_classes=self.num_classes, activation="softmax")
        self.activation = nn.Tanh()  # nn.Sigmoid()

    def forward(self, x):
        # for i in range(23, 32, 1):
        #    slice_to_show = i
        #    plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        #    plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        #    plt.show()
        #    plt.close("all")
        show_input = x.cpu().detach().numpy()
        x1 = self.unet(x)
        x1 = self.activation(x1) + 1.
        show_att = x1.cpu().detach().numpy()
        ones = torch.ones_like(x1)
        x1 = torch.cat([ones, x1], 1)  # concatenate along channel dimension, where

        x_input = x1 * x  # with modified PET
        x_ = x_input.cpu().detach().numpy()
        # seg_viewer(x_[0,0], x_[0,1])#x_[0,1])
        # plt.imshow(show_input[0,0,0], cmap="Greys_r")
        # plt.imshow(x_[0,1,0], alpha=0.5)
        # plt.show()
        # slice_to_show = 5
        # plt.imshow(x_[0, 0, slice_to_show], cmap="Greys_r")  # 0 je CT
        # plt.imshow(x_[0, 1, slice_to_show], alpha=0.3)
        # plt.show()
        # plt.close("all")
        classified = self.classifier(x_input)
        return classified  # , ()
