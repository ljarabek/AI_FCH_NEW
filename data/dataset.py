import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser
from scipy.interpolate import RectBivariateSpline
from constants import *
import os
import numpy as np
import io
import csv
from data.CSV import get_master_list
from data.convert_folder_to_array import convert_folder_to_array
from data.interpolation import interpolate2d
from data.add_label import add_label
from data.add_images_to_list import add_images
from data.crop_images import crop_images
import re
from constants import *
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pickle
from multi_slice_viewer.multi_slice_viewer import seg_viewer

# DZ JIH MAMO LE 5!!!


try:
    with open(master_pkl_dir, "rb") as f:
        # 1/0
        settings, master_list = pickle.load(f)
    if settings != m_list_settings:
        i = 1 / 0
    print("master.pkl successfully loaded")
except:
    print("master.pkl doesnt exist or settings mismatch")
    print("generating new list")
    master_list = get_master_list()
    master_list = add_label(master_list, encoding=m_list_settings['encoding'])  # 0 1 2 3 4
    master_list = add_images(master_list, wanted_shape_ct=m_list_settings['wanted_shape_ct'])

    if "cropping" in m_list_settings:
        master_list = crop_images(master_list, cropping=m_list_settings['cropping'])

    with open(master_pkl_dir, "wb") as f:
        pickle.dump((m_list_settings, master_list), f)

means_std_pet = list()
means_std_ct = list()
for e in master_list:
    means_std_pet.append((np.mean(e['PET']), np.std(e['PET'])))
    means_std_ct.append((np.mean(e['CT']), np.std(e['CT'])))

ct_mean, ct_std = np.mean(means_std_ct, axis=0)
pet_mean, pet_std = np.mean(means_std_pet, axis=0)
print("mean and std calculated:")
print("CT mean, std: %s" % np.mean(means_std_ct, axis=0))
print("PET mean, std: %s" % np.mean(means_std_pet, axis=0))

means_std_ct = np.mean(means_std_ct, axis=0)
means_std_pet = np.mean(means_std_pet, axis=0)


class PET_CT_Dataset(Dataset):
    def __init__(self, master_list_=master_list, **kwargs):
        super(PET_CT_Dataset, self).__init__()
        self.master_list = master_list_
        # self.transform = transform #TODO:implement

    def __getitem__(self, idx):
        ct = self.master_list[idx]['CT']
        pet = self.master_list[idx]['PET']
        label = self.master_list[idx]['label']

        ct -= means_std_ct[0]
        ct /= means_std_ct[1]
        ct = np.expand_dims(ct, 0)

        pet -= means_std_pet[0]  # - mean
        pet /= means_std_pet[1]  # / std
        pet = np.expand_dims(pet, 0)
        merged = np.concatenate([ct, pet], 0)
        print(merged.shape)
        # if merged.shape != (2, 48, 128, 128):
        #    print(master_list[idx])
        return ct, pet, merged, label, self.master_list[idx]  # must be of shape (without N) (N,Cin​,D,H,W)

    def __len__(self):
        return len(self.master_list)


if __name__ == "__main__":
    from collections import Counter
    import torchio as tio

    p = PET_CT_Dataset()

    for i in p:
        # print(i)
        ct, pet, merged, label, _ = i
        ct = tio.ScalarImage(tensor=merged)

        print(ct.data)
        print(ct)
        print(ct.affine)
        print("success")
        break

    # for m in master_list:
    #    print(m)
    # print(merged.shape)
    # print(label.shape)
    # for i in range(len(p)):
    #    ct, pet, merged, label = p[i]
    # for i in master_list:
    #    print(i['CT'].shape)
    #    print(i['PET'].shape)

    # TODO: proper dataset split!!
