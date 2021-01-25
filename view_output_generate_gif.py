import os
import pickle
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer
import matplotlib.pyplot as plt
import imageio
from time import sleep
import json
from tqdm import tqdm


def save_as_gif(im, seg, save_dir, pet_p=None, ct_p=None):
    i = 0
    p = 0.2
    im_min, im_max = np.percentile(im, p), np.percentile(im, 100 - p)
    if pet_p is not None:
        seg_min, seg_max = np.percentile(seg, pet_p), np.percentile(seg, 100 - pet_p)
    else:
        seg_min, seg_max = np.min(seg), np.max(seg)
    images = list()
    for volume, segmentation in zip(im, seg):
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.segmentation = segmentation
        ax.imshow(volume, cmap='gray', vmin=im_min, vmax=im_max)
        ax.imshow(segmentation, cmap='Reds', alpha=0.2, vmin=seg_min, vmax=seg_max)
        # plt.show()
        fname = os.path.join(save_dir, 'test_%s.png' % i)
        plt.savefig(fname)
        plt.close('all')
        images.append(imageio.imread(fname))
        # plt.close()
        i += 1
    imageio.mimsave(os.path.join(save_dir, "gif.gif"), images)


if __name__ == '__main__':
    main_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/v5"  # "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/v3l_better"  # /media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/sigmoid_loss" #"/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/ADAM_better"  # "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/sigmoid_loss"
    pickles_dir = main_dir + "/pickles"
    pkls = list()


    def getpkl(f: str):
        with open(f, "rb") as file:
            F = pickle.load(file)
        return F


    os.makedirs(main_dir + "/images/", exist_ok=True)
    # print(getpkl(pkls[0]))
    fw = ""
    for f in tqdm(os.listdir(pickles_dir)):
        p = os.path.join(pickles_dir, f)
        unpickled = getpkl(p)

        CT = unpickled['CT']
        PET = unpickled['PET']
        output = unpickled['output']
        zeros = np.zeros_like(CT)
        # print(output.shape)
        PET_T = np.maximum(zeros, PET - output[2])  # transformed PET with output to only see higher bound upliers
        info = unpickled['info']
        print(f, info)
        os.makedirs(main_dir + "/images/" + f, exist_ok=False)
        os.makedirs(main_dir + "/images/%s_PET" % f, exist_ok=False)

        fw += f
        for inf in info:
            fw += " , " + inf
        fw += "\n"
        # print(info)

        save_as_gif(CT, PET_T, main_dir + "/images/" + f, pet_p=0.02)
        save_as_gif(CT, PET, main_dir + "/images/%s_PET" % f, pet_p=0.2)
        # break

    file = open(os.path.join(main_dir, 'images', "identifier"), "w")
    file.write(fw)
