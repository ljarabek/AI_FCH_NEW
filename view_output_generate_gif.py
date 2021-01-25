import os
import pickle
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer
import matplotlib.pyplot as plt
import imageio
from time import sleep


def save_as_gif(im, seg, save_dir, pet_p=None, ct_p=None):
    os.makedirs(save_dir, exist_ok=False)
    fig, ax = plt.subplots()
    i = 0
    p = 0.2
    im_min, im_max = np.percentile(im, p), np.percentile(im, 100 - p)
    if pet_p is not None:
        seg_min, seg_max = np.percentile(seg, pet_p), np.percentile(seg, 100 - pet_p)
    else:
        seg_min, seg_max = np.min(seg), np.max(seg)
    images = list()
    for volume, segmentation in zip(im, seg):
        ax.volume = volume
        ax.segmentation = segmentation
        ax.imshow(volume, cmap='gray', vmin=im_min, vmax=im_max)
        ax.imshow(segmentation, cmap='Reds', alpha=0.2, vmin=seg_min, vmax=seg_max)
        # plt.show()
        fname = os.path.join(save_dir, 'test_%s.png' % i)
        plt.savefig(fname)
        images.append(imageio.imread(fname))
        # plt.close()
        i += 1
    imageio.mimsave(os.path.join(save_dir, "gif.gif"), images)


if __name__ == '__main__':
    main_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/abs_healthy"  # /media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/sigmoid_loss" #"/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/ADAM_better"  # "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/sigmoid_loss"
    pickles_dir = main_dir + "/pickles"
    pkls = list()
    for f in os.listdir(pickles_dir):
        pkls.append(os.path.join(pickles_dir, f))


    def getpkl(f: str):
        with open(f, "rb") as file:
            return pickle.load(file)


    # print(getpkl(pkls[0]))

    for p in pkls:
        unpickled = getpkl(p)

        CT = unpickled['CT']
        PET = unpickled['PET']
        output = unpickled['output']
        zeros = np.zeros_like(CT)
        # print(output.shape)
        PET_T = np.maximum(zeros, PET - output[2])  # transformed PET with output to only see higher bound upliers
        info = unpickled['info']
        print(info)
        query = "zora"
        if '/media/leon/2tbssd/PRESERNOVA/AI FCH/DICOM_all4mm/Solar_Erna_Zora/Pet_Choline_Obscitnica_2Fazalm_(Adult) - 1/AC_CT_Obscitnica_2' in info:  # Jelica...
            # seg_viewer(CT, PET_T)
            # seg_viewer(CT,PET)
            save_as_gif(CT, PET_T, main_dir + "/images/" + query, pet_p=0.02)
            save_as_gif(CT, PET, main_dir + "/images/%s_PET" % query, pet_p=0.2)
