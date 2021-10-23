from data.CSV import get_master_list
from data.add_label import add_label
from constants.constants import *
from data.convert_folder_to_array import convert_folder_to_array, image_from_folder
from data.interpolation import interpolate2d
import numpy as np
from multi_slice_viewer.multi_slice_viewer_last import seg_viewer
from multi_slice_viewer.slice_viewer import multi_slice_viewer
import torchio as tio
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


m_list = get_master_list()

# add labels
for i, m in enumerate(m_list):
    label = np.zeros(shape=5)
    label[4] = 1
    if "D" in m['histo_lokacija']:
        label[0] = 1
    if "L" in m['histo_lokacija']:
        label[1] = 1
    if "Z" in m['histo_lokacija']:
        label[2] = 1
    if "S" in m['histo_lokacija']:
        label[3] = 1
    if m['histo_lokacija'] == "healthy":
        label[4] = 0
    m_list[i]['label'] = label

# add images:

"""slist = list()
for i, m in tqdm(enumerate(m_list)):
    CT = image_from_folder(m['CT_dir'])

    PET = image_from_folder(m['PET_dir'])
    transform = tio.transforms.Resample(target=PET, image_interpolation="bspline")

    m['CT'] = transform(CT)
    m['PET'] = PET
    subject = tio.Subject(m)
    subject.check_consistent_affine()
    subject.check_consistent_orientation()
    subject.check_consistent_space()
    # subject.plot()
    slist.append(subject)

with open("master_dump.pkl", "wb") as f:
    pickle.dump(slist, f)  # torch.save FIX!"""
with open("master_dump.pkl", "rb") as f:
    slist = pickle.load(f)
if __name__ == '__main__':
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
        p=0.75,
    )
    transforms = [rescale, spatial]
    transform = tio.Compose(transforms)


    for s in slist:
        CT, PET = s['CT'][tio.DATA], s['PET'][tio.DATA]
        CT = CT.detach().cpu().numpy()
        PET = PET.detach().cpu().numpy()
        CT= CT.ravel()
        plt.hist(CT, bins=50)
        plt.yscale('log', nonposy='clip')
        plt.show()
        break
        PET.ravel()

