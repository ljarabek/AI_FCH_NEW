import os
import pickle
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer


dir1 = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/sigmoid_loss"
dir2 = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/ADAM_better"
process_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/ADAM_sigmoid"
os.makedirs(process_dir, exist_ok=True)

pkls1 = dir1 + "/pickles"
pkls2 = dir2 + "/pickles"

try:
    with open(os.path.join(process_dir, "matches.pkl"), "rb") as f:
        match_list = pickle.load(f)
except:
    match_list = list()
    for file1 in os.listdir(pkls1):
        for file2 in os.listdir(pkls2):
            with open(os.path.join(pkls1, file1), "rb") as f:
                pkl1 = pickle.load(f)
            with open(os.path.join(pkls2, file2), "rb") as f:
                pkl2 = pickle.load(f)
            if pkl1['info'][-1] == pkl2['info'][-1]:
                match_list.append((os.path.join(pkls1, file1), os.path.join(pkls2, file2)))
    with open(os.path.join(process_dir, "matches.pkl"), "wb") as f:
        pickle.dump(match_list, f)


def unpkl(unpickled):
    CT = unpickled['CT']
    PET = unpickled['PET']
    output = unpickled['output']
    zeros = np.zeros_like(CT)
    PET_T = np.maximum(zeros, PET - output[2])
    return CT, PET, PET_T

i=0
for _p1, _p2 in match_list:
    i+=1
    if i!=35:
        continue
    with open(_p1, "rb") as f:
        p1 = pickle.load(f)
    with open(_p2, "rb") as f:
        p2 = pickle.load(f)
    CT1, PET1, PET_T1 = unpkl(p1)
    CT2, PET2, PET_T2 = unpkl(p2)
    print(PET_T2.shape)
    print(PET_T1.shape)
    a1 = PET_T1>0.9
    a2 = PET_T2>2
    a =np.array(a1*a2, dtype=np.int)
    print(p1['info'])
    print(p2['info'])
    seg_viewer(CT1, a)



