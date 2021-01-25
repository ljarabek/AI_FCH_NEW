from data.convert_folder_to_array import convert_folder_to_array
import numpy as np
from data.interpolation import interpolate2d


def add_images(master_list: list, wanted_shape_ct=(200, 200)) -> list:
    print("adding images")
    for i, m in enumerate(master_list):
        pet_dir = m['PET_dir']
        ct_dir = m['CT_dir']
        ct = convert_folder_to_array(ct_dir)
        pet = convert_folder_to_array(pet_dir)
        #print(m['priimek'])
        #print(m['CT_dir'])
        #print(ct.shape)
        #print(m['PET_dir'])
        #print(pet.shape)
        final = np.zeros_like(pet)  # downsize to pet ; vsi CT in vsi PET so dejansko iste velikosti!!
        for i in range(pet.shape[0]):
            final[i] = interpolate2d(ct[i], new_size=wanted_shape_ct)
        ct = final
        #print(m['CT_dir'])
        #print(ct.shape)
        m['CT'] = ct
        m['PET'] = pet
        master_list[i] = m
    return master_list
