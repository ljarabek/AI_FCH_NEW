from copy import deepcopy


def crop_images(master: list, cropping: tuple) -> list:
    (slice_0, d_slice), (x_o, d_x), (y_o, d_y) = cropping
    print("cropping images")
    new_list = list()
    for i, e in enumerate(master):
        new_e = deepcopy(e)  # TODO: doesn't work without it? why? don't know...
        new_ct = e['CT'][slice_0 - d_slice:slice_0 + d_slice,
                 x_o - d_x: x_o + d_x, y_o - d_y: y_o + d_y]
        new_pet = e['PET'][slice_0 - d_slice:slice_0 + d_slice,
                  x_o - d_x: x_o + d_x, y_o - d_y: y_o + d_y]

        #print(e['CT_dir'])
        #print(f'oldct shape: {master[i]["CT"].shape}')
        #print(f'oldPET shape: {master[i]["PET"].shape}')
        ## master[i]['CT'] = new_ct
        # master[i]['PET'] = new_pet

        # print(f'NEWct shape: {master[i]["CT"].shape}')
        # print(f'NEWPET shape: {master[i]["PET"].shape}')

        new_e['CT'] = new_ct
        new_e['PET'] = new_pet
        new_list.append(new_e)
        print(f'NEWct shape: {new_e["CT"].shape}')
        print(f'NEWPET shape: {new_e["PET"].shape}')

        # print(e['priimek'])
        # print(e['CT_dir'])
        # print(master[i]['CT'].shape)
        # print(e['PET_dir'])
        # print(master[i]['PET'].shape)
    return new_list
