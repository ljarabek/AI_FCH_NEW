import random


def sample_by_label(lst: list, val_size: int, n_min=2,
                    label_tag="label") -> list:  # labels need to already be one_hot encoded
    no_labels = len(lst[0][label_tag])

    dct_ = dict()  # dictionary of labels: list of samples
    for i in range(no_labels):
        dct_[i] = list()

    for i in range(no_labels):
        for ie, e in enumerate(lst):
            # print(e[label_tag])
            if e[label_tag][i] == 1.:
                dct_[i].append(ie)

    fraction = val_size / len(lst)
    #print(fraction)
    val_indices = list()

    for i in dct_:
        i_frac = int(fraction * len(dct_[i]))
        #print(i_frac)
        if i_frac < n_min: i_frac = n_min
        #print(i, dct_[i], i_frac)
        val_indices.extend(random.sample(dct_[i], k=i_frac))
        #
    print(f"validation set{val_indices}")
    return val_indices
    # for i, e in enumerate(lst):
