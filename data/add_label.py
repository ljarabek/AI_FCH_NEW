import numpy as np

def clear_nones(l):
    to_delete = list()
    for i, e in enumerate(l):
        if e == None:
            to_delete.append(i)
    for i in reversed(to_delete):
        del l[i]
    return l

def make_one_hot_encoded(case, encode=('histo_lokacija', ["DS", "DZ", "LS", "LZ", "healthy"])):
    label = np.zeros(len(encode[1]), dtype=np.float32)
    try:
        indx = encode[1].index(case[encode[0]])
        label[indx] = 1.0
        case['label'] = label
        return case
    except ValueError:
        print("%s not among categories of %s" % (case[encode[0]], encode[1]))
        return None


def add_label(master, encoding=('histo_lokacija', ["DS", "DZ", "LS", "LZ", "healthy"])):
    for i, e in enumerate(master):
        new_e = make_one_hot_encoded(e, encode=encoding)
        master[i] = new_e
    master = clear_nones(master)
    return master