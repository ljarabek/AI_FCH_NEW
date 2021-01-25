import csv
from constants import *
import io
from data.sumniki import popravi_sumnike
import numpy as np
import os
import re
import SimpleITK as sitk
import pydicom


def get_csv_new():  # makes csv with indices from indices variable
    entries = list()
    ind = list()
    with open(indices, "r") as f:
        for line in f.readlines(): ind.append(line[:-1])
    with io.open(csv_path, newline="", encoding="utf-8") as f:  # utf encoding due to čšđžć
        reader = csv.reader(f, quotechar="|")  # delimiter=" "
        for row in reader:
            entry = dict()
            for idr, el in enumerate(row):
                if ind[idr] in entry:
                    if entry[ind[idr]] == "":
                        entry[ind[idr]] = popravi_sumnike(el)
                else:
                    entry[ind[idr]] = popravi_sumnike(el)
            if entry['CT_dir'] != '': entries.append(entry)
    return entries


def healthy_cases_list(dr=images_path_healthy) -> list:  # makes dict of healthy cases with indices from indices
    cases = list()
    ind = list()
    with open(indices, "r") as f:
        for line in f.readlines(): ind.append(line[:-1])

    for case in os.listdir(dr):
        # check if valid:
        valid = True
        for folder in os.listdir(os.path.join(dr, case)):
            if len(os.listdir(os.path.join(dr, case, folder))) > 60:
                valid = False
        if not valid:
            continue
        key = case[:case.find(".")]
        cases.append(dict())
        for ind_ in ind:
            cases[-1][ind_] = "healthy"
        for folder in os.listdir(os.path.join(dr, case)):
            f_o = os.listdir(os.path.join(dr, case, folder))[0]
            file = pydicom.read_file(os.path.join(dr, case, folder, f_o))
            size_ = file[0x0008103E].value
            if "4" in size_:
                cases[-1]["PET_dir"] = os.path.join(dr, case, folder)
                # cases[-1]["PET"] = convert_folder_to_array(os.path.join(dr, case, folder))
            if "CT" in size_:
                cases[-1]["CT_dir"] = os.path.join(dr, case, folder)
                # cases[-1]["CT"] = convert_folder_to_array(os.path.join(dr, case, folder))  # KEYI SO CT_ ali PT_...

    return cases


def get_master_list() -> list:
    master_list = get_csv_new()
    master_list.extend(healthy_cases_list())
    return master_list


# print(len(master_list)) 102!! YEY :D


def make_folders_from_healthy():  # runs only once,
    from shutil import copy
    zdravi_dir = "/media/leon/2tbssd/PRESERNOVA/PREŠERNOVA_ZDRAVI/"
    for case in os.listdir(zdravi_dir):
        for file in os.listdir(os.path.join(zdravi_dir, case)):
            print(re.split(r'\.', string=file))
            spl = re.split(r'\.', string=file)
            fits_in_dir = os.path.join(zdravi_dir, case, spl[1] + "_" + spl[2])
            # print(fits_in_dir)
            os.makedirs(fits_in_dir, exist_ok=True)
            copy(src=os.path.join(zdravi_dir, case, file), dst=os.path.join(fits_in_dir, file))
            print("copied source\n%s\nto destination\n%s" % (
                os.path.join(zdravi_dir, case, file), os.path.join(fits_in_dir, file)))
            os.remove(os.path.join(zdravi_dir, case, file))


def get_csv():  # deprecated
    csv_ = list()
    with io.open(csv_path, newline="", encoding="utf-8") as f:  # utf encoding due to čšđžć
        reader = csv.reader(f, delimiter=" ", quotechar="|")

        for i, row in enumerate(reader):
            row[0] = popravi_sumnike(row[0])
            if i == 0:
                keys = row[0].split(";")
            else:
                csv_.append(row[0].split(";"))
    patients_ = dict()
    for i in range(110):
        patients_[str(i)] = dict()
    for ida, attr in enumerate(keys):
        attr = popravi_sumnike(attr)
        for patient in csv_:
            if ida > 13:  # csv le do "histo" opombe
                continue

            else:
                try:
                    if "ID" in attr:
                        attr = "ID"
                    patients_[patient[0]][attr] = popravi_sumnike(patient[ida])
                except:
                    print(patient)
    patients = dict()
    for p in patients_:
        if patients_[p] != dict():
            patients[p] = patients_[p]
    return patients
