import SimpleITK as sitk
import numpy as np


def convert_folder_to_array(sick_directory: str) -> np.ndarray:
    # depth = len(os.listdir(sick_directory))
    # blank = np.zeros()
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(sick_directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    arr = np.array(arr)
    return arr


