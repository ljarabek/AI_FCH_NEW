import SimpleITK as sitk
import numpy as np
import torchio as tio


def convert_folder_to_array(sick_directory: str) -> np.ndarray:
    """
    deprecated
    :param sick_directory:
    :return:
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(sick_directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    arr = np.array(arr)
    return arr

def image_from_folder(folder_path:str)->tio.ScalarImage:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    im = tio.ScalarImage.from_sitk(image)
    return im




if __name__ == "__main__":
    convert_folder_to_array(
        "/media/leon/2tbssd/PRESERNOVA/SLIKE_bolni/Dolenc_Ida/Pet_Choline_Obscitnica_2Fazalm_(Adult) - 2/2mm_2_faza_4")
