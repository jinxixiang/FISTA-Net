import os
import argparse
import numpy as np
import pydicom
import torch
from torch_radon import Radon

def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patients_list = sorted([d for d in os.listdir(args.data_path) if 'zip' not in d])
    print(patients_list)
    # patient dir
    for p_ind, patient in enumerate(patients_list):
        patient_path = os.path.join(args.data_path, patient,
                                           "full_{}mm".format(args.mm))
        print(patient_path)
        rotView = 720
        ds_factor = [12, 6]
        theta = np.linspace(0.0, 180.0, rotView, endpoint=False)
        for path_ in [patient_path]:
            full_pixels = get_pixels_hu(load_scan(path_))
            for pi in range(len(full_pixels)):
                                
                f = normalize_(full_pixels[pi], args.norm_range_min, args.norm_range_max)
                
                # clip to circle
                lx, ly = f.shape
                X, Y = np.ogrid[0:lx, 0:ly]
                mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
                f[mask] = 0

                # instantiate Radon Transform
                radon = Radon(512, theta, clip_to_circle=True)

                device = torch.device('cuda:3')
                img = torch.FloatTensor(f).to(device)
                sinogram = radon.forward(img)
                sinogram = sinogram.squeeze()
                sinogram = sinogram.cpu().detach().numpy()

                for ds in ds_factor:
                    # downsampling
                    sinogram_ds = sinogram[0:rotView:ds, :]
                
                    io = '_ds' + str(ds) + '_target'
                    f_name = '{}_{}_{}.npy'.format(patient, pi, io)
                    np.save(os.path.join(args.save_path, f_name), f)
                    io = '_ds' + str(ds) + '_input'
                    f_name = '{}_{}_{}.npy'.format(patient, pi, io)
                    np.save(os.path.join(args.save_path, f_name), sinogram_ds)
                

        printProgressBar(p_ind, len(patients_list),
                         prefix="save image ..",
                         suffix='Complete', length=25)
        print(' ')


def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/media/ps/D/LDCT')
    parser.add_argument('--save_path', type=str, default='/media/ps/D/npy_img2')

    parser.add_argument('--test_patient', type=str, default='L096')
    parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)
