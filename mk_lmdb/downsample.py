from PIL import Image
import os
import time

def png_folder_downsample_bicubic(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    source_list = os.listdir(source_folder)
    print(source_list[:10])
    nSamples = len(source_list)
    for i in range(nSamples): 
        imageFile = source_list[i]
        img = Image.open(os.path.join(source_folder, imageFile))
        img_lr = img.resize((96, 96), Image.BICUBIC)
        img_lr.save(os.path.join(target_folder, imageFile[3:]))


if __name__ == '__main__':

    start_time = time.time()
    test_sr192_dir = '/media/cz/Data/Data/pcam_dataset/pcam_SuperResolution_dataset/pcam_superresolution_png/test_sr192/'
    train_sr192_dir = '/media/cz/Data/Data/pcam_dataset/pcam_SuperResolution_dataset/pcam_superresolution_png/train_sr192'
    test_lr96bicubic_dir = '/media/cz/Data/Data/pcam_dataset/pcam_SuperResolution_dataset/pcam_superresolution_png/test_lr96_from192_bicubic_20200222/'
    train_lr96bicubic_dir = '/media/cz/Data/Data/pcam_dataset/pcam_SuperResolution_dataset/pcam_superresolution_png/train_lr96_from192_bicubic_20200222/'
    png_folder_downsample_bicubic(source_folder=test_sr192_dir, target_folder=test_lr96bicubic_dir)
    png_folder_downsample_bicubic(source_folder=train_sr192_dir, target_folder=train_lr96bicubic_dir)
    
    print('total time:\t', (time.time()-start_time)/60)

