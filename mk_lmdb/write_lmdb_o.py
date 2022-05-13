import os
# import pdb
import sys
import lmdb
import io
import pickle
import random
import time

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def write_lmdb(lr_folder, sr_folder, lmdb_path):
    total_img = 0
    # folders_list = sorted(os.listdir(lr_folder))

    if not os.path.exists(lmdb_path):
        os.mkdir(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    txn = env.begin(write=True)
    
    # for subfolder_idx, folder_item in enumerate(folders_list):
    # print(subfolder_idx, folder_item)
    # make_val
    # subfolder = read_val_dir + '/' + folder_item
    # make train
    # folder_item = 'test'
    # subfolder = read_train_dir + '/' + folder_item

    lr_list = os.listdir(lr_folder)
    lr_list.sort()
    print(lr_list[:10])
    random.shuffle(lr_list)
    print(lr_list[:10])
    # print(imgs_list)
    # pdb.set_trace()
    nSamples = len(lr_list)
    # lmdb_path = folder+'_lmdb'
     
    cache = {}
    cnt = 1
    for i in range(nSamples): 
        imageFile = lr_list[i]
        with open(os.path.join(lr_folder,imageFile),'rb') as f:
            imageBin_lr = f.read()
        with open(os.path.join(sr_folder,'SR_'+imageFile),'rb') as f:
            imageBin_sr = f.read()

        imageKey = '{:0>6d}-'.format(cnt)+ imageFile
        imageDict = {'data':imageBin_lr, 'target': imageBin_sr} # cat=0 dog=1
        imageDictBytes = pickle.dumps(imageDict)
        cache[imageKey] = imageDictBytes
        #pdb.set_trace()
        # if cnt % 1000 == 0:
        #     writeCache(env,cache)
        #     cache = {}
        #     print('written {}/{}    subfolder {}/999'.format(cnt,nSamples, subfolder_idx))
        cnt += 1
    # writeCache(env,cache)
    print(len(cache))
    for k, v in cache.items():
        txn.put(k.encode(), v)
    
    # if subfolder_idx % 10 ==0:
    txn.commit()
    txn = env.begin(write=True)
    total_img += nSamples
    txn.commit()
    env.close()
    print('Created dataset with {} sample in {}'.format(total_img,lmdb_path))
          
if __name__ == '__main__':
    
    start_time = time.time()
    ###
    input_dir = '/media/cz/Data/Data/pcam_dataset/pcam_SuperResolution_dataset/pcam_superresolution_png/'
    train_lr_dir = input_dir + 'train_lr96_from192_bicubic_20200222/'
    train_sr192_dir = input_dir + 'train_sr192/'
    test_lr_dir = input_dir + 'test_lr96_from192_bicubic_20200222/'
    test_sr192_dir = input_dir + 'test_sr192/'

    save_dir = '/media/cz/DATA/pcam_LR2SR192_20200222/bicubic/'
    train_save_dir = save_dir + 'train_lmdb/'
    test_save_dir = save_dir + 'test_lmdb/'

    write_lmdb(test_lr_dir, test_sr192_dir, test_save_dir)
    write_lmdb(train_lr_dir, train_sr192_dir, train_save_dir)


    print('total time:\t', (time.time()-start_time)/60)









    