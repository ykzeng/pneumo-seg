import pandas as pd
import glob
import numpy as np
import sys
from tqdm import tqdm
import pydicom
from utils import rle2mask


def preproc(input_img, input_csv, limit=-1):
    input_glob = input_img+"*/*/*.dcm"
    train_fns = sorted(glob.glob(input_glob))[:limit]
    df_full = pd.read_csv(input_csv, index_col='ImageId')

    im_height = 1024
    im_width = 1024
    im_chan = 1
    # Get train images and masks
    X_train = np.zeros(
        (len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_fns), im_height, im_width, 1), dtype=np.bool)
    print('Getting train images and masks ... ')
    sys.stdout.flush()
    strCount, arrCount, noMaskCount, minus1Count = 0, 0, 0, 0

    # iterate every training dicom file
    for n, _id in tqdm(enumerate(train_fns), total=len(train_fns)):
        # read image from this dicom file
        dataset = pydicom.read_file(_id)
        # expand the image pixel vector into image pixel matrix
        X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
        try:
            # retrieve the encoded pixels by: df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']
            tempRle = df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']
            if '-1' in tempRle:
                # if a -1 is marked in training rle set for this image
                # need to mark every pixel as 1
                Y_train[n] = np.zeros((1024, 1024, 1))
                minus1Count += 1
            else:
                if type(tempRle) == str:
                    Y_train[n] = np.expand_dims(
                        rle2mask(tempRle, 1024, 1024), axis=2)
                    strCount += 1
                else:
                    Y_train[n] = np.zeros((1024, 1024, 1))
                    for x in tempRle:
                        Y_train[n] = Y_train[n] + \
                            np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
                    arrCount += 1
    #                 print(_id)
        except KeyError:
            print(
                f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
            # Assume missing masks are empty masks.
            Y_train[n] = np.zeros((1024, 1024, 1))
            noMaskCount += 1

    print('Done!')
    print('strCount:'+str(strCount))
    print('arrCount:'+str(arrCount))
    print('noMaskCount:'+str(noMaskCount))
    print('minus1Count:'+str(minus1Count))
    im_height = 128

    im_width = 128
    X_train = X_train.reshape((-1, im_height, im_width, 1))
    Y_train = Y_train.reshape((-1, im_height, im_width, 1))
    return X_train, Y_train
