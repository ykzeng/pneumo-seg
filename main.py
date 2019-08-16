#!/usr/bin/env python
from optparse import OptionParser
import glob
import pandas as pd
from preproc import preproc


def main():
    parser = OptionParser(usage="usage: python3 %prog [options]",
                          version="%prog 1.0")
    parser.add_option("--trainDir",
                      action="store",
                      dest="trainDir",
                      default="../data/dicom-images-train/",
                      help="pick the directory that stores all the training DICOM data")
    parser.add_option("--trainCsv",
                      action="store",
                      dest="trainCsv",
                      default="../data/train-rle.csv",
                      help="pick the csv file that stores training masks")
    # FIXME commented out for now until the model ran success
    # parser.add_option("--testDir",
    #                   action="store",
    #                   dest="testDir",
    #                   default="../data/dicom-images-test",
    #                   help="pick the directory that stores all the testing DICOM data")
    # we do not seem to have the testing set ground truth
    # parser.add_option("--testCsv",
    #                   action="store",
    #                   dest="testCsv",
    #                   default="../data/train-rle.csv",
    #                   help="pick the csv file that stores training masks")
    (options, args) = parser.parse_args()
    train_dir = options.trainDir
    train_csv = options.trainCsv

    train_glob = train_dir+"*/*/*.dcm"
    print("train_dir:"+train_dir)
    print("train_csv:"+train_csv)
    # train_fns = sorted(glob.glob(train_glob))[:500]
    # df_full = pd.read_csv(train_csv, index_col='ImageId')
    x, y = preproc(train_dir, train_csv, limit=50)


if __name__ == '__main__':
    main()
