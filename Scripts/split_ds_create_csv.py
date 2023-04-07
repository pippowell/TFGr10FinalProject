### Python script to split image dataset into Train, Validation, and Test folders.

# Randomly splits images to 90% train, 5% validation, and 5% test, and moves them to their respective folders.

# library
from pathlib import Path
import random
import os
import sys
import pandas as pd

# same set of random numbers are generated each time
random.seed(42)

def split_dataset(train_percent=0.9, val_percent=0.05):

    # current working directory
    pth = str(Path.cwd())

    # Path needs to be Absolute path
    image_path = pth + '/TFGr10FinalProject/Datasets/Grapes Dataset/Images'
    train_path = pth + '/images/train'
    val_path = pth + '/images/validation'
    test_path = pth + '/images/test'
    csv_path = pth + '/TFGr10FinalProject/Datasets/Grapes Dataset/grapes_ds_edited.csv'


    print(Path(image_path))

    csvfile = pd.read_csv(csv_path)

    # Get list of all images
    file_list = [path for path in Path(image_path).rglob('*.jpg')]
    print(file_list)

    file_num = len(file_list)
    print('Total images: %d' % file_num)

    train_num = int(file_num*train_percent)
    val_num = int(file_num*val_percent)
    test_num = file_num - train_num - val_num
    print('Images moving to train: %d' % train_num)
    print('Images moving to validation: %d' % val_num)
    print('Images moving to test: %d' % test_num)

    # Select 90% of files randomly and move them to train folder

    for i in range(train_num):
        # choose a random image path from file_list
        rand_img = random.choice(file_list)
        # file name
        file_name = rand_img.name
        # indices of csv file corresponding to file_name
        indices = csvfile.index[csvfile['filename'] == file_name].tolist()
        # data rows
        selected_rows = csvfile.iloc[indices]
        # path for new csv file
        new_csv_path = pth + '/images/train_labels.csv'

        # check if the CSV file exists and if it's empty
        if os.path.isfile(new_csv_path):
            # if the file exists append the new rows without the header
            selected_rows.to_csv(new_csv_path, mode='a', header=False, index=False)
        else:
            # if the file does not exist or is empty, write the header and the new rows
            selected_rows.to_csv(new_csv_path, mode='w', header=True, index=False)

        # move the image to train_path folder
        os.rename(rand_img, train_path + '/' + file_name)
        # remove the rand_img from file_list
        file_list.remove(rand_img)

    # Select 5% of remaining files and move them to validation folder

    for i in range(val_num):
        rand_img = random.choice(file_list)  # choose a random image path from file_list
        file_name = rand_img.name
        indices = csvfile.index[csvfile['filename'] == file_name].tolist()
        selected_rows = csvfile.iloc[indices]
        new_csv_path = pth + '/images/validation_labels.csv'

        # check if the CSV file exists and if it's empty
        if os.path.isfile(new_csv_path):
            # if the file exists append the new rows without the header
            selected_rows.to_csv(new_csv_path, mode='a', header=False, index=False)
        else:
            # if the file does not exist or is empty, write the header and the new rows
            selected_rows.to_csv(new_csv_path, mode='w', header=True, index=False)

        os.rename(rand_img, val_path + '/' + file_name)
        file_list.remove(rand_img)

    # Move remaining files to test folder

    for i in range(test_num):
        rand_img = random.choice(file_list)
        file_name = rand_img.name
        os.rename(rand_img, test_path + '/' + file_name)
        file_list.remove(rand_img)


split_dataset()

