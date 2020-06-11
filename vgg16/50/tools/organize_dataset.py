"""
this script is used for organize CUB200 dataset:
-- images
    -- train
        -- [:class 0]
        -- [:class 1]
            ...
        -- [:class n]
    -- val
        -- [:class 0]
        -- [:class 1]
            ...
        -- [:class n]
"""
import os
import PIL.Image
import numpy as np
import shutil


original_dataset_path = '/data/luojh/CUB_200_2011/CUB_200_2011'
dest_path = '/opt/luojh/Dataset/CUB/images'


def main():
    image_path = os.path.join(original_dataset_path, 'images/')
    # Format of images.txt: <image_id> <image_name>
    id2name = np.genfromtxt(os.path.join(
        original_dataset_path, 'images.txt'), dtype=str)
    # Format of train_test_split.txt: <image_id> <is_training_image>
    id2train = np.genfromtxt(os.path.join(
        original_dataset_path, 'train_test_split.txt'), dtype=int)

    for id_ in range(id2name.shape[0]):
        image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
        folder_name = id2name[id_, 1].split('/')[0]

        if id2train[id_, 1] == 1:
            # train
            save_path = os.path.join(dest_path, 'train', folder_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
                image.save(os.path.join(dest_path, 'train', id2name[id_, 1]))
            else:
                shutil.copyfile(os.path.join(image_path, id2name[id_, 1]),
                                os.path.join(dest_path, 'train', id2name[id_, 1]))
            image.close()

        else:
            # test
            save_path = os.path.join(dest_path, 'val', folder_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
                image.save(os.path.join(dest_path, 'val', id2name[id_, 1]))
            else:
                shutil.copyfile(os.path.join(image_path, id2name[id_, 1]),
                                os.path.join(dest_path, 'val', id2name[id_, 1]))
            image.close()


if __name__ == '__main__':
    main()
    print('finished')
