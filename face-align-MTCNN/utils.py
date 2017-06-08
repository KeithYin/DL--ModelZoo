import os
import numpy as np
from subprocess import Popen, PIPE


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def store_revision_info(src_path, output_dir, arg_string):
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()

    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout=PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def get_dataset(paths):
    dataset = []
    # print(paths)
    classes = os.listdir(paths)
    classes.sort()
    # print(classes)
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(paths, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
            dataset.append(ImageClass(class_name, image_paths))

    return dataset

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret