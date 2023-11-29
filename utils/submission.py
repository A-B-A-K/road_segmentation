import matplotlib.image as mpimg
import numpy as np
import re

def mask_to_submission_strings(image_filename, t=0.25):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)

    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = value_to_class(patch, t)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames, t=0.25):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, t))

def value_to_class(v, foreground_threshold=0.25):
    '''
    Decide whether a patch is foreground or background based on the percentage of pixels > 1
        :param v: the percentage of pixels > 1
        :param foreground_threshold: the threshold (percentage of pixels > 1 required to assign a foreground label to a patch)
        :return: 1 for foreground and 0 for background
    '''
    # print(f'unique = {np.unique(v)}')
    # print(f'shape = {v.shape}')
    df = np.sum(v)/256
    # print(f'df = {df}')
    if df > foreground_threshold:
        return 1
    else:
        return 0