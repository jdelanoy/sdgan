from collections import defaultdict
import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np
import random
import scipy.misc
from tqdm import tqdm

# TODO: should be able to use tf queue's for this somehow and not have to load entire dataset into memory. main problem is dynamically retrieving pairs of images from same image class IDs (CIDs) using tensorflow.
class HackyCIGANLoader(object):
    def __init__(self, batch_size, ninstances, data_format, cid_to_ims, h=64, w=64, c=3):
        self.data_format = data_format
        assert data_format in ['NCHW', 'NHWC']

        original_len = len(cid_to_ims)
        cid_to_ims = filter(lambda x: len(x[1]) > ninstances, cid_to_ims.items())
        cid_to_ims = {k:v for k,v in cid_to_ims}
        print '{}/{} CIDs have too few instances'.format(original_len - len(cid_to_ims), original_len)

        self.batch_size = batch_size
        self.ninstances = ninstances
        self.cids = cid_to_ims.keys()
        self.cid_to_ims = cid_to_ims
        self.h = h
        self.w = w
        self.c = 3

        if self.data_format == 'NHWC':
            self.x = tf.placeholder(tf.float32, [batch_size, h, w, ninstances * c], 'x')
        else:
            self.x = tf.placeholder(tf.float32, [batch_size, ninstances * c, h, w], 'x')

    def placeholder(self):
        return self.x

    def one_batch(self):
        _x = np.empty((self.batch_size, self.h, self.w, self.ninstances * self.c), dtype=np.float32)
        for b in xrange(self.batch_size):
            cid = random.choice(self.cids)
            cid_ims = self.cid_to_ims[cid]
            random_instances = random.sample(cid_ims, self.ninstances)
            stacked_channels = np.concatenate(random_instances, axis=2)
            _x[b] = stacked_channels

        if self.data_format == 'NCHW':
            _x = np.transpose(_x, [0, 3, 1, 2])

        return _x

    def iter_forever(self, batch_size, ninstances):
        while True:
            yield self.one_batch(batch_size, ninstances)

def get_loader(root, batch_size, ninstances, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    print root

    dataset_name = os.path.basename(root)
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    cid_to_fps = defaultdict(list)
    for fp in paths:
        fid = os.path.splitext(os.path.basename(fp))[0]
        cid, _, ratio = fid.rsplit('_', 2)

        cid_to_fps[cid].append(fp)

    cid_to_ims = defaultdict(list)
    for cid, fps in tqdm(cid_to_fps.items()):
        for fp in fps:
            im = scipy.misc.imread(fp)
            im = im.astype(np.float32)
            cid_to_ims[cid].append(im)

    return HackyCIGANLoader(batch_size, ninstances, data_format, cid_to_ims)
