import os
import glob
import trimesh
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf


def parser(num_points):
    folders = glob.glob(os.path.join("./data", "*/"))

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return tf.train.Feature(float_list=tf.train.FloatList(value=np.float32(ndarray).flatten().tolist()))

    def _int_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def pcloud_example(points, label):
        feature = {'points': _dtype_feature(points),
                   'label': _int_feature(label)}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    with tf.io.TFRecordWriter('data/train.tfrecord') as writer:
        print('::: Parsing train dataset')
        for i, folder in enumerate(reversed(folders)):
            print("processing class: {}".format(folder.split("/")[-2]))
            train_files = glob.glob(os.path.join(folder, "train/*"))
            for f in train_files:
                tf_example = pcloud_example(trimesh.load(f).sample(num_points), i)
                writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('data/dev.tfrecord') as writer:
        print('::: Parsing dev dataset')
        for i, folder in enumerate(reversed(folders)):
            print("processing class: {}".format(folder.split("/")[-2]))
            dev_files = glob.glob(os.path.join(folder, "dev/*"))
            for f in dev_files:
                tf_example = pcloud_example(trimesh.load(f).sample(num_points), i)
                writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('data/test.tfrecord') as writer:
        print('::: Parsing test dataset')
        for i, folder in enumerate(reversed(folders)):
            print("processing class: {}".format(folder.split("/")[-2]))
            test_files = glob.glob(os.path.join(folder, "test/*"))
            for f in test_files:
                tf_example = pcloud_example(trimesh.load(f).sample(num_points), i)
                writer.write(tf_example.SerializeToString())


if __name__ == '__main__':

    inputs = argparse.ArgumentParser()
    inputs.add_argument("-num_points", type=int, default=500,
                        help="Number of points sampled from each cluster's surface model")
    args = inputs.parse_args()

    parser(args.num_points)



















