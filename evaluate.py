import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Available: ", len(gpu_devices))
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

from utility.models import PointNet, PointNet2_SSG, PointNet2_MSG

tf.random.set_seed(1234)


def load_dataset(in_file, batch_size, n_points):
    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 500

    def _extract_fn(data_record):
        in_features = {'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
                       'label': tf.io.FixedLenFeature([1], tf.int64)}
        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):
        points = sample['points']
        label = sample['label']

        points = tf.reshape(points, (n_points, 3))
        points = tf.random.shuffle(points)
        return points, label

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def infer():

    if args.model == 'pointnet':
        model = PointNet(num_points=args.num_points)
        model.info()
    elif args.model == 'pointnet++_ssg':
        model = PointNet2_SSG(batch_size=args.batch_size)
        model.build(input_shape=(args.batch_size, args.num_points, 3))
        print(model.summary())
    elif args.model == 'pointnet++_msg':
        model = PointNet2_MSG(batch_size=args.batch_size)
        model.build(input_shape=(args.batch_size, args.num_points, 3))
        print(model.summary())

    test_ds = load_dataset('data/test.tfrecord', args.batch_size, args.num_points)

    if not args.logdir:
        args.logdir = args.model

    model.compile(metrics=["accuracy"])
    model.load_weights('./logs/{}/model/weights_{epoch:04d}.ckpt'.format(args.logdir, epoch=args.epoch)).expect_partial()
    model.evaluate(test_ds)


if __name__ == '__main__':

    inputs = argparse.ArgumentParser()
    inputs.add_argument("--model", type=str, required=True,
                        help="Can be: 'pointnet' / 'pointnet++_ssg' / 'pointnet+++msg'")
    inputs.add_argument("--batch_size", type=int, default=16)
    inputs.add_argument("-num_points", type=int, default=500,
                        help="Number of points sampled from each cluster's surface model - MUST BE SAME WITH PARSER")
    inputs.add_argument("--epoch", type=int, required=True, help="Epoch of the trained weights to use")
    inputs.add_argument("--logdir", type=str,
                        help="Directory of saved trained models in the 'logs' folder - Defaults to the model name")
    args = inputs.parse_args()

    infer()

