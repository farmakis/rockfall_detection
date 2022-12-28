import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

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


def train():

    if args.model == 'pointnet':
        model = PointNet(num_points=args.num_points, bn=args.bn, momentum=args.momentum, dropout=args.dropout)
        model.info()
    elif args.model == 'pointnet++_ssg':
        model = PointNet2_SSG(batch_size=args.batch_size, bn=args.bn, momentum=args.momentum, dropout=args.dropout)
        model.build(input_shape=(args.batch_size, args.num_points, 3))
        print(model.summary())
    elif args.model == 'pointnet++_msg':
        model = PointNet2_MSG(batch_size=args.batch_size, bn=args.bn, momentum=args.momentum, dropout=args.dropout)
        model.build(input_shape=(args.batch_size, args.num_points, 3))
        print(model.summary())

    train_ds = load_dataset('data/train.tfrecord', args.batch_size, args.num_points)
    dev_ds = load_dataset('data/dev.tfrecord', args.batch_size, args.num_points)
    test_ds = load_dataset('data/test.tfrecord', args.batch_size, args.num_points)

    if not args.logdir:
        args.logdir = args.model

    callbacks = [
        keras.callbacks.TensorBoard(
            './logs/{}'.format(args.logdir), update_freq='epoch', write_images=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join('./logs/{}/model'.format(args.logdir), 'weights_{epoch:04d}.ckpt'),
            monitor='val_loss', verbose=0, save_freq='epoch')
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    model.fit(
        train_ds,
        validation_data=dev_ds,
        callbacks=callbacks,
        epochs=args.epochs,
        verbose=1
    )


if __name__ == '__main__':

    inputs = argparse.ArgumentParser()
    inputs.add_argument("--model", type=str, required=True,
                        help="Can be: 'pointnet' / 'pointnet++_ssg' / 'pointnet+++msg'")
    inputs.add_argument("--batch_size", type=int, default=16)
    inputs.add_argument("-num_points", type=int, default=500,
                        help="Number of points sampled from each cluster's surface model - MUST BE SAME WITH PARSER")
    inputs.add_argument("--bn", type=bool, default=True, help="Batch normalization")
    inputs.add_argument("--momentum", type=float, default=0.99, help="Momentum in batch normalization")
    inputs.add_argument("--dropout", type=float, default=0.3, help="Keep probability for dropout layers")
    inputs.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    inputs.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    inputs.add_argument("--logdir", type=str,
                        help="Directory to save the trained models in the 'logs' folder - Defaults to the model name")
    args = inputs.parse_args()

    train()





