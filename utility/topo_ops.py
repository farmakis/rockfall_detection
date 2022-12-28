import os
import sys
import glob
import trimesh
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool1D, Layer, BatchNormalization

from .cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)
tf.random.set_seed(1234)


def parser(data_dir, num_points=1024):

	train_points = []
	train_labels = []
	dev_points = []
	dev_labels = []
	test_points = []
	test_labels = []
	class_map = []
	folders = glob.glob(os.path.join(data_dir, "*/"))

	for i, folder in enumerate(reversed(folders)):
		print("processing class: {}".format(folder.split("/")[-2]))
		# # store folder map with ID so it can be retrieved later
		# class_map[i] = folder.split("/")[-2]
		# gather all files
		train_files = glob.glob(os.path.join(folder, "train/*"))
		dev_files = glob.glob(os.path.join(folder, "dev/*"))
		test_files = glob.glob(os.path.join(folder, "test/*"))

		for f in train_files:
			train_points.append(trimesh.load(f).sample(num_points))
			train_labels.append(i)

		for f in dev_files:
			dev_points.append(trimesh.load(f).sample(num_points))
			dev_labels.append(i)

		for f in test_files:
			test_points.append(trimesh.load(f).sample(num_points))
			test_labels.append(i)

	return (np.array(train_points), np.array(dev_points), np.array(test_points),
			np.array(train_labels), np.array(dev_labels), np.array(test_labels))


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):

	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	if knn:
		_,idx = knn_point(nsample, xyz, new_xyz)
	else:
		idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		if use_xyz:
			new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
		else:
			new_points = grouped_points
	else:
		new_points = grouped_xyz

	return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):

	batch_size = xyz.get_shape()[0]
	nsample = xyz.get_shape()[1]

	new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)

	idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
	grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
	if points is not None:
		if use_xyz:
			new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
		else:
			new_points = points
		new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points, idx, grouped_xyz


