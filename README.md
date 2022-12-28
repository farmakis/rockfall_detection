# rockfall_detection
This is a 3D rockfall detection project based on point cloud classification with [PointNet](https://arxiv.org/abs/1612.00593) and [PointNet++](https://arxiv.org/abs/1612.00593) models. The project aims at separating between rockfalls an other deformation clusters detected within rock slope change detection programs. Detailed applications of the models and analysis results on real rockfall monitoring cases are demonstrated in the associated reseach [paper](https://www.sciencedirect.com/science/article/pii/S0013795222003210).
The repository includes components of the Tensorflow 2 layers provided [here](https://github.com/dgriffiths3/pointnet2-tensorflow2) and the TensorFlow operations provided [here](https://github.com/charlesq34/pointnet2/tree/master/tf_ops).

# Installation
The implementations in the associated [paper](https://www.sciencedirect.com/science/article/pii/S0013795222003210) were done in a Ubuntu 18.04 OS with the following setup:
  - python 3.6
  - tensorflow-gpu 2.2.0
  - cuda 10.1

The following python modules should also be included:
  - open3d 0.13.0
  - trimesh

To compile the TensorFlow operations make sure the '''CUDA_ROOT''' path in '''tf_ops/compile_ops.sh''' points to the correct DUCA installation folder in your machine, and the compile by executing the following commands:
'''
chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
'''

# Usage
