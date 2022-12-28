# Rockfall Detection
This is a 3D rockfall detection project based on binary point cloud classification with [PointNet](https://arxiv.org/abs/1612.00593) and [PointNet++](https://arxiv.org/abs/1612.00593) models. The project aims at separating between rockfalls an other deformation clusters detected within rock slope change detection programs. Detailed applications of the models and analysis results on real rockfall monitoring cases are demonstrated in the associated research [paper](https://www.sciencedirect.com/science/article/pii/S0013795222003210).
The repository includes components of the TensorFlow 2 layers provided [here](https://github.com/dgriffiths3/pointnet2-tensorflow2) and the TensorFlow operations provided [here](https://github.com/charlesq34/pointnet2/tree/master/tf_ops).

# <sub>Installation
The implementations in the associated [paper](https://www.sciencedirect.com/science/article/pii/S0013795222003210) were done in a Ubuntu 18.04 OS with the following setup:
  - python 3.6
  - tensorflow-gpu 2.2.0
  - cuda 10.1

The following python modules should also be included:
  - open3d 0.13.0
  - trimesh

To compile the TensorFlow operations make sure the <code>CUDA_ROOT</code> path in <code>tf_ops/compile_ops.sh</code> points to the correct CUDA installation folder in your machine. Then compile the operations by executing the following commands in the project's directory:

<pre><code>chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
</code></pre>

# <sub>Data preparation
The training data for the models should represent individual deformation clusters resulted from point cloud based change detection after de-noising, clustering, and meshing. Details on the data generation workflow are provided [here](https://www.mdpi.com/2220-9964/10/3/157). When both <code>rockfall</code>  and <code>non_rockfall</code> data are available in <code>.off</code> format and split into <code>train</code>, <code>dev</code>, <code>test</code> sets, copy them in the respective folders in the <code>data</code> folder.

Now, you are ready to parse the date into TensorFlow records by executing:
<pre><code>python parser.py --num_points ####
</code></pre>
where <code>####</code> is the number of points to be sampled from the surface of each deformation cluster (default=500).

# <sub>Training
To train a model with the parsed data, simply run the <code>train.py</code> script with the following arguments:
  - **model:** string data type that can be either <code>pointnet</code>, <code>pointnet++_ssg</code>, or <code>pointnet++_msg</code>
  - **epochs:** integer defining the number of training epochs
  - **batch_size:** intereger defining the size of each batch of data (default=16)
  - **num_points:** integer defining the number of points sampled from each cluster's surface model and MUST be same with parser (default=500)
  - **bn:** boolean defining whether batch normalization is applied (default=True)
  - **momentum:** float defining the momentum in batch normalization (default=0.99)
  - **dropout:** float defining the keep probabilify for dropout layers (default=0.3)
  - **lr:** float defining the learning rate (default=0.001)
  - **logdir:** directory to save the trained models in the folder called <code>logs</code> (default=the selected model name)

Here is an example for training a PointNet++ model with Multi-Scale Grouping (MSG) for 100 epochs and the default settings:
  <pre><code>python train.py --model pointnet++_msg --epochs 100</code></pre>

