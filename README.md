# NavioOccupancy

A repo for trainig occupancy prediction networks. Currently supports [OccRWKV](https://github.com/jmwang0117/OccRWKV) 
and [LMSCNet](https://github.com/astra-vision/LMSCNet) models. Supports SemanticKitti and Navio(our custom) datasets.


## Preparation
### Prerequisites
Tested with
* Python 3.12
* Pytorch 2.8+cu12.6


### Setup
We advise to create a new virtual environment for installation. Some of the following is required for `project2d` repo

```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install spconv-cu126
pip install tensorboard
pip install dropblock
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
pip install opencv-python
pip install tqdm
pip install pyaml
pip install open3d
pip install pypcd4
pip install imageio
```

There is also a `requirements.txt` file that you can use.

#### project2d

This project uses project2d repo. Clone it inside the root of this project.

```
# inside NavioOccupancy/
git clone ssh://git@gitlab.sberautotech.ru:7999/sd/dl/occupancy/project2d.git
```

## Dataset
General expected format: Dataset is split into sequences, there are same modalities in each sequennce. Files in
each modality dir have the same name.

```
sequence_0
    modality_dir
        file_0.ext
    another_modality
        file_0.ext
```

## Config

Configs are situated in the [cfg/](cfg/) dir. They are based on the python dataclasses. [base.py](cfg/base.py)
describes the base settings that each experint ovverrides (if there is a need for that).

The main thing the config does is that it specifies where data is located. 

```python
path_solvers={
    M.VOXELS: path_solvers.SimplePathSolver(M.VOXELS.value, "voxels", "bin"),
    M.LIDAR: path_solvers.ReplacePathSolver(M.LIDAR.value, "voxels", "bin",
                                            lambda x: x.replace("voxels", "lidar").replace(".bin", ".pcd")),
    M.LABEL_2D: path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev_ms_3", "png"),
    M.LABEL_LIDAR: path_solvers.ReplacePathSolver(M.LABEL_LIDAR.value, "voxels", "bin",
        lambda x: x.replace("voxels", "label").replace(".bin", ".json")),
    M.POSE: path_solvers.ReplacePathSolver(M.POSE.value, "voxels", "bin",
                                            lambda x: x.replace("voxels", "meta").replace(".bin", ".json")),
}
```

* `SimplePathSolver`: Just performs an `os.listdir()`
* `ReplacePathSolver`: Performs an `os.listdir()` and applies a replace lambda after that. Used when there is a 
different nuber of files in modality dirs.
* `ScaledPathSolver`: Performs an `os.listdir()` and searches different scale suffixes in the filenames (like `1_4` or `1_2`)

## Training

Training is handeled by [src/train.py](src/train.py). The script has the following cli:
* `--cfg` Path to an experimnet config
* `--dset_root` (optional) Path to the dataset root you if wish to overrite the one in the config
* `--resume` (optional) Path to an artifacts folder of the previous experiment if you wish to continue the experiment

```
python3 src/train.py --cfg cfg/kitti/rwkv_2d.py --resume outputs/ModelType.OCCRWKV_2D_DatasetType.SEMANTIC_KITTI_0115_102754
```

## Validation

Validation is handeled by [src/validate.py](src/validate.py). The script has the following cli:
* `--cfg` Path to an experimnet config
* `--weights` Path to a .pth model you wish to validate
* `--dset_root` (optional) Path to the dataset root you if wish to overrite the one in the config
* `--out_path` (optional) Path to a dir where predictions will be saved
* `--swap_c` (optional) Whether to swap 'road' and 'static' class channels. Required for compatability with older 
chekpoints

```
python3 src/validate.py --cfg cfg/navio_markup/rwkv_2d.py --weights outputs/OccRWKV2D_DatasetType.SEMANTIC_KITTI_0114_172810/ckpt/best-metric/weights_epoch_068.pth --swap_c
```

## Testing

Testing is handeled by [src/test.py](src/test.py). The scrupt has the same cli as the validation script, plus 2 more params:
* `--3d` (optional) Whether the input model is 3d
* `--title` (optinal) Name of a title for every image

