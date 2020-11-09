[ACRONYM](https://sites.google.com/nvidia.com/graspdataset) is a dataset of 17.7M simulated parallel-jaw grasps of 8872 objects. It was generated using [NVIDIA FleX](https://developer.nvidia.com/flex).

This repository contains a sample of the grasping dataset and tools to visualize grasps, generate random scenes, and render observations.

For using the full ACRONYM dataset, see instructions [below](#using-the-full-acronym-dataset).

# License
The source code is released under [MIT License](LICENSE). The dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

# Requirements
* Python3
* `python -m pip install -r requirements.txt`

# Installation
* `python -m pip install -e .`

# Use Cases

### Visualize Grasps
```
usage: acronym_visualize_grasps.py [-h] [--num_grasps NUM_GRASPS] input [input ...]

Visualize grasps from the dataset.

positional arguments:
  input                 HDF5 or JSON Grasp file(s).

optional arguments:
  -h, --help            show this help message and exit
  --num_grasps NUM_GRASPS
                        Number of grasps to show. (default: 20)
  --mesh_root MESH_ROOT
                        Directory used for loading meshes. (default: .)
```

#### Examples
The following command shows 40 grasps for a mug from the dataset. Grasp markers are colored green/red based on whether the simulation result was a success/failure:

`acronym_visualize_grasps.py --mesh_root data/examples/ data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5`

### Generate Random Scenes and Visualize Grasps
```
usage: generate_scene.py [-h] [--objects OBJECTS [OBJECTS ...]] --support
                         SUPPORT [--support_scale SUPPORT_SCALE]
                         [--show_grasps]
                         [--num_grasps_per_object NUM_GRASPS_PER_OBJECT]

Generate a random scene arrangement and filtering grasps that are in
collision.

optional arguments:
  -h, --help            show this help message and exit
  --objects OBJECTS [OBJECTS ...]
                        HDF5 or JSON Object file(s). (default: None)
  --support SUPPORT     HDF5 or JSON File for support object. (default: None)
  --support_scale SUPPORT_SCALE
                        Scale factor of support mesh. (default: 0.025)
  --mesh_root MESH_ROOT
                        Directory used for loading meshes. (default: .)
  --show_grasps         Show all grasps that are not in collision. (default:
                        False)
  --num_grasps_per_object NUM_GRASPS_PER_OBJECT
                        Maximum number of grasps to show per object. (default:
                        20)
```

#### Examples
This will show a randomly generated scene with a table as a support mesh and four mugs placed on top of it:

`acronym_generate_scene.py --mesh_root data/examples/ --objects data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 --support data/examples/grasps/Table_99cf659ae2fe4b87b72437fd995483b_0.009700376721042367.h5`

Same as above but also showing green grasp markers (maximum: 20 per object) for successful grasps (filtering those that are in collision):

`acronym_generate_scene.py --mesh_root data/examples/ --objects data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 --support data/examples/grasps/Table_99cf659ae2fe4b87b72437fd995483b_0.009700376721042367.h5 --show_grasps`

### Render and Visualize Observations
```
usage: render_observations.py [-h] [--objects OBJECTS [OBJECTS ...]] --support
                              SUPPORT [--support_scale SUPPORT_SCALE]
                              [--show_scene]

Render observations of a randomly generated scene.

optional arguments:
  -h, --help            show this help message and exit
  --objects OBJECTS [OBJECTS ...]
                        HDF5 or JSON Object file(s). (default: None)
  --support SUPPORT     HDF5 or JSON File for support object. (default: None)
  --support_scale SUPPORT_SCALE
                        Scale factor of support mesh. (default: 0.025)
  --mesh_root MESH_ROOT
                        Directory used for loading meshes. (default: .)
  --show_scene          Show the scene and camera pose from which observations
                        are rendered. (default: False)
```

#### Examples
This will show RGB image, depth, image and segmentation mask rendered from a random viewpoint):

`acronym_render_observations.py --mesh_root data/examples/ --objects data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 --support data/examples/grasps/Table_99cf659ae2fe4b87b72437fd995483b_0.009700376721042367.h5`

Same as above but also visualizes the scene and camera position in 3D:

`acroacronym_render_observations.py --mesh_root data/examples/ --objects data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 data/examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5 --support data/examples/grasps/Table_99cf659ae2fe4b87b72437fd995483b_0.009700376721042367.h5 --show_scene`


# Using the full ACRONYM dataset

1. Download the full dataset (1.6GB): [acronym.tar.gz](https://drive.google.com/file/d/1zcPARTCQx2oeiKk7a-wdN_CN-RUVX56c/view?usp=sharing)
2. Download the ShapeNetSem meshes from https://www.shapenet.org/
3. Create watertight versions of the downloaded meshes:
   1. Clone and build: https://github.com/hjwdzh/Manifold
   2. Create a watertight mesh version assuming the object path is model.obj: `manifold model.obj temp.watertight.obj -s`
   3. Simplify it: `simplify -i temp.watertight.obj -o model.obj -m -r 0.02`

For more details about the structure of the ACRONYM dataset see: https://sites.google.com/nvidia.com/graspdataset


# Citation
If you use the dataset please cite:
```
@inproceedings{acronym2020,
    title     = {{ACRONYM}: A Large-Scale Grasp Dataset Based on Simulation},
    author    = {Eppner, Clemens and Mousavian, Arsalan and Fox, Dieter},
    year      = {2020},
    booktitle = {Under Review at ICRA 2021}
}
```
