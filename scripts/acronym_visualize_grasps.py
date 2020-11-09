#!/usr/bin/env python3
"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import json
import trimesh
import argparse
import numpy as np

from acronym_tools import load_mesh, load_grasps, create_gripper_marker


def make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize grasps from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="HDF5 or JSON Grasp file(s).")
    parser.add_argument(
        "--num_grasps", type=int, default=20, help="Number of grasps to show."
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    return parser


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)

    for f in args.input:
        # load object mesh
        obj_mesh = load_mesh(f, mesh_root_dir=args.mesh_root)

        # get transformations and quality of all simulated grasps
        T, success = load_grasps(f)

        # create visual markers for grasps
        successful_grasps = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], args.num_grasps)]
        ]
        failed_grasps = [
            create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 0)[0], args.num_grasps)]
        ]

        trimesh.Scene([obj_mesh] + successful_grasps + failed_grasps).show()


if __name__ == "__main__":
    main()
