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
import pyrender
import argparse
import numpy as np
import matplotlib.pyplot as plt

from acronym_tools import Scene, load_mesh, create_gripper_marker


def make_parser():
    parser = argparse.ArgumentParser(
        description="Render observations of a randomly generated scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--objects", nargs="+", help="HDF5 or JSON Object file(s).")
    parser.add_argument(
        "--support",
        required=True,
        type=str,
        help="HDF5 or JSON File for support object.",
    )
    parser.add_argument(
        "--support_scale", default=0.025, help="Scale factor of support mesh."
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    parser.add_argument(
        "--show_scene",
        action="store_true",
        help="Show the scene and camera pose from which observations are rendered.",
    )
    return parser


class PyrenderScene(Scene):
    def as_pyrender_scene(self):
        """Return pyrender scene representation.

        Returns:
            pyrender.Scene: Representation of the scene
        """
        pyrender_scene = pyrender.Scene()
        for obj_id, obj_mesh in self._objects.items():
            mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
            pyrender_scene.add(mesh, name=obj_id, pose=self._poses[obj_id])
        return pyrender_scene


class SceneRenderer:
    def __init__(
        self,
        pyrender_scene,
        fov=np.pi / 6.0,
        width=400,
        height=400,
        aspect_ratio=1.0,
        z_near=0.001,
    ):
        """Create an image renderer for a scene.

        Args:
            pyrender_scene (pyrender.Scene): Scene description including object meshes and their poses.
            fov (float, optional): Field of view of camera. Defaults to np.pi/6.
            width (int, optional): Width of camera sensor (in pixels). Defaults to 400.
            height (int, optional): Height of camera sensor (in pixels). Defaults to 400.
            aspect_ratio (float, optional): Aspect ratio of camera sensor. Defaults to 1.0.
            z_near (float, optional): Near plane closer to which nothing is rendered. Defaults to 0.001.
        """
        self._fov = fov
        self._width = width
        self._height = height
        self._z_near = z_near
        self._scene = pyrender_scene

        self._camera = pyrender.PerspectiveCamera(
            yfov=fov, aspectRatio=aspect_ratio, znear=z_near
        )

    def get_trimesh_camera(self):
        """Get a trimesh object representing the camera intrinsics.

        Returns:
            trimesh.scene.cameras.Camera: Intrinsic parameters of the camera model
        """
        return trimesh.scene.cameras.Camera(
            fov=(np.rad2deg(self._fov), np.rad2deg(self._fov)),
            resolution=(self._height, self._width),
            z_near=self._z_near,
        )

    def _to_pointcloud(self, depth):
        """Convert depth image to pointcloud given camera intrinsics.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Point cloud.
        """
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def render(self, camera_pose, target_id="", render_pc=True):
        """Render RGB/depth image, point cloud, and segmentation mask of the scene.

        Args:
            camera_pose (np.ndarray): Homogenous 4x4 matrix describing the pose of the camera in scene coordinates.
            target_id (str, optional): Object ID which is used to create the segmentation mask. Defaults to ''.
            render_pc (bool, optional): If true, point cloud is also returned. Defaults to True.

        Returns:
            np.ndarray: Color image.
            np.ndarray: Depth image.
            np.ndarray: Point cloud.
            np.ndarray: Segmentation mask.
        """
        # Keep local to free OpenGl resources after use
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self._width, viewport_height=self._height
        )

        # add camera and light to scene
        scene = self._scene.as_pyrender_scene()
        scene.add(self._camera, pose=camera_pose, name="camera")
        light = pyrender.SpotLight(
            color=np.ones(4),
            intensity=3.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=camera_pose, name="light")

        # render the full scene
        color, depth = renderer.render(scene)

        segmentation = np.zeros(depth.shape, dtype=np.uint8)

        # hide all objects
        for node in scene.mesh_nodes:
            node.mesh.is_visible = False

        # Render only target object and add to segmentation mask
        for node in scene.mesh_nodes:
            if node.name == target_id:
                node.mesh.is_visible = True
                _, object_depth = renderer.render(scene)
                mask = np.logical_and(
                    (np.abs(object_depth - depth) < 1e-6), np.abs(depth) > 0
                )
                segmentation[mask] = 1

        for node in scene.mesh_nodes:
            node.mesh.is_visible = True

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, segmentation


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)

    # load object meshes and generate a random scene
    object_meshes = [load_mesh(o, mesh_root_dir=args.mesh_root) for o in args.objects]
    support_mesh = load_mesh(
        args.support, mesh_root_dir=args.mesh_root, scale=args.support_scale
    )
    scene = PyrenderScene.random_arrangement(object_meshes, support_mesh)

    target_obj = "obj0"

    # choose camera intrinsics and extrinsics
    renderer = SceneRenderer(scene)
    trimesh_camera = renderer.get_trimesh_camera()
    camera_pose = trimesh_camera.look_at(
        points=[scene.get_transform(target_obj, frame="com")[:3, 3]],
        rotation=trimesh.transformations.euler_matrix(
            np.random.uniform(low=np.pi / 4, high=np.pi / 3),
            0,
            np.random.uniform(low=-np.pi, high=np.pi),
        ),
        distance=np.random.uniform(low=0.7, high=0.9),
    )

    if args.show_scene:
        # show scene, including a marker representing the camera
        trimesh_scene = scene.colorize({target_obj: [255, 0, 0]}).as_trimesh_scene()
        trimesh_scene.add_geometry(
            trimesh.creation.camera_marker(trimesh_camera),
            node_name="camera",
            transform=camera_pose.dot(
                trimesh.transformations.euler_matrix(np.pi, 0, 0)
            ),
        )
        trimesh_scene.show()

    # render observations
    color, depth, pc, segmentation = renderer.render(
        camera_pose=camera_pose, target_id=target_obj
    )

    # plot everything except point cloud
    f, axarr = plt.subplots(1, 3)
    im = axarr[0].imshow(color)
    f.colorbar(im, ax=axarr[0])
    im = axarr[1].imshow(depth)
    f.colorbar(im, ax=axarr[1])
    im = axarr[2].imshow(segmentation)
    f.colorbar(im, ax=axarr[2])
    plt.show()


if __name__ == "__main__":
    main()
