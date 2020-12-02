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

import os
import json
import h5py
import trimesh
import trimesh.path
import trimesh.transformations as tra
import numpy as np


class Scene(object):
    """Represents a scene, which is a collection of objects and their poses."""

    def __init__(self):
        """Create a scene object."""
        self._objects = {}
        self._poses = {}
        self._support_objects = []

        self.collision_manager = trimesh.collision.CollisionManager()

    def add_object(self, obj_id, obj_mesh, pose, support=False):
        """Add a named object mesh to the scene.

        Args:
            obj_id (str): Name of the object.
            obj_mesh (trimesh.Trimesh): Mesh of the object to be added.
            pose (np.ndarray): Homogenous 4x4 matrix describing the objects pose in scene coordinates.
            support (bool, optional): Indicates whether this object has support surfaces for other objects. Defaults to False.
        """
        self._objects[obj_id] = obj_mesh
        self._poses[obj_id] = pose
        if support:
            self._support_objects.append(obj_mesh)

        self.collision_manager.add_object(name=obj_id, mesh=obj_mesh, transform=pose)

    def _get_support_polygons(
        self, min_area=0.01, gravity=np.array([0, 0, -1.0]), erosion_distance=0.02
    ):
        """Extract support facets by comparing normals with gravity vector and checking area.

        Args:
            min_area (float, optional): Minimum area of support facets [m^2]. Defaults to 0.01.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.

        Returns:
            list[trimesh.path.polygons.Polygon]: list of support polygons.
            list[np.ndarray]: list of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
        """
        assert np.isclose(np.linalg.norm(gravity), 1.0)

        support_polygons = []
        support_polygons_T = []

        # Add support plane if it is set (although not infinite)
        support_meshes = self._support_objects

        for obj_mesh in support_meshes:
            # get all facets that are aligned with -gravity and bigger than min_area
            support_facet_indices = np.argsort(obj_mesh.facets_area)
            support_facet_indices = [
                idx
                for idx in support_facet_indices
                if np.isclose(obj_mesh.facets_normal[idx].dot(-gravity), 1.0, atol=0.5)
                and obj_mesh.facets_area[idx] > min_area
            ]

            for inds in support_facet_indices:
                index = inds
                normal = obj_mesh.facets_normal[index]
                origin = obj_mesh.facets_origin[index]

                T = trimesh.geometry.plane_transform(origin, normal)
                vertices = trimesh.transform_points(obj_mesh.vertices, T)[:, :2]

                # find boundary edges for the facet
                edges = obj_mesh.edges_sorted.reshape((-1, 6))[
                    obj_mesh.facets[index]
                ].reshape((-1, 2))
                group = trimesh.grouping.group_rows(edges, require_count=1)

                # run the polygon conversion
                polygon = trimesh.path.polygons.edges_to_polygons(
                    edges=edges[group], vertices=vertices
                )

                assert len(polygon) == 1

                # erode to avoid object on edges
                polygon[0] = polygon[0].buffer(-erosion_distance)

                if not polygon[0].is_empty and polygon[0].area > min_area:
                    support_polygons.append(polygon[0])
                    support_polygons_T.append(trimesh.transformations.inverse_matrix(T))

        return support_polygons, support_polygons_T

    def _get_random_stable_pose(self, stable_poses, stable_poses_probs):
        """Return a stable pose according to their likelihood.

        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.

        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """
        index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])

    def find_object_placement(
        self, obj_mesh, max_iter, distance_above_support, gaussian=None
    ):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.
            distance_above_support (float): Distance the object mesh will be placed above the support surface.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.

        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=1
        )
        # stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=1)

        # Sample support index
        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            if gaussian:
                while True:
                    p = Point(
                        np.random.normal(
                            loc=np.array(gaussian[:2])
                            + support_polys[support_index].centroid,
                            scale=gaussian[2:],
                        )
                    )
                    if p.within(support_polys[support_index]):
                        pts = [p.x, p.y]
                        break
            else:
                pts = trimesh.path.polygons.sample(
                    support_polys[support_index], count=1
                )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, distance_above_support)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )

            pose = self._get_random_stable_pose(stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T, pose), tra.translation_matrix(-obj_mesh.center_mass)
            )
            # placement_T = np.dot(placement_T, pose)

            # Check collisions
            colliding = self.in_collision_with(
                obj_mesh, placement_T, min_distance=distance_above_support
            )

            iter += 1

        return not colliding, placement_T if not colliding else None

    def in_collision_with(self, mesh, transform, min_distance=0.0, epsilon=1.0 / 1e3):
        """Check whether the scene is in collision with mesh. Optional: Define a minimum distance.

        Args:
            mesh (trimesh.Trimesh): Object mesh to test with scene.
            transform (np.ndarray): Pose of the object mesh as a 4x4 homogenous matrix.
            min_distance (float, optional): Minimum distance that is considered in collision. Defaults to 0.0.
            epsilon (float, optional): Epsilon for minimum distance check. Defaults to 1.0/1e3.

        Returns:
            bool: Whether the object mesh is colliding with anything in the scene.
        """
        colliding = self.collision_manager.in_collision_single(
            mesh=mesh, transform=transform
        )
        if not colliding and min_distance > 0.0:
            distance = self.collision_manager.min_distance_single(
                mesh=mesh, transform=transform
            )
            if distance < min_distance - epsilon:
                colliding = True
        return colliding

    def place_object(
        self, obj_id, obj_mesh, max_iter=100, distance_above_support=0.0, gaussian=None
    ):
        """Add object and place it in a non-colliding stable pose on top of any support surface.

        Args:
            obj_id (str): Name of the object to place.
            obj_mesh (trimesh.Trimesh): Mesh of the object to be placed.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Returns:
            [type]: [description]
        """
        success, placement_T = self.find_object_placement(
            obj_mesh,
            max_iter,
            distance_above_support=distance_above_support,
            gaussian=gaussian,
        )

        if success:
            self.add_object(obj_id, obj_mesh, placement_T)
        else:
            print("Couldn't place object", obj_id, "!")

        return success

    def get_transform(self, obj_id, frame="com"):
        """Get object transformation in scene coordinates.

        Args:
            obj_id (str): Name of the object.
            frame (str, optional): Object reference frame to use. Either 'com' (center of mass) or 'mesh' (origin of mesh file). Defaults to 'com'.

        Raises:
            ValueError: If frame is not 'com' or 'mesh'.

        Returns:
            np.ndarray: Homogeneous 4x4 matrix.
        """
        if frame == "com":
            return np.dot(
                self._poses[obj_id],
                tra.translation_matrix(self._objects[obj_id].center_mass),
            )
        elif frame == "mesh":
            return self._poses[obj_id]
        raise ValueError("Unknown argument:", frame)

    def colorize(self, specific_objects={}, brightness=1.0):
        """Colorize meshes.

        Args:
            specific_objects (dict, optional): A dictionary of object id's to be colored. Defaults to {}.
            brightness (float, optional): Brightness of colors. Defaults to 1.0.
        """
        if not specific_objects:
            for obj_id, obj_mesh in self._objects.items():
                obj_mesh.visual.face_colors[:, :3] = (
                    trimesh.visual.random_color() * brightness
                )[:3]
        else:
            for object_id, color in specific_objects.items():
                self._objects[object_id].visual.face_colors[:, :3] = color
        return self

    def as_trimesh_scene(self):
        """Return trimesh scene representation.

        Returns:
            trimesh.Scene: Scene representation.
        """
        trimesh_scene = trimesh.scene.Scene()
        for obj_id, obj_mesh in self._objects.items():
            trimesh_scene.add_geometry(
                obj_mesh,
                node_name=obj_id,
                geom_name=obj_id,
                transform=self._poses[obj_id],
            )
        return trimesh_scene

    @classmethod
    def random_arrangement(
        cls, object_meshes, support_mesh, distance_above_support=0.002, gaussian=None
    ):
        """Generate a random scene by arranging all object meshes on any support surface of a provided support mesh.

        Args:
            object_meshes (list[trimesh.Trimesh]): List of meshes of all objects to be placed on top of the support mesh.
            support_mesh (trimesh.Trimesh): Mesh of the support object.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            gaussian (list[float], optional): Normal distribution for position in plane (mean_x, mean_y, std_x, std_y). Defaults to None.

        Returns:
            Scene: Scene representation.
        """
        s = cls()
        s.add_object("support_object", support_mesh, pose=np.eye(4), support=True)

        for i, obj_mesh in enumerate(object_meshes):
            s.place_object(
                "obj{}".format(i),
                obj_mesh,
                distance_above_support=distance_above_support,
                gaussian=gaussian,
            )

        return s


def load_mesh(filename, mesh_root_dir, scale=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh


def load_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)
    return T, success


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp
