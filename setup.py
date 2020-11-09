import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acronym_tools",
    version="0.0.1",
    author="Clemens Eppner, Arsalan Mousavian",
    author_email="ceppner@nvidia.com",
    description="A few scripts to work with the grasp dataset ACRONYM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://sites.google.com/nvidia.com/graspdataset/",
    packages=['acronym_tools'],
    scripts=[
        'scripts/acronym_generate_scene.py',
        'scripts/acronym_render_observations.py',
        'scripts/acronym_visualize_grasps.py',
    ],
    package_data={'acronym_tools': ['data/franka_gripper_collision_mesh.stl']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
