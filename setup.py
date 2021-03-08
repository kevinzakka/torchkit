from setuptools import find_packages, setup

setup(
    name="torchkit",
    version="0.0.1",
    author="Kevin",
    description="PyTorch Utilities for Research",
    python_requires=">=3.6",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=[
        "torch>=1.3",
        "torchvision>=0.4",
        "tensorboard",
        "prettytable",
        "opencv-python",
    ],
)
