from setuptools import find_packages, setup

setup(
    name="mnist_neptune",
    packages=find_packages(where="mnist_neptune"),
    package_dir={"": "mnist_neptune"},
)
