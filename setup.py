from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "opencv-python",
    "Polygon3",
    "pyclipper",
    "tensorflow-gpu",
]

setup(
    name="psenet",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="PSENet",
)
