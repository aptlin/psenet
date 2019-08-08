from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["opencv-python" "Polygon3" "pyclipper" "tensorflow"]
CUSTOM_PACKAGES = [
    "git+git://github.com/qubvel/segmentation_models"
    + "@feature/tf.keras#egg=segmentation-models"
]

setup(
    name="psenet",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=CUSTOM_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="PSENet",
)
