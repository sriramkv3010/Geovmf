"""
setup.py
========
Minimal setuptools configuration so the project can be installed as a
package with ``pip install -e .`` for import convenience in notebooks.
"""

from setuptools import setup, find_packages

setup(
    name        = "agg_neuropalsy_gaze",
    version     = "1.0.0",
    description = (
        "Adaptive Geometric Gaze Estimation under "
        "Neuropathological Conditions (AGG + vMF)"
    ),
    packages    = find_packages(
        exclude=["results", "notebooks", "scripts"]
    ),
    python_requires = ">=3.10",
    install_requires = [
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "scikit-learn>=1.4",
        "kagglehub>=0.2",
        "tqdm>=4.66",
    ],
    extras_require = {
        "dev": ["pytest", "black", "isort", "flake8"],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
