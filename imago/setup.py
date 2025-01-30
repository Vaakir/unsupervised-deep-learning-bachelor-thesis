from setuptools import setup, find_packages

setup(
    name="imago",  # Name of your module
    version="0.1.0",  # Initial version
    description="A Python module for reading and operating on MRI scans",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find submodules
    install_requires=[
        "numpy>=1.21.0",  # Replace with dependencies for your module
        "matplotlib>=3.4.0",
        "nibabel>=5.1.0",  # Common library for working with MRI formats
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],  # Development dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
