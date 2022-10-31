from setuptools import setup

setup(
    name = "Visualization",
    version = "0.0.1",
    author = "Yi Huang",
    author_email = "yhuang2@bnl.gov",
    description = ("Visualzation"),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "TBD",
    packages=['visualization'],
    long_description="",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "tqdm",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
