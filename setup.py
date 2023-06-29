from setuptools import setup, find_packages

"""
    This is only here for legacy purpose and should not be used by your system see https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html:

    If compatibility with legacy builds or versions of tools that don’t support certain packaging standards 
    (e.g. PEP 517 or PEP 660), a simple setup.py script can be added to your project (while keeping the configuration in pyproject.toml):
"""

setup(
    name="dhiret",
    version="0.5",
    author="David Tschirschwitz",
    author_email="david.tschirschwitz@uni-weimar.de",
    description="Package that allows the training, evaluation and usage of a retrieval pipeline developed for historical image retrieval.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pytest",
        "annoy",
        "open_clip_torch",
        "matplotlib",
        "tqdm",
        "pandas",
        "learn2learn",
        "timm",
        "tensorboard",
        "jupyterlab",
        "ipywidgets"
    ],
)