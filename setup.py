import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name='tfne',
    version='0.21.0',
    scripts=['tfne_visualizer'],
    author='Paul Pauls',
    author_email='tfne@paulpauls.de',
    description='A modular Neuroevolution framework for Tensorflow models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/PaulPauls/Tensorflow-Neuroevolution",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow >= 2.0.0",
        "ray",
        "graphviz",
        "matplotlib",
        "PyQt5",
        "pydot",
    ],
    python_requires='>= 3.7',
)
