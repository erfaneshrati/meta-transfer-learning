"""
Module configuration.
"""

from setuptools import setup

setup(
    name='supervised-mtl',
    version='0.0.1',
    description='Meta-transfer learning over Reptile and MAML',
    url='https://github.com/erfaneshrati/supervised-mtl',
    author='Amir Erfan Eshratifar',
    author_email='erfaneshrati@gmail.com',
    license='MIT',
    keywords='ai machine learning',
    packages=['meta-learning'],
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'Pillow>=4.0.0,<5.0.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
