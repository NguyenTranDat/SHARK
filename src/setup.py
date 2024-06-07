from setuptools import setup, find_packages

setup(
    name='MintRec',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'numpy',
        'scikit-learn',
        'typing-extensions',
    ],
)
