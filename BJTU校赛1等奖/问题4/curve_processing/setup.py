from setuptools import setup, find_packages

setup(
    name='curve_processing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tqdm'
    ],
    author=' ',
    author_email=' ',
    description='A tool for processing space curves',
    license='MIT',
    keywords='space curve processing',
    url=' '
)