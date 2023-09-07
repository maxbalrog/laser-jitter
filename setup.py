from setuptools import setup, find_packages

from laser_jitter import __version__

setup(
    name='laser_jitter',
    version=__version__,

    url='https://github.com/maxbalrog/laser_jitter',
    author='Maksim Valialshchikov',
    author_email='maksim.valialshchikov@uni-jena.de',

    packages=find_packages(exclude=['tests', 'tests.*']),
)
