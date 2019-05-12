#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 0):
    print('ERROR: UliEngineering currently requires at least Python 3.0 to run.')
    sys.exit(1)

setup(name='UliEngineering',
      version='0.2.8',
      description='Computational tools for electronics engineering',
      author='Uli Köhler',
      author_email='ukoehler@techoverflow.net',
      url='https://techoverflow.net/',
      license='Apache License v2.0',
      packages=find_packages(exclude=['tests*']),
      include_package_data=True,
      install_requires=['numpy (>= 1.5)', 'toolz (>= 0.5)'],
      extras_require= {
        'SciPy functionality': ['scipy (>= 0.5)']
      },
      test_suite='nose.collector',
      tests_require=['nose', 'coverage', 'mock', 'rednose', 'parameterized'],
      setup_requires=['nose>=1.0'],
      platforms="any",
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'License :: DFSG approved',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ]
)
