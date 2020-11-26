# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup
import setuptools

setup(
    name='MakiPoseNet',
    packages=setuptools.find_packages(),
    package_data={'pose_estimation': ['model/utils/pafprocess/*.so']},
    version='0.5.0',
    description='A set of utilities for training pose estimation models',
    long_description='...',
    author='Kilbas Igor, Gribanov Danil',
    author_email='igor.kilbas.ai@gmail.com',
    url='https://github.com/MakiResearchTeam/MakiPoseNet.git',
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)