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