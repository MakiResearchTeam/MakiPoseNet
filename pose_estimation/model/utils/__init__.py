

from .constants import CONNECT_INDEXES_FOR_PAFF, NUMBER_OF_KEYPOINTS

try:
    from .pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess.')
    exit(-1)
