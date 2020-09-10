
from .constants import NUMBER_OF_KEYPOINTS
import itertools

try:
    from .pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess.')
    exit(-1)


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


def estimate_paf(peaks, heat_mat, paf_mat):
    """
    Estimate paff by using heatmap and peaks

    Parameters
    ----------
    peaks : np.ndarray
        Numpy array of the peaks which is product of the NMS (Non maximum suppresion) from the heatmap
    heat_mat : np.ndarray
        Numpy array of the heatmap which is usually prediction of the network
    paf_mat : np.ndarray
        Numpy array of the PAF (Party affinity fields) which is usually prediction of the network

    Returns
    -------
    list
        List of the Human which contains body keypoints
    """
    pafprocess.process_paf(peaks, heat_mat, paf_mat)
    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(NUMBER_OF_KEYPOINTS):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)),
                float(pafprocess.get_part_y(c_idx)),
                pafprocess.get_part_score(c_idx)
            )

        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans


def merge_similar_skelets(humans: list, th_hold_x=0.04, th_hold_y=0.04):
    """
    Merge similar skelets into one skelet

    Parameters
    ----------
    humans : list
        List of the predicted skelets from `estimate_paf` script
    th_hold_x : float
        Threshold from what value do we count keypoints similar by axis X,
        By default equal to 0.04
    th_hold_y : float
        Threshold from what value do we count keypoints similar by axis Y,
        By default equal to 0.04

    Returns
    -------
    dict
        Dictionary where key is number of skelet, value is skelet
    """
    humans_dict = dict([(str(i), humans[i]) for i in range(len(humans))])

    while True:
        is_merge = False
        for h1, h2 in itertools.combinations(list(range(len(humans_dict))), 2):
            if humans_dict.get(str(h1)) is None or humans_dict.get(str(h2)) is None:
                continue
            for c1, c2 in itertools.product(humans_dict[str(h1)].body_parts, humans_dict[str(h2)].body_parts):
                single_1 = humans[h1].body_parts[c1]
                single_2 = humans[h2].body_parts[c2]
                if abs(single_1.x - single_2.x) < th_hold_x and abs(single_1.y - single_2.y) < th_hold_y:
                    is_merge = True
                    humans_dict[str(h1)].body_parts.update(humans[h2].body_parts)
                    humans_dict.pop(str(h2))
                    break

        if not is_merge:
            break

    return humans_dict
