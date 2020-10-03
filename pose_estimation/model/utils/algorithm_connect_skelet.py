
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
    Store keypoints of the single human

    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self):
        """
        Init class to store keypoints of the single human

        """
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def to_list(self, th_hold=0.2) -> list:
        """
        Transform keypoints stored in this class to list

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        list
            List with lenght NK * 3, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for visibility of the points
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        list_data = []
        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is None or take_single.score < th_hold:
                list_data += [0.0, 0.0, 0.0]
            else:
                list_data += [
                    self.body_parts[i].x,
                    self.body_parts[i].y,
                    self.body_parts[i].score,
                ]

        return list_data

    def to_dict(self, th_hold=0.2) -> dict:
        """
        Transform keypoints stored in this class to dict

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        dict
            Dict of the keypoints,
            { NumKeypoints: [x_coord, y_coord, score],
              NumKeypoints_1: [x_coord, y_coord, score],
              ..........................................
            }
            Where NumKeypoints, NumKeypoints_1 ... are integer value responsible for index of the keypoint,
            x_coord - coordinate of the keypoint on X axis
            y_coord - coordinate of the keypoint on Y axis
            score - confidence of the neural network
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        dict_data = {}
        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is not None and take_single.score >= th_hold:
                dict_data.update({
                    i: [take_single.x, take_single.y, take_single.score]
                })
            else:
                dict_data.update({
                    i: [0.0, 0.0, 0.0]
                })

        return dict_data

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    Store single keypoints with certain coordinates and score

    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        """
        Init

        Parameters
        ----------
        uidx : str
            String stored number of the human and number of this keypoint
        part_idx :
        x : float
            Coordinate of the keypoint at the x-axis
        y : float
            Coordinate of the keypoint at the y-axis
        score : float
            Confidence score from neural network
        """
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
        human = Human()
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
        # if at least one keypoint was visible add `human` to all humans
        # Otherwise skip
        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans


def merge_similar_skelets(humans: list, th_hold_x=0.04, th_hold_y=0.04):
    """
    Merge similar skeletons into one skelet

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
        Dictionary with key equal to number of the human and value as Human class
    """
    humans_dict = dict([(str(i), humans[i]) for i in range(len(humans))])

    while True:
        is_merge = False
        for h1, h2 in itertools.combinations(list(range(len(humans_dict))), 2):
            if humans_dict.get(str(h1)) is None or humans_dict.get(str(h2)) is None:
                continue

            for c1, c2 in itertools.product(humans_dict[str(h1)].body_parts, humans_dict[str(h2)].body_parts):
                single_keypoints_1 = humans[h1].body_parts[c1]
                single_keypoints_2 = humans[h2].body_parts[c2]
                if (abs(single_keypoints_1.x - single_keypoints_2.x) < th_hold_x and
                    abs(single_keypoints_1.y - single_keypoints_2.y) < th_hold_y
                ):
                    is_merge = True
                    humans_dict[str(h1)].body_parts.update(humans[h2].body_parts)
                    humans_dict.pop(str(h2))
                    break

        if not is_merge:
            break

    return humans_dict
