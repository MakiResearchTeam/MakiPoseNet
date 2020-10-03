# ------------------------------------------------------------------------------
# Adapted from https://github.com/cocodataset/cocoapi
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np
import datetime
import time
from collections import defaultdict
import copy

from .utils.eval_constants import MAKI_SKELET_K
from .utils import KEYPOINTS


class MYeval_wholebody:

    def __init__(self, cocoGt=None, cocoDt=None):
        """
        Initialize CocoEval using coco APIs for gt and dt

        Parameters
        ----------
        cocoGt: coco object
            Coco object with ground truth annotations
        cocoDt: coco object
            Coco object with detection results

        """
        self.cocoGt = cocoGt                            # ground truth COCO API
        self.cocoDt = cocoDt                            # detections COCO API

        self.params = {}                                # evaluation parameters

        self.evalImgs = defaultdict(list)               # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}                                  # accumulated evaluation results

        self._gts = defaultdict(list)                   # gt for evaluation
        self._dts = defaultdict(list)                   # dt for evaluation

        self.params = Params()                          # parameters

        self._paramsEval = {}                           # parameters for evaluation
        self.stats = []                                 # result summarization
        self.stats_str = []                             # result in str

        self.ious = {}                                  # ious between all gts and dts

        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params

        """

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # Ignore images where no keypoints, i. e. skip them or set ignore flag
        for gt in gts:
            if gt.get(KEYPOINTS) is None:
                continue

            whole_body_gt = gt[KEYPOINTS]

            g = np.array(whole_body_gt)
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)

            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            gt['ignore'] = (k1 == 0) or gt['ignore']

        self._gts = defaultdict(list)               # gt for evaluation
        self._dts = defaultdict(list)               # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)           # per-image per-category evaluation results
        self.eval = {}                              # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs

        """
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params

        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        self.ious = {
            (imgId, catId): self.computeOks(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]
        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeOks(self, imgId, catId):
        p = self.params

        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))

        sigmas = np.array(MAKI_SKELET_K)
        vars = (sigmas * 2) ** 2
        k = len(sigmas)

        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            if gt.get(KEYPOINTS) is None:
                continue
            # create bounds for ignore regions(double the gt bbox)
            whole_body_gt = gt[KEYPOINTS]
            g = np.array(whole_body_gt)

            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]

            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']

            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2

            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2

            for i, dt in enumerate(dts):
                whole_body_dt = dt[KEYPOINTS]
                d = np.array(whole_body_dt)

                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image

        Return
        ------
        dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g.get(KEYPOINTS) is not None and (g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1])):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval

        Parameters
        ----------
        p: Params
            Input params for evaluation

        """
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval

        catIds = _pe.catIds if _pe.useCats else [-1]

        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]

        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            single_final_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            print(single_final_string)
            return mean_s, single_final_string

        def _summarizeKps():
            stats = np.zeros((10,))
            stats_str = [0] * 10
            stats[0], stats_str[0] = _summarize(1, maxDets=self.params.maxDets[0])
            stats[1], stats_str[1] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.5)
            stats[2], stats_str[2] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.75)
            stats[3], stats_str[3] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='medium')
            stats[4], stats_str[4] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='large')
            stats[5], stats_str[5] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[6], stats_str[6] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.5)
            stats[7], stats_str[7] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.75)
            stats[8], stats_str[8] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='medium')
            stats[9], stats_str[9] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='large')
            return stats, stats_str

        if not self.eval:
            raise Exception('Please run accumulate() first')

        self.stats, self.stats_str = _summarizeKps()

    def get_stats_str(self) -> str:
        final_str = ""
        for single_str in self.stats_str:
            final_str += single_str + "\n"

        return final_str

    def get_AP_AR_with_IoU_50(self):
        """
        Return AP and AR with IoU equal to 0.5

        """
        return self.stats[1], self.stats[6]

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

