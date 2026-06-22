from functools import partial
import hashlib
import numpy as np
from multiprocessing import Pool
from mmengine.fileio import dump, load
from mmengine.utils import ProgressBar
from .AP import instance_match, average_precision
import prettytable
from time import time
from functools import cached_property
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from mmengine.config import Config
from copy import deepcopy
import os
from collections import defaultdict
from mmengine.registry import DATASETS as MMENGINE_DATASETS

try:
    from plugin.datasets.map_utils.rdx_schema import (
        ATTRIBUTE_SCHEMAS,
        ATTRIBUTE_VALUE_TO_INDEX,
    )
except ImportError:  # pragma: no cover - fallback when schema unavailable
    ATTRIBUTE_SCHEMAS = {}
    ATTRIBUTE_VALUE_TO_INDEX = {}

from ..builder import build_dataloader

INTERP_NUM = 200 # number of points to interpolate during evaluation
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
N_WORKERS = int(os.getenv('MAPTRACKER_EVAL_WORKERS', '16'))  # num workers to parallel
SAMPLE_DIST = 0.15


class VectorEvaluate(object):
    """Evaluator for vectorized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS) -> None:
        self.dataset = MMENGINE_DATASETS.build(dataset_cfg)
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        cfg_n_workers = dataset_cfg.get('n_workers', n_workers) if hasattr(dataset_cfg, 'get') else n_workers
        self.n_workers = max(int(cfg_n_workers), 0)
        self.eval_attributes = bool(dataset_cfg.get('eval_attributes', True)) if hasattr(dataset_cfg, 'get') else True
        self.new_split = 'newsplit' in self.dataset.ann_file
        self.roi_size = self.dataset.roi_size
        if self.roi_size == (60, 30):
            self.thresholds = [0.5, 1.0, 1.5]
        elif self.roi_size == (100, 50):
            self.thresholds = [1.0, 1.5, 2.0]
        else:
            self.thresholds = [0.5, 1.0, 1.5]

        self.attr_specs_by_cls = {}
        self.attr_value_names = {}
        if ATTRIBUTE_SCHEMAS:
            for attr_name, schema in ATTRIBUTE_SCHEMAS.items():
                applies = []
                for cls_name in schema.get('applies_to', []):
                    if cls_name in self.cat2id:
                        applies.append(self.cat2id[cls_name])
                if not applies:
                    continue
                value_to_index = ATTRIBUTE_VALUE_TO_INDEX.get(attr_name, {})
                default_label = schema.get('default')
                default_index = value_to_index.get(default_label, 0)
                value_name_map = {idx: name for name, idx in value_to_index.items()}
                if value_name_map:
                    self.attr_value_names[attr_name] = value_name_map
                for cls_id in applies:
                    specs = self.attr_specs_by_cls.setdefault(cls_id, {})
                    specs[attr_name] = dict(
                        value_to_index=value_to_index,
                        default_index=default_index,
                    )
        self.global_attr_names = sorted({attr for specs in self.attr_specs_by_cls.values() for attr in specs})
        if self.thresholds:
            self.attr_eval_thr_idx = min(len(self.thresholds)//2, len(self.thresholds)-1)
        else:
            self.attr_eval_thr_idx = 0
        
    @cached_property
    def gts(self) -> Dict[str, Dict[int, List[NDArray]]]:
        roi_size = self.dataset.roi_size
        if 'av2' in self.dataset.ann_file:
            dataset = 'av2'
        elif 'rdx' in self.dataset.ann_file.lower():
            dataset = 'RDX'
        else:
            dataset = 'nusc'
        # Include a hash of the annotation file path so different splits
        # don't collide on the same cache file.
        ann_hash = hashlib.md5(self.dataset.ann_file.encode()).hexdigest()[:8]
        if self.new_split:
            tmp_file = f'./tmp_gts_{dataset}_{roi_size[0]}x{roi_size[1]}_{ann_hash}_newsplit.pkl'
        else:
            tmp_file = f'./tmp_gts_{dataset}_{roi_size[0]}x{roi_size[1]}_{ann_hash}.pkl'
        if os.path.exists(tmp_file):
            print(f'loading cached gts from {tmp_file}')
            gts = load(tmp_file)
            return gts
        
        print('collecting gts...')
        gts = {}
        self.dataloader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.n_workers,
            num_gpus=1,
            dist=False,
            shuffle=False,
            runner_type=dict(type='IterBasedRunner'))
        pbar = ProgressBar(len(self.dataloader))
        for data in self.dataloader:
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['vectors'].data[0][0])
            gts[token] = gt
            pbar.update()
            del data # avoid dataloader memory crash
        
        if not os.path.exists(tmp_file):
            print(f"saving gt to {tmp_file}")
            dump(gts, tmp_file)
        return gts
    
    def interp_fixed_num(self, 
                         vector: NDArray, 
                         num_pts: int) -> NDArray:
        ''' Interpolate a polyline.
        
        Args:
            vector (array): line coordinates, shape (M, 2)
            num_pts (int): 
        
        Returns:
            sampled_points (array): interpolated coordinates
        '''
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_pts)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        
        return sampled_points
    
    def interp_fixed_dist(self, 
                          vector: NDArray,
                          sample_dist: float) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            vector (LineString): vector
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''
        line = LineString(vector)
        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    def _evaluate_single(self, 
                         pred_vectors: List, 
                         scores: List, 
                         groundtruth: List, 
                         attr_preds: Dict[str, List[int]],
                         attr_gts: Dict[str, List[int]],
                         thresholds: List, 
                         metric: str='metric') -> Dict[int, NDArray]:
        ''' Do single-frame matching for one class.
        
        Args:
            pred_vectors (List): List[vector(ndarray) (different length)], 
            scores (List): List[score(float)]
            groundtruth (List): List of vectors
            thresholds (List): List of thresholds
        
        Returns:
            Dict: matching results keyed by threshold. Also accumulates attribute stats.
        '''

        pred_lines = []

        # interpolate predictions
        for vector in pred_vectors:
            vector = np.array(vector)
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        if pred_lines:
            pred_lines = np.stack(pred_lines)
        else:
            pred_lines = np.zeros((0, INTERP_NUM, 2))

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        if gt_lines:
            gt_lines = np.stack(gt_lines)
        else:
            gt_lines = np.zeros((0, INTERP_NUM, 2))
        
        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric)
        tp_fp_score_by_thr = {}
        match_indices_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp, match_idx = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
            match_indices_by_thr[thr] = match_idx

        attr_counts = {}
        attr_value_counts = {}
        if self.eval_attributes and attr_preds:
            mid_idx = min(len(thresholds) // 2, len(thresholds) - 1)
            thr_key = thresholds[mid_idx]
            match_idx_thr = match_indices_by_thr[thr_key]
            for attr_name, pred_values in attr_preds.items():
                value_to_index = ATTRIBUTE_VALUE_TO_INDEX.get(attr_name, {})
                if not value_to_index:
                    continue
                value_counter = {idx: {'tp': 0, 'fp': 0, 'fn': 0} for idx in value_to_index.values()}
                preds_array = np.asarray(pred_values, dtype=np.int64)
                if preds_array.ndim == 0:
                    preds_array = np.array([int(preds_array)])
                gt_values = np.asarray(attr_gts.get(attr_name, []), dtype=np.int64)
                tp_attr = 0
                num_preds_attr = min(len(preds_array), len(match_idx_thr))
                for det_idx in range(num_preds_attr):
                    matched_gt = match_idx_thr[det_idx]
                    if matched_gt < 0 or matched_gt >= gt_values.shape[0]:
                        continue
                    pred_idx = int(preds_array[det_idx])
                    gt_idx = int(gt_values[matched_gt])
                    value_counter.setdefault(pred_idx, {'tp': 0, 'fp': 0, 'fn': 0})
                    value_counter.setdefault(gt_idx, {'tp': 0, 'fp': 0, 'fn': 0})
                    if pred_idx == gt_idx:
                        value_counter[gt_idx]['tp'] += 1
                        tp_attr += 1
                    else:
                        value_counter[pred_idx]['fp'] += 1
                        value_counter[gt_idx]['fn'] += 1
                fp_attr = sum(v['fp'] for v in value_counter.values())
                fn_attr = sum(v['fn'] for v in value_counter.values())
                attr_counts[attr_name] = {
                    'tp': tp_attr,
                    'fp': fp_attr,
                    'fn': fn_attr,
                }
                attr_value_counts[attr_name] = value_counter

        return tp_fp_score_by_thr, attr_counts, attr_value_counts
        
    def evaluate(self, 
                 result_path: str, 
                 metric: str='chamfer', 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. We use multi-worker to speed up.
        
        Args:
            result_path (str): path to submission file
            metric (str): distance metric. Default: 'chamfer'
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            new_result_dict (Dict): evaluation results. AP by categories.
        '''
        
        results = load(result_path)
        results = results['results']
        
        # re-group samples and gt by label
        samples_by_cls = {label: [] for label in self.id2cat.keys()}
        num_gts = {label: 0 for label in self.id2cat.keys()}
        num_preds = {label: 0 for label in self.id2cat.keys()}

        attr_global_counts = {}
        attr_value_global_counts = {}
        if self.eval_attributes:
            attr_global_counts = {attr: {'tp': 0, 'fp': 0, 'fn': 0} for attr in self.global_attr_names}
            attr_value_global_counts = {
                attr: {idx: {'tp': 0, 'fp': 0, 'fn': 0} for idx in value_map.keys()}
                for attr, value_map in self.attr_value_names.items()
            }

        # align by token
        for token, gt in self.gts.items():
            if token in results.keys():
                pred = results[token]
            else:
                pred = {'vectors': [], 'scores': [], 'labels': [], 'attr_preds': {}}
            
            # for every sample
            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}
            attr_preds_by_cls = None
            gt_attr_by_cls = None
            if self.eval_attributes:
                attr_preds_by_cls = {label: defaultdict(list) for label in self.id2cat.keys()}
                gt_attr_by_cls = {label: defaultdict(list) for label in self.id2cat.keys()}
                for label_id, gt_vectors in gt.items():
                    class_specs = self.attr_specs_by_cls.get(label_id, {})
                    if not class_specs:
                        continue
                    for vec in gt_vectors:
                        attrs = getattr(vec, 'attrs', {}) if hasattr(vec, 'attrs') else {}
                        for attr_name, spec in class_specs.items():
                            value = attrs.get(attr_name, None)
                            if isinstance(value, (list, tuple)) and value:
                                value = value[0]
                            if value is None:
                                attr_idx = spec['default_index']
                            else:
                                attr_idx = spec['value_to_index'].get(value, spec['default_index'])
                            gt_attr_by_cls[label_id][attr_name].append(int(attr_idx))
                    # Ensure counts align even when empty
                    for attr_name, spec in class_specs.items():
                        gt_attr_by_cls[label_id].setdefault(attr_name, [])

            for i in range(len(pred['labels'])):
                # i-th pred line in sample
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                score = pred['scores'][i]

                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)
                if self.eval_attributes:
                    attr_preds_case = pred.get('attr_preds', {})
                    for attr_name, attr_values in attr_preds_case.items():
                        if len(attr_values) > i:
                            attr_preds_by_cls[label][attr_name].append(int(attr_values[i]))

            for label in self.id2cat.keys():
                attr_pred_dict = {}
                attr_gt_dict = {}
                if self.eval_attributes:
                    class_specs = self.attr_specs_by_cls.get(label, {})
                    if class_specs:
                        for attr_name in class_specs.keys():
                            attr_pred_dict[attr_name] = list(attr_preds_by_cls[label].get(attr_name, []))
                            attr_gt_dict[attr_name] = list(gt_attr_by_cls[label].get(attr_name, []))

                gt_vectors_label = gt.get(label, [])
                new_sample = (vectors_by_cls[label], scores_by_cls[label], gt_vectors_label,
                              attr_pred_dict, attr_gt_dict)
                num_gts[label] += len(gt_vectors_label)
                num_preds[label] += len(scores_by_cls[label])
                samples_by_cls[label].append(new_sample)

        result_dict = {}

        print(f'\nevaluating {len(self.id2cat)} categories...')
        start = time()
        if self.n_workers > 0:
            pool = Pool(self.n_workers)
        
        sum_mAP = 0
        pbar = ProgressBar(len(self.id2cat))
        for label in self.id2cat.keys():
            samples = samples_by_cls[label] # List[(pred_lines, scores, gts)]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label]
            }
            sum_AP = 0

            fn = partial(self._evaluate_single, thresholds=self.thresholds, metric=metric)
            if self.n_workers > 0:
                eval_results = pool.starmap(fn, samples)
            else:
                eval_results = [fn(*sample) for sample in samples]

            tpfp_score_list = [res[0] for res in eval_results]
            attr_counts_list = [res[1] for res in eval_results]
            attr_value_counts_list = [res[2] for res in eval_results]
            
            for thr in self.thresholds:
                tp_fp_score = [i[thr] for i in tpfp_score_list]
                tp_fp_score = np.vstack(tp_fp_score) # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])

                tp = tp_fp_score[sort_inds, 0] # (num_dets,)
                fp = tp_fp_score[sort_inds, 1] # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum((tp + fp), eps)

                AP = average_precision(recalls, precisions, 'area')
                sum_AP += AP
                result_dict[self.id2cat[label]].update({f'AP@{thr}': AP})

            pbar.update()
            
            AP = sum_AP / len(self.thresholds)
            sum_mAP += AP

            result_dict[self.id2cat[label]].update({f'AP': AP})

            if self.eval_attributes:
                for sample_attr_counts in attr_counts_list:
                    for attr_name, counts in sample_attr_counts.items():
                        if attr_name in attr_global_counts:
                            attr_global_counts[attr_name]['tp'] += counts['tp']
                            attr_global_counts[attr_name]['fp'] += counts['fp']
                            attr_global_counts[attr_name]['fn'] += counts['fn']
                for sample_attr_value_counts in attr_value_counts_list:
                    for attr_name, value_counts in sample_attr_value_counts.items():
                        if attr_name not in attr_value_global_counts:
                            attr_value_global_counts[attr_name] = {}
                        for idx, counts in value_counts.items():
                            agg_map = attr_value_global_counts[attr_name]
                            if idx not in agg_map:
                                agg_map[idx] = {'tp': 0, 'fp': 0, 'fn': 0}
                            agg_map[idx]['tp'] += counts['tp']
                            agg_map[idx]['fp'] += counts['fp']
                            agg_map[idx]['fn'] += counts['fn']
        
        if self.n_workers > 0:
            pool.close()
        
        mAP = sum_mAP / len(self.id2cat.keys())
        result_dict.update({'mAP': mAP})

        attr_metrics = {}
        attr_value_metrics = {}
        if self.eval_attributes:
            eps = np.finfo(np.float32).eps
            for attr_name in self.global_attr_names:
                counts = attr_global_counts[attr_name]
                tp = counts['tp']
                fp = counts['fp']
                fn = counts['fn']
                precision = tp / max(tp + fp, eps) if (tp + fp) > 0 else 0.0
                recall = tp / max(tp + fn, eps) if (tp + fn) > 0 else 0.0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                attr_metrics[attr_name] = dict(
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
                result_dict[f'attr_{attr_name}_precision'] = precision
                result_dict[f'attr_{attr_name}_recall'] = recall
                result_dict[f'attr_{attr_name}_f1'] = f1
                value_metrics = {}
                value_counts_map = attr_value_global_counts.get(attr_name, {})
                name_map = self.attr_value_names.get(attr_name, {})
                for idx, count_vals in value_counts_map.items():
                    tp_v = count_vals['tp']
                    fp_v = count_vals['fp']
                    fn_v = count_vals['fn']
                    precision_v = tp_v / max(tp_v + fp_v, eps) if (tp_v + fp_v) > 0 else 0.0
                    recall_v = tp_v / max(tp_v + fn_v, eps) if (tp_v + fn_v) > 0 else 0.0
                    if precision_v + recall_v > 0:
                        f1_v = 2 * precision_v * recall_v / (precision_v + recall_v)
                    else:
                        f1_v = 0.0
                    value_metrics[idx] = dict(
                        precision=precision_v,
                        recall=recall_v,
                        f1=f1_v,
                        tp=tp_v,
                        fp=fp_v,
                        fn=fn_v,
                    )
                    value_label = name_map.get(idx, f'value_{idx}')
                    safe_label = value_label.replace(' ', '_')
                    metric_prefix = f'attr_{attr_name}_{safe_label}'
                    result_dict[f'{metric_prefix}_precision'] = precision_v
                    result_dict[f'{metric_prefix}_recall'] = recall_v
                    result_dict[f'{metric_prefix}_f1'] = f1_v
                if value_metrics:
                    attr_value_metrics[attr_name] = value_metrics

        print(f"finished in {time() - start:.2f}s")

        # print results
        table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
                [f'AP@{thr}' for thr in self.thresholds] + ['AP'])
        for label in self.id2cat.keys():
            table.add_row([
                self.id2cat[label], 
                result_dict[self.id2cat[label]]['num_preds'],
                result_dict[self.id2cat[label]]['num_gts'],
                *[round(result_dict[self.id2cat[label]][f'AP@{thr}'], 4) for thr in self.thresholds],
                round(result_dict[self.id2cat[label]]['AP'], 4),
            ])
        
        from mmengine.logging import print_log
        print_log('\n'+str(table), logger=logger)
        if attr_metrics:
            attr_table = prettytable.PrettyTable(['attribute', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'])
            for attr_name, metrics in attr_metrics.items():
                attr_table.add_row([
                    attr_name,
                    round(metrics['precision'], 4),
                    round(metrics['recall'], 4),
                    round(metrics['f1'], 4),
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn'],
                ])
                value_metrics = attr_value_metrics.get(attr_name, {})
                name_map = self.attr_value_names.get(attr_name, {})
                for idx, vm in value_metrics.items():
                    label = name_map.get(idx, f'value_{idx}')
                    attr_table.add_row([
                        f'{attr_name}/{label}',
                        round(vm['precision'], 4),
                        round(vm['recall'], 4),
                        round(vm['f1'], 4),
                        vm['tp'],
                        vm['fp'],
                        vm['fn'],
                    ])
            print_log('\n'+str(attr_table), logger=logger)
            acc_table = prettytable.PrettyTable(['attribute', 'accuracy', 'total_preds', 'total_gts', 'tp'])
            for attr_name, metrics in attr_metrics.items():
                total_preds_attr = metrics['tp'] + metrics['fp']
                accuracy_pred = metrics['tp'] / total_preds_attr if total_preds_attr > 0 else 0.0
                acc_table.add_row([
                    attr_name,
                    round(accuracy_pred, 4),
                    total_preds_attr,
                    metrics['tp'] + metrics['fn'],
                    metrics['tp'],
                ])
            print_log('\n'+str(acc_table), logger=logger)

        mAP_normal = 0
        for label in self.id2cat.keys():
            for thr in self.thresholds:
                mAP_normal += result_dict[self.id2cat[label]][f'AP@{thr}']

        if 'av2' in self.dataset.ann_file:
            divisor = 9#dataset = 'av2'
        elif 'rdx' in self.dataset.ann_file.lower():
            divisor = 10*3#dataset = 'RDX'
        else:
            divisor = 9#'nusc'

        mAP_normal = mAP_normal / divisor#9
        print_log(f'mAP_normal = {mAP_normal:.4f}\n', logger=logger)
        # print_log(f'mAP_hard = {mAP_easy:.4f}\n', logger=logger)

        new_result_dict = {}
        for name in self.cat2id:
            new_result_dict[name] = result_dict[name]['AP']

        new_result_dict['mAP']= mAP_normal
        for attr_name, metrics in attr_metrics.items():
            new_result_dict[f'attr_{attr_name}_precision'] = metrics['precision']
            new_result_dict[f'attr_{attr_name}_recall'] = metrics['recall']
            new_result_dict[f'attr_{attr_name}_f1'] = metrics['f1']
            total_preds_attr = metrics['tp'] + metrics['fp']
            accuracy_pred = metrics['tp'] / total_preds_attr if total_preds_attr > 0 else 0.0
            new_result_dict[f'attr_{attr_name}_accuracy'] = accuracy_pred
        for attr_name, value_metrics in attr_value_metrics.items():
            name_map = self.attr_value_names.get(attr_name, {})
            for idx, metrics in value_metrics.items():
                label = name_map.get(idx, f'value_{idx}')
                safe_label = label.replace(' ', '_')
                prefix = f'attr_{attr_name}_{safe_label}'
                new_result_dict[f'{prefix}_precision'] = metrics['precision']
                new_result_dict[f'{prefix}_recall'] = metrics['recall']
                new_result_dict[f'{prefix}_f1'] = metrics['f1']

        return new_result_dict
