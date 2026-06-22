import torch
from mmengine.fileio import load
from mmengine.utils import track_iter_progress
from functools import cached_property
import prettytable
from numpy.typing import NDArray
from typing import Dict, Optional
from logging import Logger
from mmengine.config import Config
from copy import deepcopy
from mmengine.registry import DATASETS as MMENGINE_DATASETS

from ..builder import build_dataloader

N_WORKERS = 16

class RasterEvaluate(object):
    """Evaluator for rasterized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS):
        self.dataset = MMENGINE_DATASETS.build(dataset_cfg)
        self.dataloader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=n_workers,
            num_gpus=1,
            dist=False,
            shuffle=False,
            runner_type=dict(type='IterBasedRunner'))
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers

    @cached_property
    def gts(self) -> Dict[str, NDArray]:
        print('collecting gts...')
        gts = {}
        total = len(self.dataloader.dataset)
        for data in track_iter_progress((self.dataloader, total)):
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['semantic_mask'].data[0][0])
            gts[token] = gt
            del data # avoid dataloader memory crash
        
        return gts

    def evaluate(self, 
                 result_path: str, 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. 
        
        Args:
            result_path (str): path to submission file
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            result_dict (Dict): evaluation results. IoU by categories.
        '''
        
        predictions = load(result_path)
        meta = predictions['meta']
        results = predictions['results']

        result_dict = {}

        gts = []
        preds = []
        for token, gt in self.gts.items():
            gts.append(gt)
            pred = torch.zeros((len(self.cat2id), gt.shape[1], gt.shape[2])).bool()
            if token in results:
                semantic_mask = torch.tensor(results[token]['semantic_mask'])
                for label_i in range(gt.shape[0]):
                    pred[label_i] = (semantic_mask == label_i+1)
            preds.append(pred)
        
        preds = torch.stack(preds).bool()
        gts = torch.stack(gts).bool()

        # TODO: flip the gt
        gts = torch.flip(gts, [2,])

        # for every label
        total = 0
        for i in range(gts.shape[1]):
            category = self.id2cat[i]
            pred = preds[:, i]
            gt = gts[:, i]
            intersect = (pred & gt).sum().float().item()
            union = (pred | gt).sum().float().item()
            result_dict[category] = intersect / (union + 1e-7)
            total += result_dict[category]
        
        mIoU = total / gts.shape[1]
        result_dict['mIoU'] = mIoU
        
        categories = list(self.cat2id.keys())
        table = prettytable.PrettyTable([' ', *categories, 'mean'])
        table.add_row(['IoU', 
            *[round(result_dict[cat], 4) for cat in categories], 
            round(mIoU, 4)])
        
        if logger:
            from mmengine.logging import print_log
            print_log('\n'+str(table), logger=logger)
            print_log(f'mIoU = {mIoU:.4f}\n', logger=logger)

        return result_dict
