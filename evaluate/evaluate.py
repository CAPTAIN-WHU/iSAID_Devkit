#modified by akshitac8

"""Functions for evaluating results computed for a json dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import uuid

from pycocotools.cocoeval import COCOeval

from core.config import cfg
from utils.io import save_object
import utils.boxes as box_utils
from datasets.json_dataset import JsonDataset
import argparse
parser = argparse.ArgumentParser(description='Model evaluation')

parser.add_argument('--result_file', default='/home/aditya/evaluation_code_isaid/evaluate/result/segmentations_dota_patch_test_results.json', type=str, help='path for the predictions')
parser.add_argument('--set', default="val,test", type=str, help='evaluate mode')
parser.add_argument('--output_dir', default='result/', type=str, help='directory path for storing the pickle format of the results')


args = parser.parse_args()
sets=args.set.split(',')
res_file=args.result_file
output_dir=args.output_dir
for i in sets:
    if i == "val":
        dataset_name='isaid_patch_val'
    elif i == "test":
        dataset_name='isaid_patch_test'
    else:
        print("wrong input")

json_dataset=JsonDataset(dataset_name)

def _do_segmentation_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'segmentation_results.pkl')
    save_object(coco_eval, eval_file)
    print('Wrote json eval results to: {}'.format(eval_file))


def _do_detection_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'detection_results.pkl')
    save_object(coco_eval, eval_file)
    print('Wrote json eval results to: {}'.format(eval_file))


def _log_detection_eval_metrics(json_dataset, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2] # all 2000 boxes
    ap_default = np.mean(precision[precision > -1])
    print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == 'unlabeled':
            continue
        # minus 1 because of unlabeled
        
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2] # all 2000 boxes
        ap = np.mean(precision[precision > -1])
        print('{},{:.1f}'.format((cls),(100 * ap)))
    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()
    
def main():
	
	_do_segmentation_eval(json_dataset, res_file, output_dir)
	#_do_detection_eval(json_dataset, res_file, output_dir)

    
if __name__ == '__main__':
	main()
	


