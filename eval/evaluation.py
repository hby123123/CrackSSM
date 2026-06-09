import numpy as np
import os
import logging
import glob
import cv2
import argparse 

def cal_global_acc(pred, gt):
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls = 2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))


    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])

        counts.append([tp, fp, fn])

    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls

    return global_acc, mean_acc, mean_iou_acc

def prf_metrics(pred_list, gt_list):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (((pred / np.max(pred))>0.5)).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))

    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    print("tp:{}, fp:{}, fn:{}".format(tp,fp,fn))
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r_acc = tp / (tp + fn)
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc,r_acc,f_acc


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

    return final_accuracy_all

def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)

            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            F1_list.append(F1)

        assert len(p_acc_list)==100, "p_acc_list is not 100"
        assert len(r_acc_list)==100, "r_acc_list is not 100"
        assert len(F1_list)==100, "F1_list is not 100"
        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)

    final_F1 = np.sum(np.array(final_F1_list))/len(final_F1_list)
    return final_F1

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    save_data = {
        "ODS": [],
    }
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc=0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc==0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)

        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if im is None:
        print(f"Warning: cannot read image {path}")
        return None
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(gt_dir, pred_dir):
    TARGET_SIZE = (448, 448) 
    pred_files = glob.glob(os.path.join(pred_dir, '*'))
    pred_map = {os.path.splitext(os.path.basename(p))[0]: p for p in pred_files}
    
    gt_files = glob.glob(os.path.join(gt_dir, '*'))
    
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    
    for gt_path in gt_files:
        gt_basename = os.path.splitext(os.path.basename(gt_path))[0]
        if gt_basename in pred_map:
            pred_path = pred_map[gt_basename]
            gt_img = imread(gt_path, thresh=127)
            pred_img = imread(pred_path)
            
            if gt_img is not None and pred_img is not None:
                gt_img_resized = cv2.resize(gt_img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                _, gt_img_binary = cv2.threshold(gt_img_resized, 127, 255, cv2.THRESH_BINARY)
                gt_imgs.append(gt_img_binary)  
                pred_imgs.append(pred_img) 
                gt_imgs_names.append(gt_path)
                pred_imgs_names.append(pred_path)

        else:
            print(f"No matching prediction for gt {gt_path}")
            
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate segmentation results.')
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth image directory.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Prediction image directory.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"GT dir: {args.gt_dir}")
    logging.info(f"Pred dir: {args.pred_dir}")
    
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(args.gt_dir, args.pred_dir)
    
    if not src_img_list:
        logging.error("No matched image pairs found, exit.")
        exit()
        
    logging.info(f"Matched {len(src_img_list)} image pairs for evaluation.")
    assert len(src_img_list) == len(tgt_img_list)

    final_accuracy_all = cal_prf_metrics(src_img_list, tgt_img_list)
    final_accuracy_all = np.array(final_accuracy_all)

    thresholds_list = final_accuracy_all[:, 0]
    Precision_list = final_accuracy_all[:, 1]
    Recall_list = final_accuracy_all[:, 2]
    F_list = final_accuracy_all[:, 3]

    best_idx = np.argmax(F_list)
    optimal_threshold = thresholds_list[best_idx]
    optimal_P = Precision_list[best_idx]
    optimal_R = Recall_list[best_idx]
    optimal_F1 = F_list[best_idx]

    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list) 
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list) 
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)

    print("\n----------- Evaluation Results -----------")
    print(f"mIoU : {mIoU:.4f}")
    print(f"F: {optimal_F1:.4f}")
    print(f"Precision : {optimal_P:.4f}")
    print(f"Recall : {optimal_R:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"ODS:{ODS:.4f}")
    # print("---------------------------------")
    print(f"OIS : {OIS:.4f}")
    print("Done.")


