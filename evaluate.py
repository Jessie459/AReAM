import argparse
import multiprocessing
import os
import os.path as osp
from datetime import datetime
from pprint import pprint

import numpy as np
from PIL import Image
from tqdm import tqdm


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type="png", threshold=1.0):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value("i", 0, lock=True))
        P.append(multiprocessing.Value("i", 0, lock=True))
        T.append(multiprocessing.Value("i", 0, lock=True))

    def compare(start, step, TP, P, T, input_type, threshold, verbose=False):
        for idx in tqdm(range(start, len(name_list), step), disable=(verbose is False)):
            name = name_list[idx]
            if input_type == "png":
                predict_file = os.path.join(predict_folder, "%s.png" % name)
                if not osp.exists(predict_file):
                    print(predict_file + " not exists")
                    predict = None
                else:
                    predict = np.array(Image.open(predict_file))
            elif input_type == "npy":
                predict_file = os.path.join(predict_folder, "%s.npy" % name)
                if not osp.exists(predict_file):
                    print(predict_file + " not exists")
                    predict = None
                else:
                    predict_dict = np.load(predict_file, allow_pickle=True).item()
                    keys = predict_dict["key"]
                    cams = predict_dict["cam"]

                    keys = np.pad(keys + 1, (1, 0), mode="constant")
                    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=threshold)
                    predict = np.argmax(cams, axis=0)
                    predict = keys[predict]
                    predict = predict.astype(np.uint8)

            gt_file = os.path.join(gt_folder, "%s.png" % name)
            gt = np.array(Image.open(gt_file))
            if predict is None:
                predict = np.copy(gt)
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, threshold, (i == 0)))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    # for i in range(num_cls):
    #     loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist["mIoU"] = miou * 100
    return loglist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="voc", choices=["voc", "coco"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--write_to", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--type", type=str, default="npy", choices=["npy", "png"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=99)

    args = parser.parse_args()
    pprint(vars(args))

    if args.dataset == "voc":
        cls_num = 21
        name_list_path = osp.join("data/voc", args.split + ".txt")
        gt_dir = osp.expanduser("~/data/VOCdevkit/VOC2012/SegmentationClass")
    else:
        cls_num = 81
        name_list_path = osp.join("data/coco", args.split + ".txt")
        gt_dir = osp.expanduser("~/data/COCO/SegmentationClass/train2014")

    name_list = np.loadtxt(name_list_path, dtype=str)
    print("name list:", len(name_list))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = "time: " + now + "\n" + "data: " + args.split + "\n"
    if args.note:
        msg += "note: " + args.note + "\n"

    best_result = None
    best_thresh = None

    if args.type == "png":
        result = do_python_eval(args.pred_dir, gt_dir, name_list, cls_num, args.type)
        best_result = result
    elif args.type == "npy":
        for i in range(args.start, args.stop):
            thresh = i / 100.0
            result = do_python_eval(args.pred_dir, gt_dir, name_list, cls_num, args.type, thresh)

            msg_per_thresh = f"thresh: {thresh:.2f} mIoU:{result['mIoU']:.2f}%"
            print(msg_per_thresh)

            if best_result is None or best_result["mIoU"] < result["mIoU"]:
                best_result = result
                best_thresh = thresh
    else:
        raise ValueError(f"invalid input type")

    msg += f"best result:\n"
    for k, v in best_result.items():
        msg += f"{k}: {v:.2f}\n"
    if args.type == "npy":
        msg += f"best thresh: {best_thresh:.2f}\n"
    print(msg)

    if args.write_to:
        dirname = os.path.dirname(os.path.abspath(args.write_to))
        os.makedirs(dirname, exist_ok=True)

        with open(args.write_to, mode="a") as f:
            f.write("-" * 30 + "\n")
            f.write(msg)
            f.write("-" * 30 + "\n")


if __name__ == "__main__":
    main()
