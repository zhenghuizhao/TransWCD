# 将 val数据集换为test，取消val(如ChangeFormer)

import argparse
import os
from collections import OrderedDict
from PIL import Image
from utils.camutils_CD import cam_to_label, multi_scale_cam
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import weaklyCD
from utils import evaluate_CD
from models.model_transwcd import TransWCD_single, TransWCD_dual

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config",default='configs/LEVIR.yaml',type=str,
                    help="config")
parser.add_argument("--save_dir", default="./results/WHU", type=str, help="save_dir")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument("--model_path", default="/data/zhenghui.zhao/Code/TransWCD/transwcd/results/test_models/transwcd_iter_28000.pth", type=str, help="model_path")

parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side (256 or 512)")



def test(model, dataset, test_scales=1.0):
    gts, cams = [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader)):
            ### 注意此处cls_label ###
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()

            b, c, h, w = inputs_A.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()


            _, _, h, w = inputs_A.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_A = F.interpolate(inputs_A, size=(_h, _w), mode='bilinear', align_corners=False)

            _, _, h, w = inputs_B.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_B = F.interpolate(inputs_B, size=(_h, _w), mode='bilinear', align_corners=False)


            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            # 以png格式保存cam结果
            cam_path = args.save_dir + '/prediction/' + name[0] + '.png'
            cam_img = Image.fromarray((cam_label.squeeze().cpu().numpy() * 255).astype(np.uint8))
            cam_img.save(cam_path)

            ### FN and FP color ###
            cam = cam_label.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Create RGB image from labels
            label_rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            label_rgb[labels == 0] = [0, 0, 0]  # Background (black)
            label_rgb[labels == 1] = [255, 255, 255]  # Foreground (white)

            # Mark FN pixels as blue
            fn_pixels = np.logical_and(cam == 0, labels == 1)  # False Negatives
            label_rgb[fn_pixels] = [0, 0, 255]  # Blue

            # Mark FP pixels as red
            fp_pixels = np.logical_and(cam == 1, labels == 0)  # False Positives
            label_rgb[fp_pixels] = [255, 0, 0]  # Red

            # Save the labeled image
            label_with_fn_fp_path = args.save_dir + '/prediction_color/' + name[0] + '.png'
            label_with_fn_fp_img = Image.fromarray(label_rgb)
            label_with_fn_fp_img.save(label_with_fn_fp_path)

        return inputs_A, inputs_B, gts, cams


def main(cfg):
    test_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='test',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    if cfg.scheme == "transwcd_dual":
        transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=256,
                                 pretrained=True,
                                 pooling=args.pooling, )
    elif cfg.scheme == "transwcd_single":
        transwcd = TransWCD_single(backbone=cfg.backbone.config,
                                   stride=cfg.backbone.stride,
                                   num_classes=cfg.dataset.num_classes,
                                   embedding_dim=256,
                                   pretrained=True,
                                   pooling=args.pooling, )
    else:
        print('Please fill in cfg.scheme!')

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        if 'diff.0.bias' in k:
            k = k.replace('diff.0.bias', 'diff.bias')
        if 'diff.0.weight' in k:
            k = k.replace('diff.0.weight', 'diff.weight')
        new_state_dict[k] = v

    transwcd.load_state_dict(state_dict=new_state_dict, strict=True)  # True
    transwcd.eval()

    ###   test 输出 ###
    inputs_A, inputs_B, gts, cams = test(model=transwcd, dataset=test_dataset, test_scales=[1.0])
    torch.cuda.empty_cache()


    cams_score = evaluate_CD.scores(gts, cams)

    print("cams score:")
    print(cams_score)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)

    os.makedirs(args.save_dir + "/prediction", exist_ok=True)
    os.makedirs(args.save_dir + "/prediction_color", exist_ok=True)

    main(cfg=cfg)



