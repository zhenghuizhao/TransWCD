import torch
import torch.nn.functional as F


def cam_to_label(cam, cls_label, *, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    cam_value, _pseudo_label = cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    pseudo_label = torch.ones_like(_pseudo_label)

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return cam, pseudo_label


def multi_scale_cam(model, inputs_A, inputs_B, scales):
    # cam_list = []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam: torch.Size([8, 1, 16, 16])

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam, _,_

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam