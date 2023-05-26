import numpy as np
import torch
import torch.nn.functional as F


def gen_msf_cam(model, images, scales, args, ent):
    B, _, H, W = images.shape
    C = model.cls_token.shape[1]

    cam_list = []
    for scale in scales:
        size = [int(H * scale), int(W * scale)]
        if size[0] == H and size[1] == W:
            img = images
        else:
            img = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        img_cat = torch.cat([img, img.flip(-1)], dim=0)

        _, _, cam, att = model(img_cat)
        att = att.softmax(dim=-1)  # softmax  # (num_layers, batch_size, num_heads, seq_len, seq_len)
        att = att.mean(2)  # averaged over attention heads

        if C > 1:  # MCTformer
            cls_att = att.mean(0)[:, :C, C:].reshape(cam.shape)
            cam = cam * cls_att

        if not args.use_ent:
            pat_att = att[: args.layer_index].mean(0)[:, C:, C:]
        else:
            w = 1.0 - ent[: args.layer_index]
            pat_att = att[: args.layer_index] * w.reshape(w.shape[0], 1, 1, 1)
            pat_att = pat_att.mean(0)[:, C:, C:]

        b, c, h, w = cam.shape
        cam = torch.einsum("bmn,bcn->bcm", pat_att, cam.reshape(b, c, -1)).reshape(b, c, h, w)

        cam = F.interpolate(cam, size=[H, W], mode="bilinear", align_corners=False)
        # cam = torch.max(cam[:B, ...], cam[B:, ...].flip(-1))
        # cam_list.append(cam)
        cam_list.append(cam[:B])
        cam_list.append(cam[B:].flip(-1))

    cam = torch.stack(cam_list, dim=0).sum(dim=0)
    cam = F.relu(cam)
    cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
    cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-6)
    return cam


def cam_to_seg(par, imgs, keys, cams, score, img_box=None):
    assert cams.size()[0] == keys.size()[0] and cams.size()[1] == keys.size()[1]
    B, _, H, W = imgs.shape

    # pad keys
    pads = torch.ones(B, 1)
    pads = pads.to(device=keys.device, dtype=keys.dtype)
    keys = torch.cat([pads, keys], dim=1)

    # pad cams
    pads = torch.ones(B, 1, H, W) * score
    pads = pads.to(device=cams.device, dtype=cams.dtype)
    cams = torch.cat([pads, cams], dim=1)

    size = [int(H / 2), int(W / 2)]
    imgs = F.interpolate(imgs, size=size, mode="bilinear", align_corners=False)
    cams = F.interpolate(cams, size=size, mode="bilinear", align_corners=False)

    segs = torch.ones((B, H, W), device=cams.device, dtype=torch.int64) * 255

    for b in range(B):
        key = torch.nonzero(keys[b])[:, 0]
        img = imgs[b].unsqueeze(0)
        cam = cams[b].unsqueeze(0)
        cam = cam[:, key, :, :].softmax(1)

        ref_cam = par(img, cam)
        ref_cam = F.interpolate(ref_cam, size=[H, W], mode="bilinear", align_corners=False)
        seg = torch.argmax(ref_cam, dim=1)
        seg = key[seg]

        if img_box is not None:
            i1, i2, j1, j2 = img_box[b]
            segs[b, i1:i2, j1:j2] = seg[0, i1:i2, j1:j2]
        else:
            segs[b, :, :] = seg[0, :, :]

    return segs


def seg_to_aff(seg, ignore_index=255):
    B, H, W = seg.size()
    seg = F.interpolate(seg.unsqueeze(1).float(), size=[H // 16, W // 16], mode="nearest").squeeze(1)
    B, H, W = seg.size()

    seg = seg.reshape(B, 1, H * W).repeat(1, H * W, 1)
    out = (seg == seg.transpose(1, 2)).to(torch.int64)

    for b in range(B):
        out[b, :, seg[b, 0, :] == ignore_index] = ignore_index
        out[b, seg[b, 0, :] == ignore_index, :] = ignore_index
    return out
