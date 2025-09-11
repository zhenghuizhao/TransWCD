# modules/SA_module.py
# -*- coding: utf-8 -*-
"""
Scene-Adaptive (SA) module:
- SAPredictor: 轻量像素解码器（膨胀卷积 + Dropout2d + 1x1 分类）
- L_pixel: 像素级 BCE（对 logits 使用 BCEWithLogitsLoss）
- L_sg: Scene-Gated 约束（仅未变场景，对误检触发 hinge 惩罚，top-k 近似）
- SAModule: 上述组件的一体封装（前向返回 logits、各损失与总和）
- build_sg_constraint_from_cfg: 从 cfg 构建模块的便捷工厂
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

__all__ = [
    "SAPredictor",
    "SAModule",
    "scene_gated_constraint",
    "build_sg_constraint_from_cfg",
]

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _ensure_single_channel(logits: torch.Tensor) -> torch.Tensor:
    """
    统一为单通道 logits (B,1,H,W)。若 C>=2，仅取第1通道作为“变”通道。
    """
    return logits[:, :1] if logits.size(1) > 1 else logits


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def sa_pixel_loss(
    logits_sa: torch.Tensor,
    target_probs: torch.Tensor,
    pos_weight: Optional[float] = None
) -> torch.Tensor:
    """
    L_pixel：像素级 BCE（对 logits 使用 BCEWithLogitsLoss）
    logits_sa   : (B,1|2,h,w)
    target_probs: (B,1,h,w) ∈ [0,1]（未变场景=全0；变场景=主干 p_change）
    返回：标量
    """
    logits_sa = _ensure_single_channel(logits_sa)   # (B,1,h,w)
    t = target_probs.clamp(1e-4, 1 - 1e-4)         # 数值稳定
    bce = nn.BCEWithLogitsLoss(
        pos_weight=(torch.tensor([pos_weight], device=logits_sa.device)
                    if pos_weight is not None else None)
    )
    return bce(logits_sa, t)


def scene_gated_constraint(
    p_change: torch.Tensor,
    cls_labels: torch.Tensor,
    tau: float = 0.6,
    topk_ratio: float = 0.02
) -> torch.Tensor:
    """
    L_sg：只在未变化场景 y=0 时、对误检触发（1-δ(·) 的可微近似）
    p_change : (B,1,H,W) 像素变化概率（来自 CAM / 主干像素概率）
    cls_labels: (B,) 或 (B,1)；0=未变，1=有变
    tau      : 触发阈值
    topk_ratio: 取前 top-k 像素的比例（默认约2%）
    返回：标量
    """
    B, _, H, W = p_change.shape
    y = cls_labels.view(-1).float()               # (B,)
    uc_mask = (1.0 - y)                           # 仅未变场景有效（y=0）

    # Top-k pooling：对小面积误检更敏感
    k = max(1, int(round(topk_ratio * H * W)))
    flat = p_change.flatten(2)                    # (B,1,HW)
    topk_vals, _ = flat.topk(k, dim=2)           # (B,1,k)
    topk_mean = topk_vals.mean(dim=2).squeeze(1) # (B,)

    sg_per = torch.relu(topk_mean - tau) * uc_mask
    denom = (uc_mask > 0).float().sum().clamp_min(1.0)  # 仅对未变样本平均
    return sg_per.sum() / denom


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class SAPredictor(nn.Module):
    """
    轻量 SA 头 + Dropout2d：几层膨胀卷积 + 1x1 分类
    - in_ch:  输入特征通道（建议取主干的 embedding_dim，如 256）
    - mid:    中间通道
    - rates:  空洞率序列
    - out_ch: 输出通道(推荐 1：直接用 BCEWithLogitsLoss 对 logits 监督)
    - p_drop: Dropout2d 概率
    - drop_after: 'relu' 或 'conv'（dropout放置位置，推荐 'relu'）
    """
    def __init__(
        self,
        in_ch: int,
        mid: int = 128,
        rates: Tuple[int, ...] = (1, 2, 4, 8),
        out_ch: int = 1,
        p_drop: float = 0.1,
        drop_after: str = 'relu'
    ):
        super().__init__()
        assert drop_after in ('relu', 'conv'), f"drop_after must be 'relu' or 'conv', got {drop_after}"
        assert 0.0 <= p_drop < 1.0, f"p_drop must be in [0,1), got {p_drop}"
        self.drop_after = drop_after

        blocks, c = [], in_ch
        for r in rates:
            conv = nn.Conv2d(c, mid, 3, padding=r, dilation=r, bias=False)
            # 小数据/小 batch 更稳：GroupNorm 代替 BatchNorm
            gn   = nn.GroupNorm(num_groups=min(32, mid), num_channels=mid)
            relu = nn.ReLU(inplace=True)
            drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

            if drop_after == 'conv':   # conv -> dropout -> gn -> relu
                blocks += [conv, drop, gn, relu]
            else:                      # conv -> gn -> relu -> dropout（推荐）
                blocks += [conv, gn, relu, drop]
            c = mid

        self.body = nn.Sequential(*blocks)
        self.cls  = nn.Conv2d(mid, out_ch, 1, bias=True)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B,Cf,h,w)
        return: (logits_sa, feat_mid)
        """
        x = self.body(feat)            # (B,mid,h,w)
        logits = self.cls(x)           # (B,out_ch,h,w) —— 返回 logits（不做激活）
        return logits, x


# -----------------------------------------------------------------------------
# Integrated SA module: decoder + losses
# -----------------------------------------------------------------------------

class SAModule(nn.Module):
    """
    一体化 SA 模块：含解码器（SAPredictor）与 L_pixel / L_sg
    forward 返回：
      {
        "logits": logits_sa,      # (B,1,h,w)
        "feat": feat_mid,         # (B,mid,h,w)
        "L_pixel": L_pixel,
        "L_sg": L_sg,
        "L_sa_total": lambda_sa*L_pixel + lambda_sg*L_sg
      }
    """
    def __init__(
        self,
        in_ch: int,
        mid: int = 128,
        rates: Tuple[int, ...] = (1, 2, 4, 8),
        out_ch: int = 1,
        p_drop: float = 0.1,
        tau: float = 0.6,
        topk_ratio: float = 0.02,
        lambda_sa: float = 0.1,
        lambda_sg: float = 1.0,
        drop_after: str = 'relu'
    ):
        super().__init__()
        assert 0.0 <= p_drop < 1.0, f"p_drop out of range: {p_drop}"
        assert drop_after in ("relu", "conv"), f"drop_after must be 'relu' or 'conv', got {drop_after}"

        self.pred = SAPredictor(
            in_ch=in_ch,
            mid=mid,
            rates=rates,
            out_ch=out_ch,
            p_drop=p_drop,         # ✅ 正确传参
            drop_after=drop_after  # ✅ 正确传参
        )
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.lambda_sa = lambda_sa
        self.lambda_sg = lambda_sg

    @torch.no_grad()
    def _build_supervision(self, p_change: torch.Tensor, cls_labels: torch.Tensor) -> torch.Tensor:
        """
        根据场景标签构造像素监督 S：未变=0，变=主干概率；不回传梯度
        p_change  : (B,1,H,W) 概率
        cls_labels: (B,) or (B,1)
        return    : (B,1,H,W)
        """
        y = cls_labels.view(-1).float().view(-1, 1, 1, 1).to(p_change.device)
        yuc = (1.0 - y)                           # 未变
        yc  = y                                   # 变
        S = yuc * torch.zeros_like(p_change) + yc * p_change
        return S

    def forward(
        self,
        feat_sa: torch.Tensor,
        cls_labels: torch.Tensor,
        p_change: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None
    ):
        """
        feat_sa   : (B,Cf,h,w) —— 主干返回的空间特征
        cls_labels: (B,) or (B,1) —— 场景级标签
        p_change  : (B,1,H,W) —— 主干像素概率（用于生成 S）；若 S 已给，可以不传
        S         : (B,1,H,W) —— 直接提供的像素监督；若 None，则用 p_change 和 cls_labels 生成
        pos_weight: BCEWithLogits 的正样本权重（可选，处理类不平衡）
        """
        logits_sa, feat_mid = self.pred(feat_sa)     # (B,1|2,h,w)

        # 生成/对齐监督 S
        if S is None:
            assert p_change is not None, "SAModule: either S or p_change must be provided."
            with torch.no_grad():  # 避免梯度泄漏回 CAM/主干路径
                S = self._build_supervision(p_change.detach(), cls_labels)

        S_sa = F.interpolate(S, size=logits_sa.shape[-2:], mode="bilinear", align_corners=False)

        # L_pixel（对 logits 使用 BCEWithLogitsLoss）
        L_pixel = sa_pixel_loss(logits_sa, S_sa, pos_weight=pos_weight)

        # L_sg（若提供了 p_change 则计算；否则视为 0）
        if p_change is not None:
            L_sg = scene_gated_constraint(p_change.detach(), cls_labels, tau=self.tau, topk_ratio=self.topk_ratio)
        else:
            L_sg = torch.zeros((), device=feat_sa.device)

        L_total = self.lambda_sa * L_pixel + self.lambda_sg * L_sg

        return {
            "logits": logits_sa,
            "feat": feat_mid,
            "L_pixel": L_pixel,
            "L_sg": L_sg,
            "L_sa_total": L_total
        }


# -----------------------------------------------------------------------------
# Factory from cfg
# -----------------------------------------------------------------------------

def build_sg_constraint_from_cfg(cfg, in_ch: int) -> SAModule:
    """
    根据 cfg 构建包含 Scene-Gated Constraint（L_sg）与 L_pixel 的 SA 模块。
    注意：仍返回 SAModule（内部集成 SA 头 + L_pixel + L_sg）。
    """
    sa_cfg = getattr(cfg, "sa", None) or {}
    try:
        if isinstance(sa_cfg, DictConfig):
            sa_cfg = OmegaConf.to_container(sa_cfg, resolve=True) or {}
    except Exception:
        pass

    return SAModule(
        in_ch=in_ch,
        mid=sa_cfg.get("mid", 128),
        rates=tuple(sa_cfg.get("rates", (1, 2, 4, 8))),
        out_ch=sa_cfg.get("out_ch", 1),
        p_drop=sa_cfg.get("p_drop", 0.1),         # 更稳的默认
        tau=sa_cfg.get("tau", 0.6),               # 放宽阈值
        topk_ratio=sa_cfg.get("topk_ratio", 0.02),
        lambda_sa=sa_cfg.get("lambda_sa", 0.1),   # 与类默认一致
        lambda_sg=sa_cfg.get("lambda_sg", 1.0),
        drop_after=sa_cfg.get("drop_after", "relu"),
    )
