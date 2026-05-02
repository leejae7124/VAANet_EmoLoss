# DAC 용 테스트 (상대적 평가)
from core.utils import AverageMeter, process_data_item, calculate_accuracy
import matplotlib.pyplot as plt
import os
import time
import torch


def visualize_saliency_effect(visual, saliency_map, step=0, video_id="unknown", save_root="vis_results"):
    seq_idx = 0
    batch_idx = 0
    depth_idx = step

    original_frame = visual[seq_idx, batch_idx, :, depth_idx].cpu()
    sal_map = saliency_map[seq_idx, batch_idx, 0, depth_idx].cpu()
    applied = original_frame * sal_map

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_frame.permute(1, 2, 0).numpy())
    axs[0].set_title(f"Video: {video_id}\nOriginal Frame")

    axs[1].imshow(sal_map.numpy(), cmap='gray')
    axs[1].set_title("Saliency Map")

    axs[2].imshow(applied.permute(1, 2, 0).numpy())
    axs[2].set_title("Applied (Visual * Saliency)")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()

    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{video_id}_frame{step:03d}.png")
    plt.savefig(save_path)
    plt.close(fig)

best_val_acc = 0.0
best_val_loss = float("inf")
best_val_macro_f1 = 0.0   # ✅ 추가: 최고 macro-F1



@torch.no_grad()
def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """
    y_true, y_pred: shape [N], int64
    macro-F1 = (1/C) * Σ_c F1_c
    F1_c = 2*TP / (2*TP + FP + FN)  (분모 0이면 0으로 처리)
    """
    # one-pass로 TP/FP/FN 집계
    f1_sum = 0.0
    eps = 1e-12
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        denom = (2*tp + fp + fn)
        f1_c = (2*tp) / (denom + eps) if denom > 0 else 0.0
        f1_sum += f1_c
    return f1_sum / max(num_classes, 1)

#DAC 전용 유틸
def heatmap_to_bbox(hm, thresh_ratio=0.6, min_max=1e-8):
    m = hm.max()
    if m <= min_max:
        return None  # cover 안 함
    thresh = m * thresh_ratio
    ys, xs = torch.where(hm >= thresh)
    if ys.numel() == 0:
        return None
    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1
    return y1, y2, x1, x2

def cover_visual_with_gaussian_noise(visual, cam_map, thresh_ratio=0.6):
    visual_cov = visual.clone()
    B, Seq, C, D, H, W = visual_cov.shape

    for b in range(B):
        for s in range(Seq):
            bbox = heatmap_to_bbox(cam_map[b, s], thresh_ratio)
            if bbox is None:
                continue

            y1, y2, x1, x2 = bbox
            region = visual_cov[b, s, :, :, y1:y2, x1:x2]
            mu = region.mean()
            std = region.std().clamp_min(1e-6)
            noise = torch.randn_like(region) * std + mu
            visual_cov[b, s, :, :, y1:y2, x1:x2] = noise

    return visual_cov

def val_epoch(epoch, data_loader, model, criterion, opt, writer):
    global best_val_acc, best_val_loss, best_val_macro_f1
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # ✅ 추가: 전체 예측/라벨 누적용 리스트
    all_preds = []
    all_targets = []

    all_preds_cov = []
    all_targets_cov = []

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        # print("📦 [Val] saliency_map shape:", saliency_map.shape)

        if i == 0:
            video_id = visualization_item[0][0]  # 예: "vid001"
            visualize_saliency_effect(visual, saliency_map, step=5, video_id=video_id)
            
        data_time.update(time.time() - end_time)
        # Round 1
        with torch.enable_grad():
            output, alpha, beta, gamma, cam_map = model(
                visual, audio, saliency_map,
                target_class=None,
                compute_gradcam=True
            )
            loss = criterion(output, target)

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        # ✅ 추가: 예측/정답 누적 (argmax로 클래스 예측)
        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        # Covering
        visual_cov = cover_visual_with_gaussian_noise(visual, cam_map.detach(), thresh_ratio=0.8)

        # Round 2
        with torch.no_grad():
            output_cov, _, _, _ = model(
                visual_cov, audio, saliency_map,
                compute_gradcam=False
            )
        
        preds_cov = torch.argmax(output_cov, dim=1)
        all_preds_cov.append(preds_cov.detach().cpu())
        all_targets_cov.append(target.detach().cpu())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    val_loss = losses.avg
    val_acc = accuracies.avg

    # ✅ 에폭 끝에서 macro-F1 계산
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    y_pred_cov = torch.cat(all_preds_cov, dim=0)
    y_true_cov = torch.cat(all_targets_cov, dim=0)

    num_classes = getattr(opt, 'n_classes', output.size(1))  # 안전하게 추출
    # F1
    f1_round1 = compute_macro_f1(y_true, y_pred, num_classes)
    f1_round2 = compute_macro_f1(y_true_cov, y_pred_cov, num_classes)
    dac_f1 = f1_round1 - f1_round2

    # Accuracy
    acc_round1 = (y_pred == y_true).float().mean().item()
    acc_round2 = (y_pred_cov == y_true_cov).float().mean().item()
    dac_acc = acc_round1 - acc_round2

    print(f"Round1 macro-F1: {f1_round1:.4f}")
    print(f"Round2 macro-F1: {f1_round2:.4f}")
    print(f"DAC-F1: {dac_f1:.4f}")

    if writer is not None:
        writer.add_scalar('dac/macro_f1_round1', f1_round1, epoch)
        writer.add_scalar('dac/macro_f1_round2', f1_round2, epoch)
        writer.add_scalar('dac/dac_f1', dac_f1, epoch)

        writer.add_scalar('dac/acc_round1', acc_round1, epoch)
        writer.add_scalar('dac/acc_round2', acc_round2, epoch)
        writer.add_scalar('dac/dac_acc', dac_acc, epoch)

    return {
        "f1_round1": f1_round1,
        "f1_round2": f1_round2,
        "dac_f1": dac_f1,
        "acc_round1": acc_round1,
        "acc_round2": acc_round2,
        "dac_acc": dac_acc,
        "loss_round1": losses.avg,
    }

