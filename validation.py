# from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
# import matplotlib.pyplot as plt
# import os
# import time
# import torch
# def visualize_saliency_effect(visual, saliency_map, step=0, video_id="unknown", save_root="vis_results"):
#     seq_idx = 0
#     batch_idx = 0
#     depth_idx = step

#     original_frame = visual[seq_idx, batch_idx, :, depth_idx].cpu()
#     sal_map = saliency_map[seq_idx, batch_idx, 0, depth_idx].cpu()
#     applied = original_frame * sal_map

#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#     axs[0].imshow(original_frame.permute(1, 2, 0).numpy())
#     axs[0].set_title(f"Video: {video_id}\nOriginal Frame")

#     axs[1].imshow(sal_map.numpy(), cmap='gray')
#     axs[1].set_title("Saliency Map")

#     axs[2].imshow(applied.permute(1, 2, 0).numpy())
#     axs[2].set_title("Applied (Visual * Saliency)")

#     for ax in axs:
#         ax.axis('off')
#     plt.tight_layout()

#     os.makedirs(save_root, exist_ok=True)
#     save_path = os.path.join(save_root, f"{video_id}_frame{step:03d}.png")
#     plt.savefig(save_path)
#     plt.close(fig)


# def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
#     print("# ---------------------------------------------------------------------- #")
#     print('Validation at epoch {}'.format(epoch))
#     model.eval()

#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     accuracies = AverageMeter()

#     end_time = time.time()

#     for i, data_item in enumerate(data_loader):
#         visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
#         print("📦 [Val] saliency_map shape:", saliency_map.shape)

#         if i == 0:
#             video_id = visualization_item[0][0]  # 예: "vid001"
#             visualize_saliency_effect(visual, saliency_map, step=5, video_id=video_id)
            
#         data_time.update(time.time() - end_time)
#         with torch.no_grad():
#             output, loss = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)

#         acc = calculate_accuracy(output, target)

#         losses.update(loss.item(), batch_size)
#         accuracies.update(acc, batch_size)
#         batch_time.update(time.time() - end_time)
#         end_time = time.time()

#     writer.add_scalar('val/loss', losses.avg, epoch)
#     writer.add_scalar('val/acc', accuracies.avg, epoch)
#     print("Val loss: {:.4f}".format(losses.avg))
#     print("Val acc: {:.4f}".format(accuracies.avg))

#     save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
#     states = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     torch.save(states, save_file_path)
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
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

def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
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

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        # print("📦 [Val] saliency_map shape:", saliency_map.shape)

        if i == 0:
            video_id = visualization_item[0][0]  # 예: "vid001"
            visualize_saliency_effect(visual, saliency_map, step=5, video_id=video_id)
            
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        # ✅ 추가: 예측/정답 누적 (argmax로 클래스 예측)
        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    val_loss = losses.avg
    val_acc = accuracies.avg

    # ✅ 에폭 끝에서 macro-F1 계산
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    num_classes = getattr(opt, 'n_classes', output.size(1))  # 안전하게 추출
    val_macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/acc', val_acc, epoch)
    writer.add_scalar('val/macro_f1', val_macro_f1, epoch)   # ✅ 추가

    print("Val loss: {:.4f}".format(val_loss))
    print("Val acc: {:.4f}".format(val_acc))

    # ✅ Best Accuracy 갱신 시 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_file_path = os.path.join(opt.ckpt_path, f'save_{epoch}_{val_acc:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-acc model saved: {save_file_path}")
        
    # ✅ Best Loss 갱신 시 저장
    elif val_loss < best_val_loss:
        best_val_loss = val_loss
        save_file_path = os.path.join(opt.ckpt_path, f'save_{epoch}_{val_loss:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-loss model saved: {save_file_path}")
    
    # ✅ Best Macro-F1 기준 저장 (새로 추가)
    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        save_file_path = os.path.join(opt.ckpt_path, f'save_f1_ep{epoch}_{val_macro_f1:.4f}.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metric': 'macro_f1',
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1,
        }
        torch.save(states, save_file_path)
        print(f"✅ New best-macroF1 model saved: {save_file_path}")

