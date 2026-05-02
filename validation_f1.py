from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
import time
import torch

def _metrics_from_confusion(cm: torch.Tensor):
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    macro_f1     = f1.mean().item()
    balanced_acc = recall.mean().item()
    return macro_f1, balanced_acc

@torch.no_grad()
def val_epoch(epoch, data_loader, model, criterion, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    all_preds = []
    all_targets = []
    num_classes = None

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)

        data_time.update(time.time() - end_time)
        output, loss = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)

        if num_classes is None:
            num_classes = output.size(1)

        preds = output.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    val_loss = losses.avg
    val_acc  = accuracies.avg

    y_pred = torch.cat(all_preds)   # [N] on CPU
    y_true = torch.cat(all_targets) # [N]

    # 혼동행렬: 한 번만 계산 (벡터화)
    cm = torch.bincount(
        (y_true * num_classes + y_pred).to(torch.int64),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    macro_f1, balanced_acc = _metrics_from_confusion(cm)

    print(f"Val loss             : {val_loss:.4f}")
    print(f"Val acc (accuracy)   : {val_acc:.4f}")        # 0~1 비율
    print(f"Val macro F1         : {macro_f1:.4f}")
    print(f"Val balanced acc(UAR): {balanced_acc:.4f}")
