def test_epoch(epoch, data_loader, model, criterion, opt, writer, cam, cam_save_dir, wrapped_model):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    # --- meters (GT / Pred 각각 deletion & insertion)
    class M:
        def __init__(self): self.del_auc, self.ins_auc = AUCMeter(), AUCMeter()
    meters = {"gt": M(), "pred": M()}

    steps = torch.linspace(0, 1, 21, device=opt.device)  # 0~1, 5% 간격
    baseline_val = 0.5

    # --- 시각화 & 평균 곡선 누적 준비
    from pathlib import Path
    save_root = Path(cam_save_dir) / f"auc_epoch_{epoch:04d}"
    save_root.mkdir(parents=True, exist_ok=True)
    p_vals = steps.detach().cpu().numpy()

    def zeros_like_p(): return np.zeros_like(p_vals, dtype=np.float64)
    mean_curve_sum = {
        "gt": {"del": zeros_like_p(), "ins": zeros_like_p()},
        "pred": {"del": zeros_like_p(), "ins": zeros_like_p()}
    }
    n_curves = {"gt": 0, "pred": 0}
    SAVE_PER_CLIP = getattr(opt, "save_auc_per_clip", False)

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)

        with torch.no_grad():
            output, loss = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)

        # 배치·시퀀스 축 정리
        if visual.size(0) == opt.seq_len:   # [seq, batch, C,T,H,W]
            seq_first, bs = True, visual.size(1)
        else:                               # [batch, seq, C,T,H,W]
            seq_first, bs = False, visual.size(0)

        for sample_idx in range(bs):
            # 샘플별 시퀀스/오디오/라벨
            video_sample = (visual[:, sample_idx] if seq_first else visual[sample_idx])       # [seq_len, C,T,H,W]
            audio_sample = audio[sample_idx].unsqueeze(0)
            # 시퀀스 예측 클래스(현 방식 유지)
            seq_pred_class = int(output.argmax(1)[sample_idx].item())
            # GT 클래스(one-hot/정수 모두 대응)
            if target.dim() >= 2 and target.size(-1) > 1:
                gt_class = int(target[sample_idx].argmax(dim=-1).item())
            else:
                gt_class = int(target[sample_idx].item())

            # (원하면 클립 예측 기준으로 바꾸기: 아래 한 줄로 seq_pred_class를 대체)
            # with torch.no_grad(): seq_pred_class = int(wrapped_model(video_sample[0].unsqueeze(0)).argmax(1).item())

            for clip_idx in range(opt.seq_len):
                # --- 입력 구성
                clip_tensor_5d = video_sample[clip_idx].unsqueeze(0)  # [1,C,T,H,W]
                if seq_first:
                    sal_clip_5d = saliency_map[clip_idx, sample_idx]  # [1,T,H,W]
                else:
                    sal_clip_5d = saliency_map[sample_idx, clip_idx]  # [1,T,H,W]
                sal_clip_5d = sal_clip_5d.unsqueeze(0).to(clip_tensor_5d.device, dtype=clip_tensor_5d.dtype)  # [1,1,T,H,W]

                cam.model.set_audio(audio_sample)
                cam.model.set_saliency(sal_clip_5d)

                # --- 공통 마스크 인덱스 준비(두 모드에서 동일)
                # 먼저 하나의 CAM으로 중요도 순서를 얻고 공유해도 되지만,
                # 모드별 CAM이 달라지는 것이 자연스러우므로 각 모드에서 CAM을 따로 만듭니다.
                base_del = torch.full_like(clip_tensor_5d, baseline_val)
                base_ins = strong_blur_5d(clip_tensor_5d, spatial_scale=0.125)

                for mode, cls_idx in (("gt", gt_class), ("pred", seq_pred_class)):
                    # --- CAM & mask
                    targets = [ClassifierOutputTarget(cls_idx)]
                    grayscale_cam_clip = cam(input_tensor=clip_tensor_5d, targets=targets)  # [1,Hc,Wc]
                    grayscale_cam_2d = grayscale_cam_clip[0]
                    mask_5d = cam_to_mask(grayscale_cam_2d, clip_tensor_5d)  # [1,1,T,H,W]

                    # 중요도 순 정렬
                    flat = mask_5d.reshape(-1)
                    _, idx_sorted = torch.sort(flat, descending=True)
                    N = flat.numel()
                    cum_mask_flat = torch.zeros_like(flat)

                    del_scores, ins_scores = [], []
                    with torch.no_grad():
                        for p in steps:
                            K = int(round(p.item() * N))
                            already = int(cum_mask_flat.sum().item())
                            if K > already:
                                cum_mask_flat[idx_sorted[already:K]] = 1.0
                            del_mask = cum_mask_flat.view_as(mask_5d)

                            # deletion
                            x_del = clip_tensor_5d * (1 - del_mask) + base_del * del_mask
                            logits_del = wrapped_model(x_del)
                            prob_del = F.softmax(logits_del, dim=-1)
                            del_scores.append(prob_del[0, cls_idx].item())

                            # insertion
                            x_ins = base_ins * (1 - del_mask) + clip_tensor_5d * del_mask
                            logits_ins = wrapped_model(x_ins)
                            prob_ins = F.softmax(logits_ins, dim=-1)
                            ins_scores.append(prob_ins[0, cls_idx].item())

                    # --- AUC & 평균 곡선 누적
                    del_curve = np.asarray(del_scores, dtype=np.float64)
                    ins_curve = np.asarray(ins_scores, dtype=np.float64)
                    auc_del = np.trapz(del_curve, p_vals)
                    auc_ins = np.trapz(ins_curve, p_vals)

                    meters[mode].del_auc.update(auc_del)
                    meters[mode].ins_auc.update(auc_ins)
                    mean_curve_sum[mode]["del"] += del_curve
                    mean_curve_sum[mode]["ins"] += ins_curve
                    n_curves[mode] += 1

                    # per-clip PNG (옵션)
                    if SAVE_PER_CLIP:
                        fig = plt.figure(figsize=(6, 4), dpi=160)
                        plt.plot(p_vals, del_curve, label=f"Deletion (AUC={auc_del:.4f})")
                        plt.plot(p_vals, ins_curve, label=f"Insertion (AUC={auc_ins:.4f})")
                        plt.fill_between(p_vals, del_curve, alpha=0.15)
                        plt.fill_between(p_vals, ins_curve, alpha=0.15)
                        plt.title(f"[{mode}] Sample {sample_idx}, Clip {clip_idx}, cls={cls_idx}")
                        plt.xlabel("p (top-k ratio)"); plt.ylabel("p(class)")
                        plt.grid(True, alpha=0.3); plt.legend()
                        out_png = save_root / f"{mode}_auc_sample{sample_idx:03d}_clip{clip_idx:03d}.png"
                        fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

                # 로그 (선택)
                print(f"[AUC clip={clip_idx:02d}] "
                      f"GT(del={meters['gt'].del_auc.avg:.4f}, ins={meters['gt'].ins_auc.avg:.4f}) | "
                      f"PRED(del={meters['pred'].del_auc.avg:.4f}, ins={meters['pred'].ins_auc.avg:.4f})")

        if (i + 1) % 5 == 0:
            print(f"-- batch {i+1}/{len(data_loader)} "
                  f"GT(mean_del={meters['gt'].del_auc.avg:.4f}, mean_ins={meters['gt'].ins_auc.avg:.4f}) | "
                  f"PRED(mean_del={meters['pred'].del_auc.avg:.4f}, mean_ins={meters['pred'].ins_auc.avg:.4f})")

    # --- 에폭 결과 요약 출력 & TensorBoard 스칼라 기록
    print(f"[VAL-AUC][GT]   deletion={meters['gt'].del_auc.avg:.4f}  insertion={meters['gt'].ins_auc.avg:.4f}  (N={n_curves['gt']})")
    print(f"[VAL-AUC][PRED] deletion={meters['pred'].del_auc.avg:.4f}  insertion={meters['pred'].ins_auc.avg:.4f}  (N={n_curves['pred']})")

    writer.add_scalar('val_auc_gt/deletion',  meters['gt'].del_auc.avg, epoch)
    writer.add_scalar('val_auc_gt/insertion', meters['gt'].ins_auc.avg, epoch)
    writer.add_scalar('val_auc_pred/deletion',  meters['pred'].del_auc.avg, epoch)
    writer.add_scalar('val_auc_pred/insertion', meters['pred'].ins_auc.avg, epoch)

    # --- 에폭 평균 곡선 시각화 & 저장 & TensorBoard
    for mode in ("gt", "pred"):
        if n_curves[mode] == 0:
            continue
        mean_del = mean_curve_sum[mode]["del"] / n_curves[mode]
        mean_ins = mean_curve_sum[mode]["ins"] / n_curves[mode]
        auc_del_curve = np.trapz(mean_del, p_vals)
        auc_ins_curve = np.trapz(mean_ins, p_vals)

        fig = plt.figure(figsize=(7, 5), dpi=160)
        plt.plot(p_vals, mean_del,
                 label=f"Deletion  ⟨AUC⟩={meters[mode].del_auc.avg:.4f} | curveAUC={auc_del_curve:.4f}")
        plt.plot(p_vals, mean_ins,
                 label=f"Insertion ⟨AUC⟩={meters[mode].ins_auc.avg:.4f} | curveAUC={auc_ins_curve:.4f}")
        plt.fill_between(p_vals, mean_del, alpha=0.15)
        plt.fill_between(p_vals, mean_ins, alpha=0.15)
        plt.title(f"[{mode.upper()}] Mean Deletion/Insertion — epoch {epoch}")
        plt.xlabel("p (top-k ratio)"); plt.ylabel("p(class)")
        plt.grid(True, alpha=0.3); plt.legend()

        writer.add_figure(f'val_auc_{mode}/mean_curves', fig, epoch)
        out_png = save_root / f"{mode}_auc_curves_epoch{epoch:04d}.png"
        fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

        # 원시 데이터 저장
        np.savez(save_root / f"{mode}_auc_curves_epoch{epoch:04d}.npz",
                 p=p_vals, mean_del=mean_del, mean_ins=mean_ins,
                 mean_del_auc_scalar=meters[mode].del_auc.avg, mean_ins_auc_scalar=meters[mode].ins_auc.avg,
                 mean_del_curve_auc=auc_del_curve,   mean_ins_curve_auc=auc_ins_curve)

    # --- (옵션) GT vs Pred 비교 오버레이 한 장
    if n_curves["gt"] > 0 and n_curves["pred"] > 0:
        mean_del_gt = mean_curve_sum["gt"]["del"] / n_curves["gt"]
        mean_ins_gt = mean_curve_sum["gt"]["ins"] / n_curves["gt"]
        mean_del_pr = mean_curve_sum["pred"]["del"] / n_curves["pred"]
        mean_ins_pr = mean_curve_sum["pred"]["ins"] / n_curves["pred"]

        fig = plt.figure(figsize=(7, 5), dpi=160)
        plt.plot(p_vals, mean_del_gt, label="GT Deletion")
        plt.plot(p_vals, mean_ins_gt, label="GT Insertion")
        plt.plot(p_vals, mean_del_pr, label="Pred Deletion")
        plt.plot(p_vals, mean_ins_pr, label="Pred Insertion")
        plt.title(f"GT vs Pred — epoch {epoch}")
        plt.xlabel("p (top-k ratio)"); plt.ylabel("p(class)")
        plt.grid(True, alpha=0.3); plt.legend()
        writer.add_figure('val_auc_compare/gt_vs_pred', fig, epoch)
        fig.savefig(save_root / f"compare_gt_pred_epoch{epoch:04d}.png", bbox_inches='tight')
        plt.close(fig)
