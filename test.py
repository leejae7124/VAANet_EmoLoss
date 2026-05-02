from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
import matplotlib.pyplot as plt
import os
import time
import torch
import cv2
import numpy as np
# --- CAM 관련 라이브러리 추가 ---
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
def visualize_saliency_effect(visual, saliency_map, step=0, video_id="unknown", save_root="test_vis_results_"):
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

def test_epoch(epoch, data_loader, model, criterion, opt, writer, cam, cam_save_dir):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, saliency_map, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        
        if i == 0:
            video_id = visualization_item[0][0]  # 예: "vid001"
            visualize_saliency_effect(visual, saliency_map, step=5, video_id=video_id)
        
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target, audio, saliency_map], model, criterion, i)
        for sample_idx in range(batch_size):
            # 첫 번째 샘플이 아닌, 현재 인덱스의 샘플 선택
            video_sample = visual[sample_idx]
            audio_sample = audio[sample_idx]
            predicted_class = torch.argmax(output, dim=1)[sample_idx].item()
            
            # 시퀀스의 모든 클립을 순회
            for clip_idx in range(opt.seq_len):
                clip_tensor_5d = video_sample[clip_idx].unsqueeze(0)
                # print("clip tensor: ", clip_tensor.shape)
                # audio_clip_tensor = audio_sample[clip_idx].unsqueeze(0)
                
                 # ✅ saliency도 현재 sample/clip 기준으로 5D 모양으로 준비
                # saliency_map: [B, Seq, 1, T, H, W]
                sal_clip_5d = saliency_map[sample_idx, clip_idx]       # [1, T, H, W]
                sal_clip_5d = sal_clip_5d.unsqueeze(0)                 # [1, 1, T, H, W]
                sal_clip_5d = sal_clip_5d.to(clip_tensor_5d.device, dtype=clip_tensor_5d.dtype)
                print("clip_tensor_5d:", clip_tensor_5d.shape)  # [1, C, T, H, W]
                print("sal_clip_5d   :", sal_clip_5d.shape)     # [1, 1, T, H, W]
                print(
                        f"[Saliency] min={sal_clip_5d.min().item():.4f}, "
                        f"max={sal_clip_5d.max().item():.4f}, "
                        f"mean={sal_clip_5d.mean().item():.4f}"
                    )


                
                # targets = [ClassifierOutputTarget(predicted_class)] #cam을 계산할 대상 클래스 지정. 예측 클래스를 타겟으로!
                # print("target size: ", targets)
                # target_size = (opt.sample_size, opt.sample_size) # (112, 112)
                # test.py (루프 내부 CAM 구간 교체)
                # clip_tensor_5d = video_sample[clip_idx].unsqueeze(0)      # [1, C, T, H, W]
                if audio_sample.dim() == 2:
                    audio_sample = audio_sample.unsqueeze(0)
                cam.model.set_audio(audio_sample)
                cam.model.set_saliency(sal_clip_5d)       # ★ saliency도 세팅# B=1 가정
                
                targets = [ClassifierOutputTarget(predicted_class)]
                
                # 클립 전체로 한 번에 CAM 계산
                grayscale_cam_clip = cam(input_tensor=clip_tensor_5d, targets=targets)  # [1, Hc, Wc]
                grayscale_cam_2d = grayscale_cam_clip[0, :]
                
                # 클립의 모든 프레임에 같은 CAM 오버레이
                for frame_idx in range(clip_tensor_5d.size(2)):  # T 프레임
                    frame = clip_tensor_5d[0, :, frame_idx, :, :].cpu().numpy()
                    frame = np.transpose(frame, (1, 2, 0))
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
                
                    vis = show_cam_on_image(frame, grayscale_cam_2d, use_rgb=True)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                    unique_sample_id = i * opt.batch_size + sample_idx
                    save_path = os.path.join(
                        cam_save_dir,
                        f"sample_{unique_sample_id:04d}_clip_{clip_idx:02d}_frame_{frame_idx:02d}_pred_{predicted_class}.png"
                    )
                    cv2.imwrite(save_path, np.uint8(vis_bgr))

        with torch.no_grad():
            acc = calculate_accuracy(output, target)
    
            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        if (i+1) % 10 == 0:
            print(f"Processed batch {i+1}/{len(data_loader)}")
        # --- [추천 위치] ---
        # 현재 배치의 모든 작업이 끝났고, 다음 배치를 불러오기 직전이므로
        # 여기서 캐시를 비워주는 것이 가장 효과적입니다.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    

    # save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
    # states = {
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # torch.save(states, save_file_path)
