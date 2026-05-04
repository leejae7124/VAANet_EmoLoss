import os
import json
import functools
import librosa
import numpy as np
import torch
import torch.utils.data as data

from torchvision import get_image_backend
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_saliency_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    return pil_loader


def get_default_saliency_image_loader():
    return pil_saliency_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, "{:06d}.jpg".format(i))
        assert os.path.exists(image_path), f"image does not exist: {image_path}"
        video.append(image_loader(image_path))
    return video


def saliency_loader(saliency_dir_path, frame_indices, saliency_image_loader):
    video = []
    for i in frame_indices:
        saliency_path = os.path.join(saliency_dir_path, "{:06d}.jpg".format(i))
        assert os.path.exists(saliency_path), f"saliency does not exist: {saliency_path}"
        video.append(saliency_image_loader(saliency_path))
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_default_saliency_loader():
    saliency_image_loader = get_default_saliency_image_loader()
    return functools.partial(saliency_loader, saliency_image_loader=saliency_image_loader)


def load_annotation_data(annotation_file_path):
    with open(annotation_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs


def normalize_label(x):
    return str(x).strip().lower()


def get_class_labels(frame_root_path):
    """
    frame_root_path:
      /EmoVidMovie_frames_split/
        train/
        validation/
        test/
    """
    class_names = set()

    for subset in ["train", "validation", "test"]:
        subset_dir = os.path.join(frame_root_path, subset)
        if not os.path.isdir(subset_dir):
            continue

        for class_name in os.listdir(subset_dir):
            class_dir = os.path.join(subset_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.add(class_name)

    class_names = sorted(list(class_names))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def make_dataset(frame_root_path,
                 audio_root_path,
                 annotation_root_path,
                 saliency_root_path,
                 subset,
                 fps=30,
                 need_audio=True,
                 need_saliency=True):
    class_to_idx, idx_to_class = get_class_labels(frame_root_path)

    subset_frame_root = os.path.join(frame_root_path, subset)
    subset_audio_root = os.path.join(audio_root_path, subset) if audio_root_path is not None else None
    subset_annotation_root = os.path.join(annotation_root_path, subset)
    subset_saliency_root = os.path.join(saliency_root_path, subset) if saliency_root_path is not None else None

    assert os.path.isdir(subset_frame_root), f"frame subset dir does not exist: {subset_frame_root}"
    assert os.path.isdir(subset_annotation_root), f"annotation subset dir does not exist: {subset_annotation_root}"
    if need_audio:
        assert os.path.isdir(subset_audio_root), f"audio subset dir does not exist: {subset_audio_root}"
    if need_saliency:
        assert os.path.isdir(subset_saliency_root), f"saliency subset dir does not exist: {subset_saliency_root}"

    dataset = []
    ORIGINAL_FPS = 30
    step = max(1, ORIGINAL_FPS // fps)

    for class_name in sorted(os.listdir(subset_frame_root)):
        class_dir = os.path.join(subset_frame_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for clip_id in sorted(os.listdir(class_dir)):
            frame_dir = os.path.join(class_dir, clip_id)
            if not os.path.isdir(frame_dir):
                continue

            audio_path = os.path.join(subset_audio_root, class_name, f"{clip_id}.mp3") if need_audio else None
            annotation_path = os.path.join(subset_annotation_root, class_name, f"{clip_id}.json")
            saliency_dir = os.path.join(subset_saliency_root, class_name, clip_id) if need_saliency else None

            assert os.path.exists(annotation_path), f"annotation does not exist: {annotation_path}"
            if need_audio:
                assert os.path.exists(audio_path), f"audio does not exist: {audio_path}"
            if need_saliency:
                assert os.path.isdir(saliency_dir), f"saliency dir does not exist: {saliency_dir}"

            meta = load_annotation_data(annotation_path)

            # annotation의 emotion과 폴더명이 다르면 경고만 출력
            anno_label = normalize_label(meta.get("emotion", class_name))
            if anno_label != normalize_label(class_name):
                print(f"[warn] label mismatch: folder={class_name}, annotation={anno_label}, clip={clip_id}")

            n_frames_file_path = os.path.join(frame_dir, "n_frames")
            if os.path.exists(n_frames_file_path):
                with open(n_frames_file_path, "r") as f:
                    n_frames = int(f.read().strip())
            else:
                jpgs = [x for x in os.listdir(frame_dir) if x.endswith(".jpg")]
                n_frames = len(jpgs)

            if n_frames <= 0:
                continue

            sample = {
                "video": frame_dir,
                "audio": audio_path,
                "annotation": annotation_path,
                "saliency": saliency_dir,
                "segment": [1, n_frames],
                "n_frames": n_frames,
                "video_id": clip_id,
                "source_video_id": meta.get("video_id", clip_id),
                "label_name": class_name,
                "label": class_to_idx[class_name],
                "frame_indices": list(range(1, n_frames + 1, step)),
                "duration": meta.get("duration", None),
                "caption": meta.get("caption", None),
                "brightness": meta.get("brightness", None),
                "colorfulness": meta.get("colorfulness", None),
                "hue": meta.get("hue", None),
            }
            dataset.append(sample)

    print(f"[{subset}] total samples: {len(dataset)}")
    print("class_to_idx:", class_to_idx)

    return dataset, idx_to_class


class _BaseEmoVidDataset(data.Dataset):
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 saliency_path=None,
                 subset="train",
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 saliency_transform=None,
                 get_loader=get_default_video_loader,
                 get_saliency_loader=get_default_saliency_loader,
                 need_audio=True,
                 need_saliency=True):
        self.data, self.class_names = make_dataset(
            frame_root_path=video_path,
            audio_root_path=audio_path,
            annotation_root_path=annotation_path,
            saliency_root_path=saliency_path,
            subset=subset,
            fps=fps,
            need_audio=need_audio,
            need_saliency=need_saliency
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.saliency_transform = saliency_transform
        self.loader = get_loader()
        self.saliency_loader = get_saliency_loader() if need_saliency else None
        self.fps = fps
        self.need_audio = need_audio
        self.need_saliency = need_saliency

    def _load_audio(self, data_item):
        if not self.need_audio:
            return []

        timeseries_length = 4096
        audio_path = data_item["audio"]
        feature = preprocess_audio(audio_path).T
        k = timeseries_length // feature.shape[0] + 1
        feature = np.tile(feature, reps=(k, 1))
        audios = feature[:timeseries_length, :]
        audios = torch.FloatTensor(audios)
        return audios

    def __getitem__(self, index):
        data_item = self.data[index]

        video_path = data_item["video"]
        saliency_path = data_item["saliency"]
        frame_indices = data_item["frame_indices"]

        if self.temporal_transform is not None:
            snippets_frame_idx = self.temporal_transform(frame_indices)
        else:
            snippets_frame_idx = [frame_indices]

        audios = self._load_audio(data_item)

        snippets = []
        saliency_snippets = []

        if self.spatial_transform is not None and hasattr(self.spatial_transform, "randomize_parameters"):
            self.spatial_transform.randomize_parameters()

        if self.need_saliency and self.saliency_transform is not None and hasattr(self.saliency_transform, "randomize_parameters"):
            self.saliency_transform.randomize_parameters()

        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            if self.spatial_transform is not None:
                snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets.append(snippet)

            if self.need_saliency:
                saliency_snippet = self.saliency_loader(saliency_path, snippet_frame_idx)
                if self.saliency_transform is not None:
                    saliency_snippet = [self.saliency_transform(img) for img in saliency_snippet]
                saliency_snippet = torch.stack(saliency_snippet, 0).permute(1, 0, 2, 3)
                saliency_snippets.append(saliency_snippet)

        snippets = torch.stack(snippets, 0)
        target = self.target_transform(data_item) if self.target_transform is not None else data_item["label"]
        visualization_item = [data_item["video_id"]]

        if self.need_saliency:
            saliency_snippets = torch.stack(saliency_snippets, 0)
            return snippets, saliency_snippets, target, audios, visualization_item

        return snippets, target, audios, visualization_item

    def __len__(self):
        return len(self.data)


class EmoVidDataset(_BaseEmoVidDataset):
    """
    VAANet + Saliency 용
    return:
      snippets, saliency_snippets, target, audios, visualization_item
    """
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 saliency_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 saliency_transform=None,
                 get_loader=get_default_video_loader,
                 get_saliency_loader=get_default_saliency_loader,
                 need_audio=True):
        super().__init__(
            video_path=video_path,
            audio_path=audio_path,
            annotation_path=annotation_path,
            saliency_path=saliency_path,
            subset=subset,
            fps=fps,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            saliency_transform=saliency_transform,
            get_loader=get_loader,
            get_saliency_loader=get_saliency_loader,
            need_audio=need_audio,
            need_saliency=True
        )


class EmoVidBaselineDataset(_BaseEmoVidDataset):
    """
    VAANet baseline 용
    return:
      snippets, target, audios, visualization_item
    """
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 need_audio=True):
        super().__init__(
            video_path=video_path,
            audio_path=audio_path,
            annotation_path=annotation_path,
            saliency_path=None,
            subset=subset,
            fps=fps,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            saliency_transform=None,
            get_loader=get_loader,
            get_saliency_loader=get_default_saliency_loader,
            need_audio=need_audio,
            need_saliency=False
        )