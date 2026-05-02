#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tsn_cache.py

VE8Dataset + TSN(Temporal Segment Network)에서 랜덤으로 선택되는
frame indices(snippets_frame_idx)를 training/validation 전체에 대해 1회 생성하여 .pt로 저장.

- 프레임(jpg) 자체는 로드하지 않고, dataset.data만 순회하면서 frame index만 계산/저장함.
- TSN은 python random을 사용하므로(TemporalRandomCrop), 재현을 원하면 환경변수 TSN_CACHE_SEED를 설정하세요.
  예) TSN_CACHE_SEED=0 python build_tsn_cache.py --expr_name cache_build
"""

import os
import time
import random
from typing import Dict, Any, List, Optional

import torch

from opts import parse_opts
from transforms.temporal import TSN
from datasets.ve8 import VE8Dataset


def _try_local2global_path(opt):
    """
    main_feature_saliency.py에서 쓰는 local2global_path(opt)를 그대로 쓰되,
    import가 안되면 최소한의 fallback join을 수행.
    """
    try:
        from core.utils import local2global_path
        local2global_path(opt)
        return
    except Exception:
        root = getattr(opt, "root_path", "") or ""

        def _join_if_relative(p):
            if p is None:
                return p
            if os.path.isabs(p):
                return p
            return os.path.join(root, p)

        opt.video_path = _join_if_relative(opt.video_path)
        opt.audio_path = _join_if_relative(opt.audio_path)
        opt.annotation_path = _join_if_relative(opt.annotation_path)
        opt.saliency_path = _join_if_relative(opt.saliency_path)
        opt.result_path = _join_if_relative(opt.result_path)


def _get_seed_from_env() -> Optional[int]:
    v = os.environ.get("TSN_CACHE_SEED", "").strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        raise ValueError("TSN_CACHE_SEED must be int, got: {}".format(v))


def _video_key_from_sample(sample: Dict[str, Any], video_root: str) -> str:
    """
    sample['video']는 .../<video_root>/<label>/<key> 형태의 디렉토리 경로.
    video_root 기준 상대경로를 key로 쓰면 label/key 형태가 되어 충돌 가능성이 낮음.
    """
    video_dir = sample["video"]
    return os.path.relpath(video_dir, video_root).replace("\\", "/")


def build_cache_for_subset(
    opt,
    subset: str,
    center: bool,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    subset: 'training' or 'validation'
    center: TSN center crop 여부(기본 False로 baseline과 동일)
    seed: 재현을 원하면 지정. subset별로 seed+offset을 적용해도 됨.
    """
    if seed is not None:
        offset = 0 if subset == "training" else 1
        random.seed(seed + offset)

    temporal_transform = TSN(
        seq_len=opt.seq_len,
        snippet_duration=opt.snippet_duration,
        center=center,
    )

    # 캐시 생성에서는 이미지/오디오를 실제로 로드할 필요가 없으므로 transform은 None으로 둬도 됨.
    ds = VE8Dataset(
        video_path=opt.video_path,
        audio_path=opt.audio_path,
        annotation_path=opt.annotation_path,
        saliency_path=opt.saliency_path,
        subset=subset,
        fps=opt.fps,
        spatial_transform=None,
        temporal_transform=temporal_transform,
        target_transform=None,
        saliency_transform=None,
        need_audio=False,
    )

    index: Dict[str, List[List[int]]] = {}
    skipped = 0

    for sample in ds.data:
        key = _video_key_from_sample(sample, opt.video_path)
        frame_indices = sample["frame_indices"]

        try:
            snippets_frame_idx = temporal_transform(frame_indices)
        except Exception as e:
            skipped += 1
            print("[WARN] skip video={} reason={}: {}".format(key, type(e).__name__, e))
            continue

        index[key] = snippets_frame_idx

    cache_obj = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subset": subset,
            "fps": opt.fps,
            "seq_len": opt.seq_len,
            "snippet_duration": opt.snippet_duration,
            "center": center,
            "seed": seed,
            "n_items_total": len(ds.data),
            "n_items_cached": len(index),
            "n_items_skipped": skipped,
            "annotation_path": opt.annotation_path,
            "video_path": opt.video_path,
            "saliency_path": opt.saliency_path,
        },
        "index": index,
    }
    return cache_obj


def main():
    opt = parse_opts()
    _try_local2global_path(opt)

    seed = _get_seed_from_env()
    center = False  # baseline과 동일하게 유지

    expr = (getattr(opt, "expr_name", "") or "").strip() or "tsn_cache"
    out_dir = os.path.join(opt.result_path, expr, "tsn_cache")
    os.makedirs(out_dir, exist_ok=True)

    for subset in ["training", "validation"]:
        print("\n[INFO] Building TSN cache for subset={} (center={}, seed={})".format(subset, center, seed))
        cache_obj = build_cache_for_subset(opt, subset=subset, center=center, seed=seed)

        out_name = (
            "tsn_cache_{subset}_fps{fps}_t{t}_k{k}_center{center}{seed}.pt".format(
                subset=subset,
                fps=opt.fps,
                t=opt.seq_len,
                k=opt.snippet_duration,
                center=int(center),
                seed="" if seed is None else "_seed{}".format(seed),
            )
        )
        out_path = os.path.join(out_dir, out_name)

        torch.save(cache_obj, out_path)
        print("[OK] saved: {}".format(out_path))
        print("     cached={}  skipped={}".format(cache_obj["meta"]["n_items_cached"], cache_obj["meta"]["n_items_skipped"]))

        some_key = next(iter(cache_obj["index"].keys()), None)
        if some_key is not None:
            print("     example key: {}".format(some_key))
            print("     example snippets[0]: {}".format(cache_obj["index"][some_key][0]))

    print("\n[DONE] TSN cache build complete.")


if __name__ == "__main__":
    main()
