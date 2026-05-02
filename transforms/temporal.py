import random


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object): #함수 역할: 프레임 인덱스 리스트에서 길이(size)만큼을 시간축으로 랜덤 위치에서 잘라서 반환하는 용도
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size, loop the indices as many times as necessary.
    """

    def __init__(self, size, seed=0):
        self.size = size

    def __call__(self, frame_indices):
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin = random.randint(0, rand_end)
        end = min(begin + self.size, len(frame_indices))
        out = frame_indices[begin:end]
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at the center.
    If the number of frames is less than the size, loop the indices as many times as necessary.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        center_index = len(frame_indices) // 2
        begin = max(0, center_index - (self.size // 2))
        end = min(begin + self.size, len(frame_indices))

        out = frame_indices[begin:end]
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out


class TSN(object): #Temporal Segment Networks (TSN): 긴 영상 전체를 대표하도록 시간 축을 몇 개 구간(segment)으로 나누고, 각 구간에서 짧은 snippet(짧은 클립)을 희소하게 샘플링한다.
    def __init__(self, seq_len=12, snippet_duration=16, center=False):
        self.seq_len = seq_len #구간 개수
        self.snippets_duration = snippet_duration #프레임 개수 => 정리: snippet_duration짜리 짧은 클립(snippet)을 seq_len개 구간만큼 나눈다.
        #center: 각 구간에서 스니펫을 랜덤으로 뽑을지(center=False), 가운데로 뽑을지(center=True)
        self.crop = TemporalRandomCrop(size=self.snippets_duration) if center == False else TemporalCenterCrop(size=self.snippets_duration)

    def __call__(self, frame_indices): #frame_indices: 프레임 번호 리스트
        snippets = []
        pad = LoopPadding(size=self.seq_len * self.snippets_duration)
        frame_indices = pad(frame_indices)
        num_frames = len(frame_indices)
        segment_duration = num_frames // self.seq_len
        assert segment_duration >= self.snippets_duration

        # crop = TemporalRandomCrop(size=self.snippets_duration)
        for i in range(self.seq_len):
            snippets.append(self.crop(frame_indices[segment_duration * i: segment_duration * (i + 1)]))
        return snippets
