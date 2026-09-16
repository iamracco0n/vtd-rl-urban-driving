"""학생 정책 — 관측 인코딩, 신경망, 데이터 수집·학습·평가."""
import torch


def device(prefer: str = "auto") -> torch.device:
    """`auto` 면 CUDA 가 있을 때 CUDA. 정책망이 작아 CPU 로도 학습되지만, 수집이 CPU 를 다 쓴다."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)
