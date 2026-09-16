import os

import numpy as np
import pytest
import torch

from vtd_rl.policy import device
from vtd_rl.world.board import load_curriculum

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
STAGE1 = os.path.join(REPO, "curricula", "stage1.json")
STAGE2 = os.path.join(REPO, "curricula", "stage2.json")


def test_토치와_numpy_짝():
    assert torch.__version__.split(".")[0] == "2"
    assert np.__version__ == "1.26.4"
    x = torch.zeros(3, dtype=torch.float32)
    assert x.numpy().dtype == np.float32           # numpy 2.x 면 여기서 깨진다


def test_장치_고르기():
    assert device("cpu").type == "cpu"
    assert device("auto").type in ("cuda", "cpu")


def test_단계2_커리큘럼():
    name1, b1 = load_curriculum(STAGE1)
    name2, b2 = load_curriculum(STAGE2)
    assert [b.name for b in b2] == [b.name for b in b1]
    assert all(b.signals == "cycle" for b in b2)
    assert all(b.signals == "always_green" for b in b1)
