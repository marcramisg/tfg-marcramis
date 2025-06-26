import random

import numpy as np
import torch
from torch.backends import cudnn


class RandomUtils:
    @staticmethod
    def set_randomness(seed: None | int = 42, deterministic: bool = False) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        if deterministic and torch.cuda.is_available():
            # This can decrease the performance
            cudnn.deterministic = True
            cudnn.benchmark = False
