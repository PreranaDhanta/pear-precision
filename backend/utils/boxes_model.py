from dataclasses import dataclass
from backend.utils import spray_rules

@dataclass
class YieldConfig:
    pears_per_box: float = 100.0  # average pears per standard box (adjust to your orchard)
    detection_recall: float = 0.9 # compensate undercounting if measured

def counts_to_boxes(count:int, cfg:YieldConfig)->float:
    corrected = count / max(cfg.detection_recall, 1e-6)
    return corrected / cfg.pears_per_box
