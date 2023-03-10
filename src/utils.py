import random
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.labels import DATA_LABEL_DICT


def get_img_path_and_label_list(
        root_dir: str,
        data_label_dict: Dict[str, int] = DATA_LABEL_DICT):
    """画像とラベルのデータセットを取得する
    Args:
        root_dir: 画像データのルートディレクトリ
        data_label_dict: データラベル名とインデックス辞書

    Returns:
        (画像パスリスト, ラベルリスト)
    """
    img_path_list = []
    label_list = []
    for label, label_id in data_label_dict.items():
        img_root = Path(root_dir) / label
        partial_img_path_list = list(
            glob(str(img_root / '**/*.png'), recursive=True)
        )
        partial_label_list = [label_id] * len(partial_img_path_list)
        img_path_list += partial_img_path_list
        label_list += partial_label_list
    return np.array(img_path_list), np.array(label_list)


def save_model_weight(model, dst: str):
    """モデルの重みを保存する
    Args:
        model: 保存モデル
        dst: 保存先
    """
    torch.save(model.state_dict(), dst)


def load_model_weight(model, src: str):
    """モデルの重みをロードする
    Args:
        model: モデルロード対象
        src: 重みファイルパス
    """
    model.load_state_dict(torch.load(src))


def fix_seeds(seed=0):
    """seed値の固定関数
    Args:
        seed: seed値
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
