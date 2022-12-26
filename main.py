import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.labels import DATA_LABEL_DICT
from src.model.dataset import (BasicTrainImgTransform,
                               BasicValidTestImgTransform, GoodBadImgDataset)
from src.model.model_eval import (inference_dl, plot_confusion_metrix,
                                  plot_roc_curve, plot_train_val_score)
from src.model.simple_model.model import Net
from src.model.train_valid import Trainer
from src.utils import (fix_seeds, get_img_path_and_label_list,
                       load_model_weight, save_model_weight)


def get_arg():
    parser = argparse.ArgumentParser(
        description='train anormary detect dataset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001
    )
    parser.add_argument(
        '--img_root_path',
        default='./Dataset',
        help='画像データセットへのパス'
    )
    parser.add_argument(
        '--default_weights_path',
        default='./weights/default_model_weight.pth',
        help='初回評価に使用する重みパス'
    )
    parser.add_argument(
        '--default_eval_dst',
        default='./evals/default_model/',
        help='初回評価の結果保存先パス'
    )
    parser.add_argument(
        '--train_weights_dst',
        default='./weights/trained_weights/',
        help='学習済みモデルの重み保存ディレクトリ名'
    )
    parser.add_argument(
        '--train_eval_dst',
        default='./evals/trained_model_eval',
        help='学習済みモデルにおける評価の結果保存先パス'
    )
    parser.add_argument(
        '--use_model',
        default='simple_model',
        help='学習に使用するモデル名'
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='seed値'
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='学習時及び訓練時のバッチサイズ'
    )
    parser.add_argument(
        '--fold',
        default=5,
        type=int,
        help='交差検証のfold数'
    )
    return parser.parse_args()


def main():
    args = get_arg()
    img_path_arr, label_arr = get_img_path_and_label_list(
        args.img_root_path
    )
    train_img_path_arr, test_img_path_arr, train_label_arr, test_label_arr = train_test_split(
        img_path_arr, label_arr,
        stratify=label_arr,
        test_size=0.1,
        random_state=args.seed
    )
    valid_and_test_trans = BasicValidTestImgTransform()
    test_ds = GoodBadImgDataset(
        test_img_path_arr,
        test_label_arr,
        valid_and_test_trans
    )
    test_dl = DataLoader(test_ds, shuffle=False)
    # == デフォルトの重みで評価 ==
    # ディレクトリ準備
    eval_dst_dir = Path(args.default_eval_dst)
    eval_dst_dir.mkdir(parents=True, exist_ok=True)
    # モデル読み込み
    model = Net()
    load_model_weight(
        model,
        args.default_weights_path
    )
    # testデータの予測
    y_test, y_pred = inference_dl(model, test_dl)
    y_pred_labels = np.argmax(y_pred, axis=1)
    # 混同行列とROCカーブ記述
    plot_confusion_metrix(y_test, y_pred_labels, dst=eval_dst_dir / 'confusion_metrix.png')
    plot_roc_curve(y_test, y_pred[:, DATA_LABEL_DICT['good']], dst=eval_dst_dir / 'roc.png')
    del model

    # == 学習 ==
    train_trans = BasicTrainImgTransform()
    valid_trans = BasicValidTestImgTransform()
    skf = StratifiedKFold(random_state=args.seed, shuffle=True, n_splits=args.fold)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for i, (train_idx, valid_idx) in enumerate(skf.split(train_img_path_arr, train_label_arr)):
        fix_seeds(args.seed)
        print(f'{i + 1} / {args.fold} ...')
        fold_train_img_path_arr, fold_valid_img_path_arr = (
            train_img_path_arr[train_idx],
            train_img_path_arr[valid_idx]
        )
        fold_train_label_arr, fold_valid_label_arr = (
            train_label_arr[train_idx],
            train_label_arr[valid_idx]
        )
        train_ds = GoodBadImgDataset(
            fold_train_img_path_arr,
            fold_train_label_arr,
            train_trans,
            to_onehot=True
        )
        train_dl = DataLoader(train_ds, args.batch_size, shuffle=True)
        valid_ds = GoodBadImgDataset(
            fold_valid_img_path_arr,
            fold_valid_label_arr,
            valid_trans,
            to_onehot=True
        )
        valid_dl = DataLoader(valid_ds, args.batch_size, shuffle=False)
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(
            epochs=args.epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            model=model
        )
        history = trainer.train(train_dl, valid_dl)


if __name__ == '__main__':
    main()
