import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             confusion_matrix)

from src.labels import DATA_LABEL_DICT


def inference_dl(model, dl):
    """データローダのデータを推論して返す
    Args:
        model: 推論に使用するモデル
        dl: データローダ
    Returns:
        (ラベル, 予測結果)
    """
    model.eval()
    y_test = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in dl:
            pred = model(x_batch)
            y_test.append(y_batch.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    return np.concatenate(y_test), np.concatenate(y_pred)


def plot_roc_curve(y_test, y_pred, dst: str = None):
    """rocカーブをプロットする
    Args:
        y_test: ラベル (N, )
        y_pred: 予測確率 (N, )
        dst: 画像保存先パス
    """
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend()
    if dst:
        plt.savefig(dst)
        plt.close()
        plt.cla()


def plot_confusion_metrix(
        y_test,
        y_pred,
        labels=list(DATA_LABEL_DICT.keys()),
        dst: str = None):
    """混同行列描画関数
    Args:
        y_test: ラベル (N, )
        y_pred: 予測確率 (N, )
        labels: 混同行列に記載するラベル名
        dst: 画像保存先パス
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    if dst:
        plt.savefig(dst)
        plt.close()
        plt.cla()


def plot_train_val_score(
        train_scores,
        valid_scores,
        score_name='auc',
        dst: str = None):
    """学習曲線の描画関数
    Args:
        train_scores: 学習データのスコア配列
        valid_scores: 検証データのスコア配列
        score_name: 評価指標名
        dst: プロット結果保存先
    """
    epochs = np.arange(len(train_scores)) + 1
    plt.plot(epochs, train_scores, '-o', label=f'train {score_name}')
    plt.plot(epochs, valid_scores, '--<', label=f'valid {score_name}')
    plt.xlabel('epochs')
    plt.ylabel(score_name)
    if dst:
        plt.savefig(dst)
        plt.close()
        plt.cla()
