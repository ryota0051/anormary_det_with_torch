## 動作の前提条件

Docker desktop がインストールされており、動作するのが確認できていること
インストールしていない場合は以下のリンクからダウンロード
https://www.docker.com/products/docker-desktop/

## 環境構築

1. 先端課題 018_020 のディレクトリに移動
2. docker-compose up -d
3. アドレスバーに「localhost:8888」を打ち込む

## model の説明

画像が良品か不良品かを判定するモデルです。
不良品でも種類はありますが、今回のモデルはどの種類の不良品であるかまでは判定しません
あくまでも、良品 or 不良品を判定するモデルです。

## フォルダ構成

.先端課題 018_020
├code
| ├model.py モデル構成ファイル
| ├model_weight.pth 学習済みの重み
| └example_code.ipynb チュートリアルのコード
└Dataset
├bad
| ├bent
| ├color
| ├flip
| └scratch
└good
