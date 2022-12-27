## 動作の前提条件

- Docker がインストールされており、動作するのが確認できていること

- Dataset ディレクトリ配下が以下の構成となっていること

  ```
  ./Dataset
  |-good
  | |_良品画像一覧
  |_bad
    |-bent
    |  |_bent画像一覧
    |-color
    |  |_color画像一覧
    |-flip
    |  |_flip画像一覧
    |_scratch
       |_scratch画像一覧
  ```

## 環境構築

1. 本リポジトリを clone して、ディレクトリに移動
2. `ocker compose run --rm anormaly_detect bash`を実行
3. `python main.py`を実行
4. `./evals/trained_model_eval`配下に各 fold の結果と全 fold の予測値を平均した値結果が格納される(model の重みは、`./weights/trained_weights/fold_N`(N は fold 数)ディレクトリ配下の weight.pth に保存される。)

## model の説明

画像が良品か不良品かを判定するモデルです。
不良品でも種類はありますが、今回のモデルはどの種類の不良品であるかまでは判定しません
あくまでも、良品 or 不良品を判定するモデルです。

## 評価方法

以下の 2 つの評価指標を使用

    - 混同行列

    - rocのaucスコア

## 工夫点

- 学習時に、以下の 3 つのデータ拡張実施

  - ランダムで上下反転

  - ランダムで左右反転

  - ランダムで-180 ~ 180 度回転

- モデルを交差検証で学習し、最終的な結果をそれぞれの検証パートでの推論結果の平均値として評価を実施
