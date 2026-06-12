# live-cv-demos

ノート PC の CPU でも動かしやすい、ライブカメラ向けのコンピュータビジョンデモ集です。オープンキャンパスなどで、カメラ映像に Deep Learning の推定結果を重ねて表示する用途を想定しています。

主な機能:

- Ultralytics YOLO による物体検出
- 複数人の姿勢推定と、立つ・座る・移動中の簡易ラベル表示
- `ElenaRyumina/face_emotion_recognition` による複数顔の感情推定
- 全感情クラス確率のリアルタイム折れ線グラフ
- 任意でインスタンスセグメンテーション
- MediaPipe Hands を使った空中お絵描きデモ

OpenCV でカメラ入力と描画を行い、重いモデルは数フレームごとに実行します。展示画面が読みにくくならないように、物体検出は BBox、姿勢推定は骨格と行動ラベル、セグメンテーションはマスクだけを表示します。

感情推定が有効な場合は、カメラ映像の下に 7 クラス全ての確率を折れ線グラフで表示します。複数の顔が検出された場合、グラフには各顔の平均確率を表示します。

## セットアップ

```bash
uv sync
```

Python は `.python-version` で `3.12.2` に固定しています。PyTorch、torchvision、OpenCV、Ultralytics の wheel が Python 3.11/3.12 で安定しやすいためです。

## ライブ CV デモ

```bash
uv run live-cv-demo
```

終了は `q` または `Esc` です。

初回起動時は YOLO の重みと Hugging Face の感情推定モデルをダウンロードします。2 回目以降はローカルファイルを優先して使います。

YOLO 重みの探索順:

- `models/yolo/<model-name>.pt`
- リポジトリ直下、例: `yolo11n.pt`
- 現在の作業ディレクトリ

感情推定モデルの既定ファイル:

```text
ElenaRyumina/face_emotion_recognition::FER_static_ResNet50_AffectNet.pt
```

Hugging Face のモデルは `models/hf/` に保存します。YOLO の重みは `models/yolo/` に保存します。

## CPU 向け設定

重い場合は、使う機能や実行頻度を下げます。

```bash
uv run live-cv-demo --no-detect
uv run live-cv-demo --no-emotion
uv run live-cv-demo --no-detect --imgsz 256
```

推論頻度を下げる例:

```bash
uv run live-cv-demo --detect-every 6 --pose-every 4 --emotion-every 10
```

PyTorch の CPU スレッド数を制限する例:

```bash
uv run live-cv-demo --torch-threads 4
```

YOLO 重みの保存先を変える例:

```bash
uv run live-cv-demo --yolo-cache-dir /path/to/yolo-weights
```

感情グラフを調整または非表示にする例:

```bash
uv run live-cv-demo --emotion-graph-height 140 --emotion-history 120
uv run live-cv-demo --no-emotion-graph
```

## セグメンテーション

インスタンスセグメンテーションは CPU では重いため、既定では無効です。

```bash
uv run live-cv-demo --segment --segment-every 12
```

## 空中お絵描きデモ

`live-cv-airdraw` は、人差し指で空中に光る線を描く別デモです。

```bash
uv run live-cv-airdraw
```

操作:

- 人差し指だけを立てると描画開始
- 両手で同時に描画可能
- パーを出すと描画停止
- グーを出すと描画を一時停止し、次の線を別ストロークにする
- ストロークごとに明るい 30 色のパレットから色を切り替える
- 描画点は約 60 秒後に弾けて消える
- 画面下部に `グー`、`人差し指`、`パー` の説明バーを表示する
- `c` でキャンバス消去
- `q` または `Esc` で終了

便利なオプション:

```bash
uv run live-cv-airdraw --camera 0 --width 1280 --height 720
uv run live-cv-airdraw --fullscreen
uv run live-cv-airdraw --lifetime 45
```

このデモでは MediaPipe Hands で 21 点の手ランドマークを取得し、OpenCV で描画、発光、粒子、消滅演出を行います。Ultralytics YOLO の pose モデルは全身の骨格推定には有効ですが、指先ランドマークは提供しないため、手のデモには MediaPipe Hands を使っています。

## モデル

既定では軽量な nano 系 YOLO モデルを使います。

- `yolo11n.pt`
- `yolo11n-pose.pt`
- `yolo11n-seg.pt`

感情推定モデルの読み込みでは、まず TorchScript として読み込みます。state dict の場合は、`FER_static_ResNet50_AffectNet.pt` で使われている VGGFace 系 ResNet-50 に対応します。読み込みに失敗した場合でも、アプリ全体は止めず、感情推定だけを無効化します。

## プライバシー

既定ではカメラ画像を保存しません。感情ラベルは展示用の推定値であり、正解や診断として扱わないでください。
