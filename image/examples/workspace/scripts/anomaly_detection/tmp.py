#%% コード 7.1 | PyTorchのプラットフォーム選択および乱数シードの指定
###### PyTorchの設定 ######
import torch

# 学習・推論を実施するプラットフォームの選択
DEVICE = 'mps'
if DEVICE == 'cuda':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'mps':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    device = 'cpu'
# 乱数シードの指定
torch.manual_seed(42)

#%% コード 7.20 | 学習済VGG-11モデルの読込と、転移学習・ファインチューニング対象層の指定
###### 1. モデル構造の決定 ######
import torch.nn as nn
from torchvision import models

DROPOUT = 0.4
# モデルのインスタンスを作成（ImageNetで学習した事前学習済み重みを指定）
model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

# ファインチューニング対象の層のパラメータ名
tuned_layers = [
    'features.16.weight',  # 後ろから2番目の畳み込み層の重み
    'features.16.bias',  # 後ろから2番目の畳み込み層のバイアス
    'features.18.weight',  # 最後の畳み込み層の重み
    'features.18.bias'  # 最後の畳み込み層のバイアス
    ]
# ファインチューニング対象層以外のパラメータを更新の対象外とする
for name, param in model.named_parameters():
    if name in tuned_layers:  # ファインチューニング対象層
        param.requires_grad = True
    else:  # 対象外の層
        param.requires_grad = False

# 全結合層を入れ替える（転移学習）
model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),  # 全結合層（9層目）
            nn.ReLU(inplace=True),  # 活性化関数（ReLU）
            nn.Dropout(p=DROPOUT),  # dropout
            nn.Linear(4096, 4096),  # 全結合層（10層目）
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(4096, out_features=2),  # 全結合層（クラス数2を指定）
        )

#%% コード 7.21 | 前処理の選択
###### 2. 前処理の選択 ######
from torchvision.transforms import v2

# 前処理を記述したインスタンス
transform = v2.Compose([
    v2.RandomRotation(degrees=10),
    v2.RandomResizedCrop(
        size=(28, 28),
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1)
    ),
    v2.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.05
    ),
    v2.Resize(224),  # TorchVisionのVGG11の入力サイズにリサイズ
    v2.ToTensor(),
])

#%% コード 7.4 | データ読込用のDatasetクラスの実装
###### 3. データの読込 ######
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image

class MNISTAnomalyDataset(MNIST):
    """Datasetを定義するクラス"""
    def __init__(self, root, train=True, transform=None, include_anomaly=True,
                 norm_labels=[9], anom_labels=[4, 7]):
        """データ全体の読込方法を指定するメソッド"""
        # 継承元の`torchvision.datasets.MNIST`クラスを初期化
        super().__init__(root=root, train=train, download=True, transform=transform)
        # 正常データ（数字の"9"）
        self.normal_data = [i for i in range(len(self.data))
                            if self.targets[i] in norm_labels]
        # 異常データ（数字の"4"と"7"）
        if include_anomaly:
            self.anomaly_data = [i for i in range(len(self.data))
                                if self.targets[i] in anom_labels]
        else:
            self.anomaly_data = []

    def __len__(self):
        """データ数のカウント用メソッド"""
        return len(self.normal_data) + len(self.anomaly_data)

    def __getitem__(self, idx):
        """データ1個をインデックスから取得するメソッド"""
        # 正常データ（ラベル=0）
        if idx < len(self.normal_data):
            img = self.data[self.normal_data[idx]]
            label = 0
        # 異常データ（ラベル=1）
        else:
            img = self.data[self.anomaly_data[idx - len(self.normal_data)]]
            label = 1
        # 画像をPIL形式のRGB画像に変換
        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")
        # 前処理を適用
        if self.transform is not None:
            img = self.transform(img)
        return img, label

#%% コード 7.5 | Datasetのインスタンス作成
# Datasetのインスタンス作成
trainset = MNISTAnomalyDataset(root='../sample_data', train=True, transform=transform)
testset = MNISTAnomalyDataset(root='../sample_data', train=False, transform=transform)

#%% コード 7.6 | DataLoaderのインスタンス作成
# DataLoaderのインスタンス作成
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
testloader = DataLoader(testset, batch_size=16, shuffle=True)

#%% コード 7.22 | バッチ内の画像の可視化
###### 4. バッチ内の画像の可視化 ######
import matplotlib.pyplot as plt

# 最初のミニバッチのデータを取得
train_iter = iter(trainloader)
images, labels = next(train_iter)
# 最初のミニバッチ内の画像を可視化
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
for i, (img, label) in enumerate(zip(images, labels)):
    # Torch.Tensorの次元の順序（ch, h, w）をplt.imshowの順序（h, w, ch）に変換
    img_permute = img.permute(1, 2, 0)
    # 画像をプロット
    axes[int(i/4)][i%4].imshow(img_permute)
plt.show()

#%% コード 7.8 | 損失関数と最適化アルゴリズムの選択
###### 5. 損失関数と最適化アルゴリズムの選択 ######
import torch.optim as optim

# 損失関数（交差エントロピー誤差）
criterion = nn.CrossEntropyLoss()
# 最適化アルゴリズム（Adam）
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% コード 7.9 | 1エポックの学習を実行する関数
###### 6. 誤差逆伝播法を用いて学習を実施 ######
def train_one_epoch(trainloader, device, model,
                    criterion, optimizer):
    """1エポックの学習を実行する関数"""
    train_running_loss = 0.0  # 損失のエポック平均
    # DataLoaderを使用してミニバッチごとにループを繰り返す
    for i, (images, labels) in enumerate(trainloader):
        # 画像とラベルを学習で使用するデバイス（GPU等）に送る
        images, labels = images.to(device), labels.to(device)
        # 損失を計算
        optimizer.zero_grad()  # 重みパラメータの勾配を初期化
        outputs = model(images)  # 順伝播（推論）
        loss = criterion(outputs, labels)  #損失関数
        # 誤差逆伝播法で重みパラメータを更新
        loss.backward()
        optimizer.step()
        # 損失の合計を計算
        train_running_loss += loss.item()
    # 損失の合計をミニバッチのループ数で割って平均を求める
    train_running_loss /= len(trainloader)

    return train_running_loss

#%% コード 7.10 | 1エポックのテストデータに対する損失評価用関数
def test_one_epoch(testloader, device, model,
                   criterion):
    """1エポックのテストデータに対する平均損失を計算する関数"""
    test_running_loss = 0.0  # 損失のエポック平均
    model.eval()  # モデルを評価モードに変更
    # 勾配計算を無効化してメモリを節約
    with torch.no_grad():
        # DataLoaderを使用してミニバッチごとにループを繰り返す
        for i, (images, labels) in enumerate(testloader):
            # 画像とラベルを学習で使用するデバイス（GPU等）に送る
            images, labels = images.to(device), labels.to(device)
            # 損失を計算
            outputs = model(images)  # 順伝播（推論）
            loss = criterion(outputs, labels)  #損失関数
            # 損失の合計を計算
            test_running_loss += loss.item()
    # 損失の合計をミニバッチのループ数で割って平均を求める
    test_running_loss /= len(testloader)

    return test_running_loss

#%% コード 7.11 | エポック数だけ学習をループ実行
import time

NUM_EPOCHS = 10  # 学習のエポック数
model.to(device)  # モデルを学習で使用するデバイス（GPU等）に送る
train_losses, test_losses = [], []  # 損失の推移を格納するリスト
start = time.time()  # 経過時間を計測
# エポック数だけループを繰り返す
for epoch in range(NUM_EPOCHS):
    # 1エポック分の学習を実行
    train_running_loss = train_one_epoch(trainloader, device, model,
                                         criterion, optimizer)
    train_losses.append(train_running_loss)
    # 1エポック分のモデル評価を実施
    test_running_loss = test_one_epoch(testloader, device, model,
                                       criterion)
    test_losses.append(test_running_loss)
    # 学習経過を表示
    elapsed_time = time.time() - start
    print(f'Epoch: {epoch + 1}, train_loss: {train_running_loss}, test_loss: {test_running_loss}, elapsed_time: {time.time() - start}')

#%% コード 7.12 | 学習データとテストデータにおける平均損失をプロットして収束を確認
###### 7. 収束の確認 ######
# 学習データとテストデータでのエポックごとの損失履歴をプロット
plt.plot(train_losses, label='Train loss', linestyle='-', color='#444444')
plt.plot(test_losses, label='Test loss', linestyle=':', color='#444444')
plt.title('Loss history')
plt.legend()
plt.show()

#%% コード 7.13 | 混同行列の計算
###### 8. 性能評価 ######
import numpy as np
from sklearn.metrics import confusion_matrix

# 正解ラベルと推論結果の格納用
labels_true = []
labels_pred = []
# テストデータに対して推論を実施
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        # ミニバッチの推論結果を保持
        images = images.to(device)
        outputs = model(images)  # 順伝播
        preds = torch.argmax(outputs, dim=1)  # 推論されたラベル
        preds = preds.cpu().detach().numpy()  # numpy.ndarrayに変換
        labels_pred.append(preds)
        # ミニバッチの正解ラベルを保持
        labels = labels.numpy()  # numpy.ndarrayに変換
        labels_true.append(labels)
# 全てのミニバッチの推論結果と正解ラベルを結合
labels_true = np.concatenate(labels_true)
labels_pred = np.concatenate(labels_pred)
# 混同行列
confmat = confusion_matrix(labels_true, labels_pred, labels=[0, 1])
print(confmat)

#%% コード 7.14 | 各種評価指標によるモデルの分類性能評価
# 各種指標の計算
tn, fp, fn, tp = confmat.ravel()
# accuracy
accuracy = (tp+tn) / (tn+fp+fn+tp)
print(f'accuracy={accuracy}')
# ヒット率
recall = tp / (tp+fn)
print(f'recall={recall}')
# 誤報率
fpr = fp / (fp+tn)
print(f'fpr={fpr}')
# F1
precision = tp / (tp+fp)
f1 = 2*precision*recall / (precision+recall)
print(f'f1={f1}')

# %%
