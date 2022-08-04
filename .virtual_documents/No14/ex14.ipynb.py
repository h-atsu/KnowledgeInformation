get_ipython().run_line_magic("matplotlib", " inline")
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os, struct
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2次元(28 x 28画素)から1次元（784画素）に変換
X_train = X_train.reshape(X_train.shape[0], 28*28, 1)
X_test = X_test.reshape(X_test.shape[0], 28*28, 1)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
X_train.shape


# 学習データとテストデータは最初のn_train_data, n_test_data個用いる
n_training_data = 1000
n_test_data = 1000

X_trn = X_train[:n_training_data][:]
y_trn = y_train[:n_training_data]
X_tst = X_test[:n_test_data][:]
y_tst = y_test[:n_test_data]

# 値の範囲を[0,1]に変換 (Keras用)
X_trn = X_trn.astype('float32')/255
X_tst = X_tst.astype('float32')/255

# One-hot encoderによりクラスラベルをバイナリに変換 (Keras用)
# 例：1 -> [1,0,...,0], 2 -> [0,1,0,...]
y_trn = utils.to_categorical(y_trn)
y_tst = utils.to_categorical(y_tst)

# 入力データの次元数(=784画素)を取得
n_dim = X_trn.shape[1]

# 出力クラス数(=10クラス)
n_out = y_trn.shape[1]

n_dim, n_out


# Accuracyの履歴のプロット
def plot_history_acc(rec):
    plt.plot(rec.history['accuracy'],"o-",label="train") 
    plt.plot(rec.history['val_accuracy'],"o-",label="test") 
    plt.title('accuracy history')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
    
# 損失関数値の履歴のプロット
def plot_history_loss(rec):
    plt.plot(rec.history['loss'],"o-",label="train",)
    plt.plot(rec.history['val_loss'],"o-",label="test")
    plt.title('loss history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()


ae = Sequential()
ae.add(Dense(500, input_dim = n_dim, activation='relu', name = 'encoder'))
ae.add(Dropout(0.5))
ae.add(Dense(n_dim, activation='relu'))

# 損失関数にMean Squared Error（平均二乗誤差），　パラメータの最適化にAdamを設定
# Adam(Adaptive moment estimation)は確率的最急降下法の一種． 
ae.compile(loss = 'mse', optimizer ='adam')

# Kerasもfit()で学習を行う． ただし戻り値として学習履歴を返す．
records_ae = ae.fit(X_trn, X_trn,
                    epochs=250,
                    batch_size=200,
                    shuffle=True,
                    verbose=0,
                    validation_data=(X_tst, X_tst))


# 学習済み重みの保存
ae.save_weights('autoencoder.h5')
# ネットワークの概要
ae.summary()
# 損失関数値の履歴のプロット
plot_history_loss(records_ae)


# テスト画像をAutoEncoderで変換
decoded_imgs = ae.predict(X_tst)

n = 10 #表示枚数
plt.figure(figsize=(20, 4))
for i in range(n):
    # 元画像の表示
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_tst[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 再構成画像の表示
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# AutoEncoderの学習結果（Encoderの学習済み重み）を取得
h = ae.get_layer('encoder').output
# 最終段にクラス数の分の出力を持つsoftmax関数を追加
y = Dense(n_out, activation='softmax', name='predictions')(h)

# 入力から出力層までを繋ぐNNモデルを生成
dnn = Model(inputs=ae.inputs, outputs=y)

# 損失関数は交差エントロピーを使用． 最近は平均二乗誤差ではなく交差エントロピーが用いられている．
dnn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

records_dnn = dnn.fit(X_trn, y_trn,
                epochs=50,
                batch_size=200,
                shuffle=True,
                verbose=0,
                validation_data=(X_tst, y_tst))


# ネットワークの概要
dnn.summary()
# 学習履歴のプロット
plot_history_acc(records_dnn)
plot_history_loss(records_dnn)


from tqdm.notebook import tqdm
ex_dnns = []
ex_ae_records = []
ex_dnn_records = []
for num_layer in tqdm(range(3)):    
    # ========== AE =========
    ae = Sequential()
    ae.add(Dense(500, input_dim = n_dim, activation='relu', name = 'encoder'))
    ae.add(Dropout(0.5))
    for i in range(2*num_layer):
        ae.add(Dense(500, input_dim = 500, activation = 'relu', name = f'intermediate{i}'))
        ae.add(Dropout(0.5))
    
    ae.add(Dense(n_dim, activation='relu'))

    # 損失関数にMean Squared Error（平均二乗誤差），　パラメータの最適化にAdamを設定
    # Adam(Adaptive moment estimation)は確率的最急降下法の一種． 
    ae.compile(loss = 'mse', optimizer ='adam')

    # Kerasもfit()で学習を行う． ただし戻り値として学習履歴を返す．
    records_ae = ae.fit(X_trn, X_trn,
                        epochs=500,
                        batch_size=200,
                        shuffle=True,
                        verbose=0,
                        validation_data=(X_tst, X_tst))
    ex_ae_records.append(records_ae)
    
    
    # ========= DNN ============
    # AutoEncoderの学習結果（Encoderの学習済み重み）を取得
    dnn = Sequential()
    for i in range(2*num_layer+2):
        dnn.add(ae.layers[i])
    
    # 事前学習済みの重みは固定
    #for layer in dnn.layers:
    #    layer.trainable = False

    # 最終段にクラス数の分の出力を持つsoftmax関数を追加
    dnn.add(Dense(n_out, activation='softmax', name='predictions'))


    # 損失関数は交差エントロピーを使用． 最近は平均二乗誤差ではなく交差エントロピーが用いられている．
    dnn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    records_dnn = dnn.fit(X_trn, y_trn,
                    epochs=100,
                    batch_size=200,
                    shuffle=True,
                    verbose=0,
                    validation_data=(X_tst, y_tst))
    ex_dnns.append(dnn)
    ex_dnn_records.append(records_dnn)


ae_train = []
ae_test = []
dnn_train = []
dnn_test = []
for i in range(3):
    ae_train.append(ex_ae_records[i].history['loss'][-1])
    ae_test.append(ex_ae_records[i].history['val_loss'][-1])
    dnn_train.append(ex_dnn_records[i].history['accuracy'][-1])
    dnn_test.append(ex_dnn_records[i].history['val_accuracy'][-1])


# ネットワークの概要
for i in range(3):
    ex_dnns[i].summary()
    # 学習履歴のプロット
    plot_history_loss(ex_ae_records[i])
    plot_history_acc(ex_dnn_records[i])


plt.plot(range(1,4), ae_train, label = 'train')
plt.plot(range(1,4), ae_test, label = 'test')
plt.xlabel('layer num')
plt.ylabel('loss function value')
plt.legend()
plt.title("auto encoder loss ")


plt.plot(range(1,4), dnn_train, label = 'train')
plt.plot(range(1,4), dnn_test, label = 'test')
plt.xlabel('layer num')
plt.ylabel('accuracy')
plt.legend()
plt.title("dnn accuracy")


hidden_num = [10,50,100,300,500,700,1000]
ex_single_dnn = []
for h in tqdm(hidden_num):
    model = Sequential()
    model.add(Dense(h, input_dim = n_dim, activation='relu', name='input_layer'))
    model.add(Dense(n_out, activation='softmax', name='predictions'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    records_dnn = dnn.fit(X_trn, y_trn,
                    epochs=100,
                    batch_size=200,
                    shuffle=True,
                    verbose=0,
                    validation_data=(X_tst, y_tst))
    ex_single_dnn.append(records_dnn)


from keras.layers import Conv2D, MaxPooling2D, Flatten

# 2次元の画像として扱うため28x28x1へ変換
X_trn_cnn = X_trn.reshape(X_trn.shape[0], 28, 28, 1)
X_tst_cnn = X_tst.reshape(X_tst.shape[0], 28, 28, 1)
