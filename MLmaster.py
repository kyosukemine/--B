# モジュールのインポート
import os
import numpy as np
import csv
# import copy
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPool1D, Input,Conv2d
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import gc


def plot_result(history, num):
    '''
    plot result
    全ての学習が終了した後に、historyを参照して、maeとlossをそれぞれplotする
    '''
    # accuracy
    plt.figure()
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.grid()
    plt.legend(loc='best')
    plt.title('mae')
    plt.savefig('./MLData_SPgas//' + SPgas_name + '/graph_mae' + str(num) + '.png')
    # plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    # plt.savefig('./MLData_SPgas/graph_loss' + str(num) +'.png')
    # plt.show()

def fileopen(file_name = ""):
    '''
    fileopen
    csvファイルを読み込み，読み込んだデータをnparrayとして出力する
    '''
    df = pd.read_csv('boin_all_table.csv',header=None,delimiter=',',dtype='float64')
    input_data = []
    output_data = []
    name_num = 0 # kyosuke
    df = df[df.iloc[:,5] == 0] # name_numデータだけ取り出す
    
    # 
    ID = np.unique(df.iloc[:,6].astype(int)) # IDのユニーク番号を取得
    for i in ID:
        input_data.append(df[df.iloc[:,6] == i].iloc[:,7:].values)
        output_data.append(df[df.iloc[:,6] == i].iloc[0,0])

    input_data = np.array(input_data)
    output_data = np.array(output_data).astype(int)

    n_labels = len(np.unique(output_data))  # 分類クラスの数 = 5
    output_data = np.eye(n_labels)[output_data]           # one hot表現に変換

    # print(input_data)
    # print(input_data.shape)

    # print(output_data)
    # print(output_data.shape)

    return input_data,output_data


def CNNLearn(Learn_num, inputdata, outputdata, validation_data_x, validation_data_y):
    '''
    CNNLearn
    CNN(畳み込みニューラルネットワーク)を用いて学習を行い，モデルを保存する
    '''

    # 入力データと出力データのサイズを表示する
    print("inputdata.shape -> " + str(inputdata.shape))
    print("outputdata.shape -> " + str(outputdata.shape))
    
    # 入力データの波形の数(data_num)，1波形の強度データの数(data_len)を取得し表示する．
    input_data_num = inputdata.shape[0]
    print("data_num -> " + str(input_data_num))
    input_data_len = inputdata.shape[1]
    print("data_len -> " + str(input_data_len))

    # 出力データの数(data_num)，1波形の確度，始点時間，終点時間のデータ数(つまり3)(data_len)を取得し表示する．
    output_data_num = outputdata.shape[0]
    print("data_num -> " + str(output_data_num))
    output_data_len = outputdata.shape[1]
    print("data_len -> " + str(output_data_len))

    # モデルの構築
    batchsize = 1                         # バッチサイズ 1,32,128,256,512
    epoch = 100                           # 学習世代
    number_filters_1 = 64                 # フィルター数
    number_filters_2 = 32                 # フィルター数
    kernel_size_1 = 20                    # カーネルサイズ
    kernel_size_2 = 10                    # カーネルサイズ
    pool_size_1 = 2                       # プールサイズ
    pool_size_2 = 2                       # プールサイズ
    stride_1 = 1                          # ストライド数
    stride_2 = 1                          # ストライド数
    drop_out = 0.25                       # ドロップアウトレート    
    activatioin_func = 'tanh'             # 活性化関数
    opt = keras.optimizers.Adam(lr=1e-4)  # オプティマイザー
    loss_func = "mae"                     # 損失関数

    # モデル構造
    inputs = Input(shape=(input_data_len, 1))


    x = Conv1D(filters=number_filters_1, kernel_size=kernel_size_1, padding='same', strides=stride_1)(inputs)
    x = Activation(activatioin_func)(x)
    x = MaxPool1D(pool_size=pool_size_1, padding='valid')(x)
    x = Conv1D(filters=number_filters_2, kernel_size=kernel_size_2, padding='same', strides=stride_2)(x)
    x = Activation(activatioin_func)(x)
    x = MaxPool1D(pool_size=pool_size_2, padding='valid')(x)
    x = Flatten()(x)
    x = Dropout(drop_out)(x)
    x = Dense(units=200)(x)
    x = Activation(activatioin_func)(x)
    x = Dense(units=output_data_len)(x)
    output = Activation('linear')(x)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(loss=loss_func,
                optimizer=opt,
                metrics=['mae'])
    model.summary()

    # モデルの保存方法の設定
    baseSaveDir = "./MLData_SPgas//" + SPgas_name
    # es_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, verbose=1, mode="auto")
    chkpt = os.path.join(baseSaveDir, 'Learned_model' + str(Learn_num) + '.h5')
    # cp_cb = ModelCheckpoint(filepath = chkpt, monitor= 'val_loss', verbose = 1, save_best_only= True, mode='auto')
    csv_logger = CSVLogger('./MLData_SPgas//' + SPgas_name + '/traing_logg' + str(Learn_num) + '.csv', separator = ',', append = False)

    # 学習
    history = model.fit(x=[inputdata],
                y=outputdata, 
                batch_size=batchsize, epochs=epoch, verbose=1, validation_data=(validation_data_x, validation_data_y)
                )

    # 学習過程を表示
    plot_result(history, Learn_num)
    
    # グラフ保存
    # plot_model(model, to_file = './MLData_SPgas/model_' + str(Learn_num) + '.png', show_shapes= True)

    # モデルの保存と後処理
    model.save("./MLData_SPgas//" + SPgas_name + '/Learned_model0.h5')
    # del model
    K.clear_session()
    gc.collect()






def main():
    '''
    main
    main関数
    '''
    num_learning = 1 # 学習回数

    # 入出力データのファイル名取得
    file_name = 'boin_all_table.csv'

    #入力データの読み込み
    input_data,output_data = fileopen(file_name)


    # 入出力データのサイズ取得と表示
    input_data_num = input_data.shape[0]            # shape[0]は行の数(波形データの数)
    print("data_num -> " + str(input_data_num))
    input_data_len = input_data.shape[1]            # shape[1]は列の数(1波形に対する強度データの数)
    print("data_len -> " + str(input_data_len))

    output_data_num = output_data.shape[0]          # shape[0]は行の数(波形データの数)
    print("data_num -> " + str(output_data_num))
    output_data_len = output_data.shape[1]          # shape[1]は列の数(1波形に対する確度，始点，終点)
    print("data_len -> " + str(output_data_len))
    
    print("input_data.shape -> " + str(input_data.shape))
    print("output_data.shape -> " + str(output_data.shape))

    # 学習のため入力データの次元を増やす
    input_data = input_data[:,:,np.newaxis]
    print("input_data.shape -> " + str(input_data.shape))

    # 訓練データとテストデータに分ける（訓練データ8割）
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, train_size=0.8, random_state=0,stratify=output_data[:,0])

    # 正規化
    num = 0
    max_numlist_test = []       # 各波形での最大強度を格納する用のリスト
    min_numlist_test = []       # 各波形での最小強度を格納する用のリスト

    for data in x_test:
        max_num = data.max()
        min_num = data.min()
        max_numlist_test.append(max_num)    # 各波形での最大強度を格納
        min_numlist_test.append(min_num)    # 各波形での最小強度を格納

        for i in range(input_data_len):
            std = (data[i]-min_num) / (max_num - min_num)   # 最大値を1，最小値を0とする処理
            scaled = std  
            x_test[num][i][0] = scaled

        num += 1

    num = 0
    for data in x_train:
        max_num = data.max()
        min_num = data.min()
        for i in range(input_data_len):
            std = (data[i]-min_num) / (max_num - min_num)
            scaled = std  
            x_train[num][i][0] = scaled
        num += 1

    # 訓練データとテストデータを保存
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()
    y_train_copy = y_train.copy()
    y_test_copy = y_test.copy()
    print(x_test_copy.shape)
    x_train_copy = x_train_copy.squeeze(axis = 2)
    x_test_copy = x_test_copy.squeeze(axis = 2)
    np.savetxt("./MLData_SPgas//" + SPgas_name + "/x_train.csv", x_train_copy, delimiter = ",")
    np.savetxt("./MLData_SPgas//" + SPgas_name + "/x_test.csv", x_test_copy, delimiter = ",")
    np.savetxt("./MLData_SPgas//" + SPgas_name + "/y_train.csv", y_train_copy, delimiter = ",")
    np.savetxt("./MLData_SPgas//" + SPgas_name + "/y_test.csv", y_test_copy, delimiter = ",")

    # 学習するかどうか判定
    if Learn_sw == 1:
        # データをシャッフルさせて学習
        for i in range(num_learning):
            shuffle = np.arange(x_train.shape[0])
            np.random.shuffle(shuffle)
            shuffle_list = np.ndarray.tolist(shuffle)
            x_train = x_train[shuffle_list,:]
            y_train = y_train[shuffle_list,:]
        
            CNNLearn(i, x_train, y_train, x_test, y_test)
    
    # モデル読み込み
    model = load_model("./MLData_SPgas//" + SPgas_name + '/Learned_model0.h5', compile=False)

    # モデル構成の確認
    model.summary()

    # 時間データ読み込み
    time = np.loadtxt('./MLData_SPgas//' + SPgas_name + '/input_SPgas_time.csv',dtype='float32', delimiter=",")

    # テストデータのサイズ取得
    test_data_num = x_test.shape[1]     # shape[1]は行の数(波形データの数)
    test_data_len = x_test.shape[0]     # shape[0]は列の数(1波形に対する強度データ)

    predict_datalist = []   # 予測データを保存するリスト


    for column_num in tqdm(range(test_data_len)):
        input_data_column_len = input_data.shape[1]     # 1波形に対する強度データの数取得
        output_data_column_len = output_data.shape[1]   # 1波形に対する正解データ(確度，始点，終点)の数(3)を取得

        # 予測用データ取得
        x_predict = x_test[column_num].reshape(1,input_data_column_len)     # 予測する波形データを用意する．モデルの入力とデータの構造を一致させる
        y_test1 = y_test[column_num].reshape(1,output_data_column_len)      # 正解データを用意する．モデルの出力とデータの構造を一致させる

        # 予測用データ(x_predict)を入力として予測を行う
        Predict = model.predict(x_predict, verbose=0)

        # 強度データの標準化を戻す
        j = 0
        for data in x_test[column_num]:
            x_test[column_num,j,0] = (max_numlist_test[column_num] - min_numlist_test[column_num]) * data[0] + min_numlist_test[column_num]
            j += 1
        
        # 予測データのデータ構造を変更する
        predict_data = Predict.reshape(-1,output_data_column_len)

        # 予測結果と正解を表示する
        print("\ny_test",y_test1[0])
        print("predict",predict_data[0])


# グラフのスタイル設定
plt.style.use('seaborn-dark-palette')

# 学習を行うかどうか(0:行わない 1:行う)
Learn_sw = 0
# グラフ表示を行うかどうか(0:行わない 1:行う)
plt_sw = 1


