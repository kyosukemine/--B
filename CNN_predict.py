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
from keras.layers import Conv1D, MaxPool1D, Input
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



def fileopen(file_name = ""):
    '''
    fileopen
    csvファイルを読み込み，読み込んだデータをnparrayとして出力する
    '''
    f = csv.reader(open(file_name,"r")) # open ,make reader
    dt = [ v for v in f]

    for x in range(len(dt)):
        dt[x] = [np.nan if i == "*" else i for i in dt[x] ]
    dt_np = np.array(dt, dtype = np.float32)

    return dt_np



def cal_peak_index(times,s,e):
    '''
    cal_peak_index
    生データ(txtファイル)から指定した時間(s:ピークの始点,e:ピークの終点)のインデックスを算出する
    '''
    peak_start = s                              # ピーク始点のi番目の値を取得
    peak_start = float(peak_start)              # フロートに型変換
    peak_end = e                                # ピーク終点のi番目の値を取得
    peak_end = float(peak_end)                  # フロートに型変換

    start_differences = np.abs(times.astype(float) - peak_start)    # 時間データとピーク始点の値の差の絶対値
    peak_start_index = np.argmin(start_differences)                 # ピーク始点のインデックス取得
    end_differences = np.abs(times.astype(float) - peak_end)        # 時間データとピーク終点の値の差の絶対値
    peak_end_index = np.argmin(end_differences)                     # ピーク終点のインデックス取得
    peak_len = peak_end_index - peak_start_index                    # ピークのデータ長さ計算

    return  peak_start_index, peak_end_index   # 始点と終点のインデックスを出力


def main():
    '''
    main
    main関数
    '''
    num_learning = 1 # 学習回数

    # 入出力データのファイル名取得
    input_file_name = "./MLData_SPgas//" + SPgas_name + "/input_SPgas_data.csv"
    output_file_name = "./MLData_SPgas//" + SPgas_name + "/output_SPgas_data.csv"

    #入力データの読み込み
    input_data = fileopen(input_file_name)
    output_data = fileopen(output_file_name)

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

    # 未知検知率，誤検知率計算用変数定義
    un_detect_N_sum = 0     # 未知検知数合計
    peak_exist_sum = 0      # ピークが存在する数
    err_detect_N_sum = 0    # 誤検知数合計

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

        # 未知検知率，誤検知率計算用
        correct_flag = y_test1[0,0]                                             # 正解データから確度を取得する
        predict_flag = 1.0 if predict_data[0,0] > CONFIDENCE_THRESHOLD else 0   # 確度閾値(CONFIDENCE_THRESHOLD)を超えていれば予測確度を1とし，それ以下であれば0とする
        predict_datalist.append(predict_data[0])                                # 予測データをリストに追加する
        un_detect_N_sum += int(int(correct_flag) and (not int(predict_flag)))   # 未知検知をカウント(正解データの確度と予測データの確度を論理演算する)
        err_detect_N_sum += int(int(correct_flag) ^ (int(predict_flag)))        # 誤検知をカウント(正解データの確度と予測データの確度を論理演算する)
        peak_exist_sum += correct_flag                                          # ピーク数をカウント

        # 台形積分法
        if int(correct_flag) == 1 and (int(predict_flag)) == 1: # 正解データの確度が1かつ予測データの確度が1のとき
            x_test[column_num] -= x_test[column_num].min()      # 予測データの1波形データから予測データの1波形のデータの最小値を引いてオフセットする

            # 正解データ，予測データの始点，終点時間をそれぞれ取得  
            correct_start_time = y_test1[0,1]                   
            correct_end_time = y_test1[0,2]                     
            predict_start_time = predict_data[0][1]
            predict_end_time = predict_data[0][2]

            # 正解データ，予測データの始点，終点時間のインデックスをそれぞれ取得
            correct_start_index , correct_end_index = cal_peak_index(times=time[column_num,:],s=correct_start_time,e=correct_end_time)
            predict_start_index , predict_end_index = cal_peak_index(times=time[column_num,:],s=predict_start_time,e=predict_end_time)

            # 面積変数初期化
            c_area1 = 0
            p_area1 = 0

            # 台形公式を用いて面積計算(正解データ)
            for i in range(correct_start_index,correct_end_index):
                c_area1 += (x_test[column_num,i] + x_test[column_num,i+1])*(time[column_num,i+1]-time[column_num,i])/2

            c_area2 = (x_test[column_num,correct_start_index] + x_test[column_num,correct_end_index])*(correct_start_time-correct_end_time)/2
            c_result_area = c_area1 - c_area2

            # 台形公式を用いて面積計算(予測データ)
            for i in range(predict_start_index,predict_end_index):
                p_area1 += (x_test[column_num,i] + x_test[column_num,i+1])*(time[column_num,i+1]-time[column_num,i])/2

            p_area2 = (x_test[column_num,correct_start_index] + x_test[column_num,correct_end_index])*(correct_start_time-correct_end_time)/2
            p_result_area = p_area1 - p_area2

            # print("correct台形面積------>" + str(float(c_result_area)))
            # print("predict台形面積------>" + str(float(p_result_area)))
            # print("台形面積相対誤差------>" + str(float((p_result_area-c_result_area)/c_result_area*100)) + "%")

            # 正解の面積，予測の面積，相対誤差をcsvファイルに保存
            with open(r".\MLDATA_SPgas\area_table.csv",'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([float(c_result_area), float(p_result_area), float((p_result_area-c_result_area)/c_result_area*100)])

        # 結果をグラフで表示
        if y_test1[0,0] != 0:   # 確度が0以外であれば

            # 正解データ，予測データをグラフ上で縦棒で表示する設定
            plt.vlines(x=y_test1[0,1:3],ymin=x_test[column_num,:,0].min(),ymax=x_test[column_num,:,0].max(), color= '#08699E', label = 'correct',linewidth=3)
            plt.vlines(x=predict_data[0][1:3],ymin=x_test[column_num,:,0].min(),ymax=x_test[column_num,:,0].max(), color='#C97C15', label='predict',linewidth=3)
            plt.scatter(time[column_num,:],x_test[column_num,:,0],c='#08699E',s=5)

            # グラフの表示範囲計算
            with open(r".\MLData_SPgas\\" + SPgas_name + "\min_max_time.csv", 'r') as f:
                reader = csv.reader(f)
                l = [row for row in reader]
                SPgas_mintime = float(l[0][0])
                SPgas_maxtime = float(l[0][1])
                cutstart_time = SPgas_mintime - (SPgas_maxtime - SPgas_mintime) * 0.1 
                cutend_time = SPgas_maxtime + (SPgas_maxtime - SPgas_mintime) * 0.1 
                if cutstart_time < 0:
                    cutstart_time = 0

            # グラフのラベル，凡例設定
            plt.legend()
            plt.xlim(cutstart_time,cutend_time)
            plt.xlabel('time [s]',fontsize=20)
            plt.ylabel('intensity',fontsize=20)
            plt.title(str(SPgas_name),fontsize=20)
            plt.tight_layout()

            # グラフ表示
            if plt_sw == 1: # グラフ表示を行うかどうか(0:行わない 1:行う) 
                plt.show()
            plt.gca().clear()

    # 未知検知率，誤検知率計算
    un_detect_rate = un_detect_N_sum / peak_exist_sum
    err_detect_rate = err_detect_N_sum / test_data_len

    # 未知検知率，誤検知率(百分率)
    un_detect_percent = round(un_detect_rate * 100, 2)
    err_detect_percent = round(err_detect_rate * 100, 2)

    # 未知検知率，誤検知率表示
    print("\n未知検知率------------> " + str(un_detect_percent) + "%")
    print("誤検知率--------------> " + str(err_detect_percent) + "%")

    # 
    predict_data_np = np.array(predict_datalist)
    print("ytest標準偏差---------> " + str(round(y_test[:,1][y_test[:,0] > 0.5].std(),4)))
    print("predict標準偏差-------> " + str(round(predict_data_np[:,1][predict_data_np[:,0] > 0.8].std(),4)))
    err_s = abs(predict_data_np[:,1][y_test[:,0] == 1] - y_test[:,1][y_test[:,0] == 1])
    err_e = abs(predict_data_np[:,2][y_test[:,0] == 1] - y_test[:,2][y_test[:,0] == 1])
    result_mae = (err_s.mean() + err_e.mean()) / 2
    print("mae-------------------> " + str(round(result_mae,4)))
    correct_data_num_rate = round(100*len(y_test[:,0][y_test[:,0]==1]) / len(y_test[:,0]),1)
    print('正解データ割合--------> ' + str(correct_data_num_rate)+'%')
    data_num = len(y_test[:,0])

    # 面積計算結果をcsvから取得
    area_table = np.loadtxt(r".\MLDATA_SPgas\area_table.csv", delimiter=',')

    # 結果をcsvに保存
    with open(r".\MLDATA_SPgas\result_table.csv",'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([SPgas_name, round(np.mean(area_table,axis=0)[0],4),round(np.mean(area_table,axis=0)[1],4),round(100*(np.mean(area_table,axis=0)[1]-np.mean(area_table,axis=0)[0])/np.mean(area_table,axis=0)[0],4), un_detect_percent, err_detect_percent,correct_data_num_rate])

    # 面積計算結果を表示
    print("平均面積(正解)--------> " + str(round(np.mean(area_table,axis=0)[0],4)))
    print("平均面積(予測)--------> " + str(round(np.mean(area_table,axis=0)[1],4)))
    print("平均絶対相対誤差------> " + str(round(np.mean(abs(area_table),axis=0)[2],2)) + "%")


    # 面積誤差散布図
    plt.scatter(area_table[:,0],area_table[:,2],c='#08699E',s=18)
    # plt.legend()
    plt.ylim(-6,13)
    plt.xlabel('area',fontsize=15)
    plt.ylabel('error ratio[%]',fontsize=15)
    plt.title(str(SPgas_name),fontsize=20)
    # plt.show()
    plt.tight_layout()
    plt.savefig("./image/zoom_range/"+str(SPgas_name) + ".svg", format="svg")

# グラフのスタイル設定
plt.style.use('seaborn-dark-palette')

# 学習を行うかどうか(0:行わない 1:行う)
Learn_sw = 0
# グラフ表示を行うかどうか(0:行わない 1:行う)
plt_sw = 1

# ガス名と出現時間を辞書方式で表記
SPgasdict = {"H2":1.9, "O2":2.6, "N2":4.7, "CO":13.7, "CO2":25.2, "CH4":21.0, "C2H4":27.3, "C2H6":28.8, "C2H2":31.6}
SPgas_name = "CO2"
# H2 N2 C2H4 C2H6 C2H2
# 確度閾値設定
CONFIDENCE_THRESHOLD = 0.9 

# area_tableをリセット
with open(r".\MLDATA_SPgas\area_table.csv",'w',newline='') as f:
        pass
main()

# for SPgas_name in SPgasdict:
#     SPgas_time = SPgasdict[SPgas_name]
#     with open(r".\MLDATA_SPgas\area_table.csv",'w',newline='') as f:
#         pass
#     main()
 