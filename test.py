import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from tqdm import tqdm
from scipy import signal


# フォルダからtxtファイル読み込み
def get_txtfile_path(path):
    """
    フォルダ内の拡張子がtxtのファイルのファイルパスを配列に格納する関数
    """
    # パスのリスト初期化
    filelist = []

    for pathname, dirname, filenames in os.walk(path):

        for filename in filenames:
            # 拡張子がtxtまたはTXTの時
            if (filename.endswith('.TXT') or filename.endswith('.txt'))and(os.path.join(pathname,filename) != ".\experimental_data\EMG_sound.txt"):
                # filelistにファイルパスを格納する
                filelist.append(os.path.join(pathname,filename))

    return filelist

# EMGファイル読み込み
def file_read(path):
    col_names = [ "c1","c2","c3","c4","c5"]

    df = pd.read_csv(path, names = col_names ,encoding="shift_jis",dtype=object)  #データ読み込み
    # ファイル名print()
    if file_print_frg == 1 :
        print(path)
    start_mark = df.index[df["c5"]=="#* M1  "][0]  #開始点のインデックス抽出
    df = df.iloc[start_mark:] #開始点からのデータを取り出す
    df = df[["c1","c2","c3","c4"]] #マーク列をのぞいた４列のみにする
    df = df.astype(float) #数値がstrになっているのでfloat変換
    df = df.abs()  #絶対値を取り整流化
    nd = df.values # to numpy array
    return nd

# 移動平均
def average(nd,ws=100):
    # ws = windowsize
    v = np.ones(ws)/ws
    y1 = np.convolve(nd[:,0],v)
    y2 = np.convolve(nd[:,1],v)
    y3 = np.convolve(nd[:,2],v)
    y4 = np.convolve(nd[:,3],v)
    nd = np.vstack([y1,y2,y3,y4]).T
    return nd

######################################
# フラグ
file_print_frg = 0
avrage_frg = 1
status_comment_frg = 0
plotfrg = 1
one_roop_plt_frg = 0
mean_print_frg = 1
#######################################

# nd = np.array([[1,2,3,4,5,6],[2,67,1,5,7,5]])
# print(nd[1][3:6])
# a1 = np.array([1,2,3,4,56])
# a2 = np.array([1,2,3,4,56])
# aaa = np.hstack([a1, a2])
# print(np.hstack([a1, a2]))
# aaa = ["a",1]
# print(aaa)
# input_file = "TICC用データ.csv"
# df = np.loadtxt(input_file, delimiter=",",encoding="utf-8_sig")
# print(df)
# np.savetxt( "Results.txt", df, delimiter=',')
# def name_to_num(name):
#     namelist_txt = "name_list.txt"
#     names = np.loadtxt(namelist_txt,delimiter=',', dtype=object)
#     num = None
#     for i in range(len(names)):
#         if names[i,0] == name:
#             num = int(names[i,1])
#     return num
# path = "experimental_data/1219_kyosuke/EMG/WAV_omusei_{}_kyosuke.txt".format(1)
# path = path.split('/')[-1]
# # print(path)
# # s_list = path.split('_')
# # no_voice = s_list[1]

# # n_o = no_voice[0]
# # if n_o == 'n':
# #     n_o = 0
# # else:
# #     n_o = 1

# # voice = no_voice[1]
# # if voice == 'm':
# #     voice = 0
# # else:
# #     voice = 1

# # muscle = int(s_list[2])
# # name_txt = s_list[3]
# # name = name_txt.split('.')[0]
# # print(name)
# # name = name_to_num(name)
# # print([n_o,voice,muscle,name])

# def status(path):
#     s_list = path.split('_')
#     print(s_list)
#     no_voice = s_list[1]

#     n_o = no_voice[0]
#     if n_o == 'n':
#         n_o = 0
#     else:
#         n_o = 1
    
#     voice = no_voice[1]
#     if voice == 'm':
#         voice = 0
#     else:
#         voice = 1
    
#     muscle = int(s_list[2])
#     name_txt = s_list[3]
#     name = name_txt.split('.')[0]
#     name = name_to_num(name)
#     return [n_o,voice,muscle,name]
# print(status(path))

# def get_txtfile_path(path):
#     """
#     フォルダ内の拡張子がtxtのファイルのファイルパスを配列に格納する関数
#     """
#     # パスのリスト初期化
#     filelist = []

#     for pathname, dirname, filenames in os.walk(path):

#         for filename in filenames:
#             # 拡張子がtxtまたはTXTの時
#             if filename.endswith('.TXT') or filename.endswith('.txt'):
#                 # filelistにファイルパスを格納する
#                 filelist.append(os.path.join(pathname,filename))

#     return filelist
# print(get_txtfile_path("./experimental_data"))
# bbb = 1
# print("a{3}i{1}u{0}e{2}o".format())
# def aaa():
#     return 1,2,3
# _,_,_ = aaa()
# print(_)
# def num_to_name(num):
#     namelist_txt = "name_list.txt"
#     names = np.loadtxt(namelist_txt,delimiter=',', dtype=object)
#     print(names)
#     name = None
#     for i in range(len(names)):
#         print(num)
#         print(names[i,1])
#         if int(names[i,1]) == num:
#             print(names[i,0])
#             name = names[i,0]
#     return name
# print(num_to_name(0))

# a = [[[1,2,3,4,5,6,7],[1,5,12,4,5]],[9]]
# b= []
# b.extend(a)
# b.extend(a)
# print(b)

# df = pd.read_csv('boin_all_table.csv',header=None,delimiter=',',dtype='float64')
# # print(df)
# # for i in range(100,130,10):
# #     plt.plot(df.iloc[i,6:])
# # plt.show()

# target_vector = df.iloc[:,0].astype(int)               # クラス分類を整数値のベクトルで表現したもの
# print(target_vector)
# n_labels = len(np.unique(target_vector))  # 分類クラスの数 = 5
# print(np.eye(n_labels)[target_vector])           # one hot表現に変換

# def fileopen(file_name = ""):
#     '''
#     fileopen
#     csvファイルを読み込み，読み込んだデータをnparrayとして出力する
#     '''
#     df = pd.read_csv(file_name,header=None,delimiter=',',dtype='float64')

#     target_vector = df.iloc[:,0].astype(int)               # クラス分類を整数値のベクトルで表現したもの
#     n_labels = len(np.unique(target_vector))  # 分類クラスの数 = 5
#     output_data = np.eye(n_labels)[target_vector]           # one hot表現に変換
#     output_data = output_data
#     input_data = df.iloc[:,7:]
#     input_data = input_data.values
#     return input_data,output_data

# print(fileopen('boin_all_table.csv'))


# df = pd.read_csv('boin_all_table.csv',header=None,delimiter=',',dtype='float64')
# input_data = []
# output_data = []
# for i in np.unique(df.iloc[:,6].astype(int)):
#     input_data.append(df[df.iloc[:,6] == i].iloc[:,7:].mean(axis=1).values)
#     output_data.append(df[df.iloc[:,6] == i].iloc[0,0])

# input_data = np.array(input_data)
# output_data = np.array(output_data).astype(int)

# n_labels = len(np.unique(output_data))  # 分類クラスの数 = 5
# output_data = np.eye(n_labels)[output_data]           # one hot表現に変換

# print(input_data)
# print(input_data.shape)

# print(output_data)
# print(output_data.shape)

# df = pd.read_csv('boin_all_table.csv',header=None,delimiter=',',dtype='float64')
# input_data = []
# output_data = []
# for i in np.unique(df.iloc[:,6].astype(int)):
#     input_data.append(df[df.iloc[:,6] == i].iloc[:,7:].values)
#     output_data.append(df[df.iloc[:,6] == i].iloc[0,0])

# input_data = np.array(input_data)
# output_data = np.array(output_data).astype(int)

# n_labels = len(np.unique(output_data))  # 分類クラスの数 = 5
# output_data = np.eye(n_labels)[output_data]           # one hot表現に変換

# select_data_num_0 = 0
# select_data_num_1 = 1
# select_data_num_2 = 2
# select_data_num_3 = 3
# select_data_num_4 = 4

# print(input_data[select_data_num_1,:,:500].T)
# print(input_data[select_data_num_1,:,:].T.shape)

# print(output_data[select_data_num_1,:])
# print(output_data.shape)

# print(np.tile(output_data[select_data_num_1,:], (500, 1)))
# c = input_data[select_data_num_1,:,0:1000].T
# # b = input_data[select_data_num_1,:,500:1000].T

# # c = np.vstack((a,b))
# print(c)
# print(c.shape)

# path = 
nd = file_read(r'C:\Users\kobayashi kyosuke\OneDrive - 埼玉大学\卒研B\workspace\experimental_data\0107_kaito\EMG\WAV_omusei_2_kaito.txt')
plt.figure()
plt.plot(nd[:,2])



# 時系列のサンプルデータ作成
n = len(nd[:,1])                         # データ数
dt = 0.001                       # サンプリング間隔
f = 1000                           # 周波数
fn = 1/(2*dt)                   # ナイキスト周波数
y = nd[:,1]

# パラメータ設定
fp = 10                          # 通過域端周波数[Hz]
fs = 20                          # 阻止域端周波数[Hz]
gpass = 1                       # 通過域最大損失量[dB]
gstop = 40                      # 阻止域最小減衰量[dB]
# 正規化
Wp = fp/fn
Ws = fs/fn

# ローパスフィルタで波形整形
# バターワースフィルタ
N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
b1, a1 = signal.butter(N, Wn, "low")
y1 = signal.filtfilt(b1, a1, y)
plt.figure()
plt.plot(y1)
plt.show()