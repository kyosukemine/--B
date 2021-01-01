import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from tqdm import tqdm
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


df = pd.read_csv('boin_all_table.csv',header=None,delimiter=',',dtype='float64')
input_data = []
output_data = []
for i in np.unique(df.iloc[:,6].astype(int)):
    input_data.append(df[df.iloc[:,6] == i].iloc[:,7:].values)
    output_data.append(df[df.iloc[:,6] == i].iloc[0,0])

input_data = np.array(input_data)
output_data = np.array(output_data).astype(int)

n_labels = len(np.unique(output_data))  # 分類クラスの数 = 5
output_data = np.eye(n_labels)[output_data]           # one hot表現に変換

print(input_data[1,:,:].shape)
print(input_data.shape)

print(output_data)
print(output_data.shape)
