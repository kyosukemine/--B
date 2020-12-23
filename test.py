import numpy as np
import os
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
bbb = 1
print("a{3}i{1}u{0}e{2}o".format())
# def aaa():
#     return 1,2,3
# _,_,_ = aaa()
# print(_)

