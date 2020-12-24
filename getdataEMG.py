import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from tqdm import tqdm




status = []
boin_a = []
boin_i = []
boin_u = []
boin_e = []
boin_o = []

siin_k = []
siin_s = []
siin_t = []
siin_n = []
siin_h = []
siin_m = []
siin_j = []
siin_r = []
siin_w = []
siin_g = []
siin_z = []
siin_d = []
siin_b = []
siin_p = []
siin_N = []
siin_kja = []
siin_kju = []
siin_kjo = []


# 名前をnumに変換
def name_to_num(name):
    namelist_txt = "name_list.txt"
    names = np.loadtxt(namelist_txt,delimiter=',', dtype=object)
    num = None
    for i in range(len(names)):
        if names[i,0] == name:
            num = int(names[i,1])
    return num

# numを名前に変換
def num_to_name(num):
    namelist_txt = "name_list.txt"
    names = np.loadtxt(namelist_txt,delimiter=',', dtype=object)
    name = None
    for i in range(len(names)):
        if int(names[i,1]) == num:
            name = names[i,0]
    return name

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
    df = df.iloc[start_mark+10000:] #開始点からのデータを取り出す
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

# status 取得
def get_status(path):
    path = path.split('\\')[-1]
    s_list = path.split('_')
    no_voice = s_list[1]
    n_o = no_voice[0]
    if n_o == 'n':
        n_o = 0
    else:
        n_o = 1
    voice = no_voice[1]
    if voice == 'm':
        voice = 0
    else:
        voice = 1
    muscle = int(s_list[2])
    name_txt = s_list[3]
    name = name_txt.split('.')[0]
    name_num = name_to_num(name)
    if status_comment_frg == 1:# フラグが1だったらステータスを見せる
        print("自然か誇張か(0=自然，1=誇張)------->{}".format(['自然','誇張'][n_o]))
        print("発声の有無(0=無し，1=有)----------->{}".format(['無','有'][voice]))
        print("筋肉------------------------------->{}".format(['筋肉１','筋肉２'][muscle-1]))
        print("名前------------------------------->{}".format(name))
    return [n_o,voice,muscle,name_num]

# 4つの筋電を別々にプロット 1ループ
def plot(nd,i,status):
    """複数のグラフを並べて描画するプログラム"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    if i == 0:
        global fig 
        fig = plt.figure(figsize=(15, 10), dpi=80)
    fig.suptitle("{} {} {} {}".format(num_to_name(status[3]),['無','有'][status[1]],
    ['自然','誇張'][status[0]],['筋肉１','筋肉２'][status[2]-1]), fontname="MS Gothic")

    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(5, 4, 1+i*4)
    ax2 = fig.add_subplot(5, 4, 2+i*4)
    ax3 = fig.add_subplot(5, 4, 3+i*4)
    ax4 = fig.add_subplot(5, 4, 4+i*4)
    
    y1 = nd[:,0]
    y2 = nd[:,1]
    y3 = nd[:,2]
    y4 = nd[:,3]

    c1,c2,c3,c4 = "blue","green","red","black"      # 各プロットの色
    l1,l2,l3,l4 = "1","2","3","4"   # 各ラベル

    ax1.plot(y1, color=c1, label=l1)
    ax2.plot(y2, color=c2, label=l2)
    ax3.plot(y3, color=c3, label=l3)
    ax4.plot(y4, color=c4, label=l4)
    # ax1.legend(loc = 'upper right') #凡例
    # ax2.legend(loc = 'upper right') #凡例
    # ax3.legend(loc = 'upper right') #凡例
    # ax4.legend(loc = 'upper right') #凡例
    s_lines = [i*2000 +1000 -1 for i in range(5)]
    # e_lines = [i*2000 +2000 -1 for i in range(5)]
    for j in s_lines:
        ax1.axvspan(j, j+1000, color = c1, alpha=0.1)
        ax2.axvspan(j, j+1000, color = c2, alpha=0.1)
        ax3.axvspan(j, j+1000, color = c3, alpha=0.1)
        ax4.axvspan(j, j+1000, color = c4, alpha=0.1)
    # ax1.vlines(lines, ymin = y1.min(), ymax = y1.max())
    # ax2.vlines(lines, ymin = y2.min(), ymax = y2.max())
    # ax3.vlines(lines, ymin = y3.min(), ymax = y3.max())
    # ax4.vlines(lines, ymin = y4.min(), ymax = y4.max())
    fig.tight_layout()              #レイアウトの設定

    if i == 4:
        if one_roop_frg == 1:# frg 1 だったら1枚づつplot
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
            plt.show()
            plt.close()

# 音素別にカット 1roop
def cut(nd,status,onnso_0,onnso_1,onnso_2,onnso_3,onnso_4):

    for i in range(4):# 筋肉の部位をiとして先頭に記録
        # [筋肉部位，n_o,voice,muscle,name,~]
        j = i
        if status[2] == 2:
            j += 4
        onnso_0.append([j] + status + np.squeeze(nd[1000:2000,i]).tolist())
        onnso_1.append([j] + status + np.squeeze(nd[3000:4000,i]).tolist())
        onnso_2.append([j] + status + np.squeeze(nd[5000:6000,i]).tolist())
        onnso_3.append([j] + status + np.squeeze(nd[7000:8000,i]).tolist())
        onnso_4.append([j] + status + np.squeeze(nd[9000:10000,i]).tolist())
    
    # return boin_a,boin_i,boin_u,boin_e,boin_o

# 平均算出
def mean(boin):# 筋肉別の平均値算出
    boin = np.array(boin)# numpy配列に変換
    boin_0 = boin[boin[:,0] == 0][:,5:] # statusを除いた値を出す
    boin_1 = boin[boin[:,0] == 1][:,5:]
    boin_2 = boin[boin[:,0] == 2][:,5:]
    boin_3 = boin[boin[:,0] == 3][:,5:]
    boin_4 = boin[boin[:,0] == 4][:,5:]
    boin_5 = boin[boin[:,0] == 5][:,5:]
    boin_6 = boin[boin[:,0] == 6][:,5:]
    boin_7 = boin[boin[:,0] == 7][:,5:]


    boin_0_mean = boin_0.mean() # numpyで全体の平均値を出している(行ごとに出したい場合はaxis=1)
    boin_1_mean = boin_1.mean()
    boin_2_mean = boin_2.mean()
    boin_3_mean = boin_3.mean()
    boin_4_mean = boin_4.mean()
    boin_5_mean = boin_5.mean()
    boin_6_mean = boin_6.mean()
    boin_7_mean = boin_7.mean()
    if mean_print_frg == 1 :# 平均値プリントするかどうか
        print('{:.5f}'.format(boin_0_mean),'{:.5f}'.format(boin_1_mean),
        '{:.5f}'.format(boin_2_mean),'{:.5f}'.format(boin_3_mean),'{:.5f}'.format(boin_3_mean),
        '{:.5f}'.format(boin_5_mean),'{:.5f}'.format(boin_6_mean),'{:.5f}'.format(boin_7_mean))
    return [boin_0_mean,boin_1_mean,boin_2_mean,boin_3_mean,boin_4_mean,boin_5_mean,boin_6_mean,boin_7_mean]

# 1ファイルに対して
def one_file(path):

    nd = file_read(path) # 1人分の1試行分のデータ(母音のスライドのみ)をnd入れる
    status = get_status(path) # ステータスを取得
    nd = average(nd,100) # 100msで移動平均化
    
    
    for i in range(5): # 1/roop
        if plotfrg == 1:# フラグが1だったらプロット
            plot(nd[i*12000+1000:i*12000+12000],i,status) # 1つのパスのデータをplot
        cut(nd[i*12000+1000:i*12000+12000],status,boin_a,boin_i,boin_u,boin_e,boin_o) # 母音別にカット


# csv save
def save_csv(csv_list,file_name):
    with open(file_name, 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
    f.close()

# status = [name, muscle,phoneme,voice]

######################################
# フラグ
file_print_frg = 0
status_comment_frg = 0
plotfrg = 1
one_roop_frg = 0
mean_print_frg = 0
#######################################


def main():
    path_list = get_txtfile_path(".\\experimental_data")
    for i in tqdm(path_list):
        one_file(i)
    if one_roop_frg == 0:
        wm = plt.get_current_fig_manager()
        wm.window.state('zoomed')
        plt.show()
    boin_list = [boin_a,boin_i,boin_u,boin_e,boin_o]

    mean_table = []
    for i in boin_list:
        mean_table.append(mean(i))

    save_csv(mean_table,'mean_table.csv')


main()











############################################################################
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
############################################################################

