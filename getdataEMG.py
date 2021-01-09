import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from tqdm import tqdm
from scipy import signal
from numpy import array, sign, zeros
from numpy import array, sign, zeros
from scipy.interpolate import interp1d

# envelope = abs(signal.hilbert(data))


boin_all = []
cnt = 0
status = []
boin_a = []
boin_i = []
boin_u = []
boin_e = []
boin_o = []
boin_set = [0,1,2,3,4]

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
                if filename == 'WAV_oyuusei_1_suzuki_siinn.txt':
                    continue
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
    if "kyosuke" in path: # 自分は10秒の遅れをしている
        start_mark += 10000
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

def low_pass(nd):
    # 時系列のサンプルデータ作成
    n = len(nd)                         # データ数
    dt = 0.001                       # サンプリング間隔
    f = 1000                           # 周波数
    fn = 1/(2*dt)                   # ナイキスト周波数
    y = nd

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
    return y1


def envelope(nd,ws):
    from numpy import array, sign, zeros
    from scipy.interpolate import interp1d


    s = nd #This is your noisy vector of values.

    q_u = zeros(s.shape)


    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0,]
    u_y = [s[0],]


    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.
    lis = [ i for i in range(ws,len(s)-1,ws)]
    

    for k in lis:
            
        u_x.append(s[k-ws:k].argmax()+k-ws)
        u_y.append(s[k-ws:k].max())


    #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(s)-1)
    u_y.append(s[-1])



    #Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = interp1d(u_x,u_y, kind = 'linear',bounds_error = False, fill_value=0.0)


    #Evaluate each model over the domain of (s)
    for k in range(0,len(s)):
        q_u[k] = u_p(k)

    return q_u


# 4つの筋電を別々にプロット 1ループ
def plot(nd,i,status,ws):
    """複数のグラフを並べて描画するプログラム"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    if i == 0:
        global fig 
        fig = plt.figure(figsize=(15, 10), dpi=80)
    # if avrage_frg == 1:
    fig.suptitle("{} {} {} {} {}".format(num_to_name(status[3]),['無','有'][status[1]],
    ['自然','誇張'][status[0]],['筋肉１','筋肉２'][status[2]-1],str(window_size) + 'ms'), fontname="MS Gothic",fontsize=20)
    # else:
    #     fig.suptitle("{} {} {} {}".format(num_to_name(status[3]),['無','有'][status[1]],
    #     ['自然','誇張'][status[0]],['筋肉１','筋肉２'][status[2]-1]), fontname="MS Gothic",fontsize=20)
    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所

    # 列方向に同じ筋肉部位を並べる
    # ax1 = fig.add_subplot(5, 4, 1+i*4)
    # ax2 = fig.add_subplot(5, 4, 2+i*4)
    # ax3 = fig.add_subplot(5, 4, 3+i*4)
    # ax4 = fig.add_subplot(5, 4, 4+i*4)

    # 行方向に同じ筋肉部位を並べる
    ax1 = fig.add_subplot(4, 5, 1+i)
    ax2 = fig.add_subplot(4, 5, 6+i)
    ax3 = fig.add_subplot(4, 5, 11+i)
    ax4 = fig.add_subplot(4, 5, 16+i)
    

    y1 = nd[:,0]
    y2 = nd[:,1]
    y3 = nd[:,2]
    y4 = nd[:,3]

    c1,c2,c3,c4 = "blue","green","red","black"      # 各プロットの色
    l1,l2,l3,l4 = "1","2","3","4"   # 各ラベル
    if row_data_plot_frg ==1 :
        ax1.plot(y1, color=c1, label=l1)
        ax2.plot(y2, color=c2, label=l2)
        ax3.plot(y3, color=c3, label=l3)
        ax4.plot(y4, color=c4, label=l4)


    if envelope_frg == 1 :

        # envelope_y1 = abs(signal.hilbert(y1))
        # envelope_y2 = abs(signal.hilbert(y2))
        # envelope_y3 = abs(signal.hilbert(y3))
        # envelope_y4 = abs(signal.hilbert(y4))


        # envelope_y1 = low_pass(y1)
        # envelope_y2 = low_pass(y2)
        # envelope_y3 = low_pass(y3)
        # envelope_y4 = low_pass(y4)
        

        envelope_y1 = envelope(y1,ws)
        envelope_y2 = envelope(y2,ws)
        envelope_y3 = envelope(y3,ws)
        envelope_y4 = envelope(y4,ws)

        ax1.plot(envelope_y1, color=c1, label=l1)
        ax2.plot(envelope_y2, color=c2, label=l2)
        ax3.plot(envelope_y3, color=c3, label=l3)
        ax4.plot(envelope_y4, color=c4, label=l4)

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
        if one_roop_save_frg == 1:
            os.makedirs("./images/svg/{}".format(save_fig_dir), exist_ok=True)
            os.makedirs("./images/png/{}".format(save_fig_dir), exist_ok=True)
            fig.savefig("./images/svg/{4}/{3}{2}{1}{0}.svg".format(num_to_name(status[3]),['無','有'][status[1]],['自然','誇張'][status[0]],['筋肉１','筋肉２'][status[2]-1],save_fig_dir))
            fig.savefig("./images/png/{4}/{3}{2}{1}{0}.png".format(num_to_name(status[3]),['無','有'][status[1]],['自然','誇張'][status[0]],['筋肉１','筋肉２'][status[2]-1],save_fig_dir))
        if one_roop_plt_frg == 1:# frg 1 だったら1枚づつplot
            wm = plt.get_current_fig_manager()
            wm.window.state('zoomed')
            plt.show()
            plt.close()
        if plt_memory_release == 1:
            plt.clf() # メモリ解放のため
            plt.close()

# 音素別にカット 1roop
def cut(nd,status,onnso_0,onnso_1,onnso_2,onnso_3,onnso_4,onnso_set,id):

    for i in range(4):# 筋肉の部位をiとして先頭に記録
        # [b音素番号，筋肉部位，n_o,voice,muscle,name,id,~]
        j = i
        if status[2] == 2:
            j += 4
        onnso_0.append([onnso_set[0]] + [j] + status + [id] + np.squeeze(nd[1000:2000,i]).tolist())# 何か追加したときはmeanの順番に気を付ける
        onnso_1.append([onnso_set[1]] + [j] + status + [id+1] + np.squeeze(nd[3000:4000,i]).tolist())
        onnso_2.append([onnso_set[2]] + [j] + status + [id+2] + np.squeeze(nd[5000:6000,i]).tolist())
        onnso_3.append([onnso_set[3]] + [j] + status + [id+3] + np.squeeze(nd[7000:8000,i]).tolist())
        onnso_4.append([onnso_set[4]] + [j] + status + [id+4] + np.squeeze(nd[9000:10000,i]).tolist())
    
    # return boin_a,boin_i,boin_u,boin_e,boin_o

# 平均算出
def mean(boin):# 筋肉別の平均値算出
    boin = np.array(boin)# numpy配列に変換
    boin_0 = boin[boin[:,1] == 0][:,7:] # statusを除いた値を出す
    boin_1 = boin[boin[:,1] == 1][:,7:]
    boin_2 = boin[boin[:,1] == 2][:,7:]
    boin_3 = boin[boin[:,1] == 3][:,7:]
    boin_4 = boin[boin[:,1] == 4][:,7:]
    boin_5 = boin[boin[:,1] == 5][:,7:]
    boin_6 = boin[boin[:,1] == 6][:,7:]
    boin_7 = boin[boin[:,1] == 7][:,7:]


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
    global cnt
    nd = file_read(path) # 1人分の1試行分のデータ(母音のスライドのみ)をnd入れる
    status = get_status(path) # ステータスを取得
    if avrage_frg == 1:
        nd = average(nd,window_size) # window_sizeで移動平均化
    
    
    for i in range(5): # 1/roop
        if plotfrg == 1:# フラグが1だったらプロット
            plot(nd[i*12000+1000:i*12000+12000],i,status,window_size) # 1つのパスのデータをplot
        cut(nd[i*12000+1000:i*12000+12000],status,boin_a,boin_i,boin_u,boin_e,boin_o,boin_set,cnt) # 母音別にカット
        cnt += 5


# csv save
def save_csv(csv_list,file_name):
    with open(file_name, 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
    f.close()

# status = [name, muscle,phoneme,voice]

######################################
# フラグ

# ターミナルに情報出力
file_print_frg = 1
mean_print_frg = 1

# 平均
avrage_frg = 0
status_comment_frg = 1

# plot関係
plotfrg = 1
row_data_plot_frg = 0
one_roop_plt_frg = 0
one_roop_save_frg = 1
plt_memory_release = 1
all_plt_frg = 0

# 包絡線
envelope_frg = 1
#######################################
save_fig_dir = "envelope_off" #フォルダ作成
save_fig_dir = "envelope_on_ws100" #フォルダ作成
save_fig_dir = "envelope_only_ws100" #フォルダ作成
save_fig_dir = "envelope_on_ws10" #フォルダ作成
window_size = 100

def main():
    path_list = get_txtfile_path(".\\experimental_data")
    for i in tqdm(path_list):
        one_file(i)
    if one_roop_plt_frg == 1:
        plt.show()
    boin_list = [boin_a,boin_i,boin_u,boin_e,boin_o]

    mean_table = []
    for i in boin_list:
        mean_table.append(mean(i))
    boin_all.extend(boin_a)
    boin_all.extend(boin_i)
    boin_all.extend(boin_u)
    boin_all.extend(boin_e)
    boin_all.extend(boin_o)
    save_csv(mean_table,'mean_table.csv')
    save_csv(boin_a,'boin_a_table.csv')
    save_csv(boin_i,'boin_i_table.csv')
    save_csv(boin_u,'boin_u_table.csv')
    save_csv(boin_e,'boin_e_table.csv')
    save_csv(boin_o,'boin_o_table.csv')
    save_csv(boin_all,'boin_all_table.csv')


main()











############################################################################
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
#                                   ゴミ                                   #
############################################################################

