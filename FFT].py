import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



col_names = [ "c1","c2","c3","c4","c5"]

df = pd.read_csv(r"experimental_data\1219_kyosuke\EMG\WAV_nmusei_1_kyosuke.txt", names = col_names ,encoding="shift_jis",dtype=object)  #データ読み込み
start_mark = df.index[df["c5"]=="#* M1  "][0]  #開始点のインデックス抽出
# print(start_mark)
df = df.iloc[start_mark:start_mark+122000] #開始点から122秒間のデータを取り出す
df = df[["c1","c2","c3","c4"]] #マーク列をのぞいた４列のみにする

df = df.astype(float) #数値がstrになっているのでfloat変換
df = df.abs()  #絶対値を取り整流化
# plt.plot(df)
# plt.show()
print(df)



import numpy as np
import wave
import struct

# ファイルを読み出し
wavf = r'experimental_data\1219_kyosuke\sound\筋肉１\201219_03.WAV'
wr = wave.open(wavf, 'r')

# waveファイルが持つ性質を取得
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

print("Channel: ", ch)
print("Sample width: ", width)
print("Frame Rate: ", fr)
print("Frame num: ", fn)
print("Params: ", wr.getparams())
print("Total time: ", 1.0 * fn / fr)

# waveの実データを取得し、数値化
data = wr.readframes(wr.getnframes())
wr.close()
x = np.frombuffer(data, dtype=np.int16)

##################
# -*- coding: utf-8 -*-
# ==================================
#
#    Short Time Fourier Trasform
#
# ==================================
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft# , ifft
from scipy import ifft # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read

from matplotlib import pylab as pl

# ======
#  STFT
# ======
"""
x : 入力信号(モノラル)
win : 窓関数
step : シフト幅
"""
def stft(x, win, step):
    l = len(x) # 入力信号の長さ
    N = len(win) # 窓幅、つまり切り出す幅
    M = int(ceil(float(l - N + step) / step)) # スペクトログラムの時間フレーム数
    
    new_x = zeros(N + ((M - 1) * step), dtype = float64)
    new_x[: l] = x # 信号をいい感じの長さにする
    
    X = zeros([M, N], dtype = complex64) # スペクトログラムの初期化(複素数型)
    for m in range(M):
        start = step * m
        X[m, :] = fft(new_x[start : start + N] * win)
    return X

# =======
#  iSTFT
# =======
def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in range(M):
        start = step * m
        ### 滑らかな接続
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2 
    pos = (wsum != 0)
    x_pre = x.copy()
    ### 窓分のスケール合わせ
    x[pos] /= wsum[pos]
    return x


if __name__ == "__main__":
    wavfile = r'experimental_data\1219_kyosuke\sound\筋肉１\201219_03.WAV'
    fs, data = read(wavfile)
    data = data[:,0]
    fftLen = 512 # とりあえず
    win = hamming(fftLen) # ハミング窓
    step = int(fftLen / 4)

    ### STFT
    spectrogram = stft(data, win, step)

    ### iSTFT
    resyn_data = istft(spectrogram, win, step)

    ### Plot
    fig = pl.figure()
    fig.add_subplot(311)
    pl.plot(data)
    pl.xlim([0, len(data)])
    pl.title("Input signal", fontsize = 20)
    fig.add_subplot(312)
    pl.imshow(abs(spectrogram[:, : int(fftLen / 2) + 1].T), aspect = "auto", origin = "lower")
    pl.title("Spectrogram", fontsize = 20)
    fig.add_subplot(313)
    pl.plot(resyn_data)
    pl.xlim([0, len(resyn_data)])
    pl.title("Resynthesized signal", fontsize = 20)
    pl.show()


    
    """

import function
import numpy as np
from matplotlib import pyplot as plt

path = r'experimental_data\1219_kyosuke\sound\筋肉１\201219_03.WAV'                           #ファイルパスを指定
data, samplerate = function.wavload(path)   #wavファイルを読み込む
x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸の作成

# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 90                                # オーバーラップ率

# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = function.ov(data, samplerate, Fs, overlap)

# ハニング窓関数をかける
time_array, acf = function.hanning(time_array, Fs, N_ave)

# FFTをかける
fft_array, fft_mean, fft_axis = function.fft_ave(time_array, samplerate, Fs, N_ave, acf)

# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T

# ここからグラフ描画
# グラフをオブジェクト指向で作成する。
fig = plt.figure()
ax1 = fig.add_subplot(111)

# データをプロットする。
im = ax1.imshow(fft_array, \
                vmin = 0, vmax = 160,
                extent = [40, final_time, 0, samplerate], \
                aspect = 'auto',\
                cmap = 'jet')

# カラーバーを設定する。
cbar = fig.colorbar(im)
cbar.set_label('SPL [dBA]')

# 軸設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# スケールの設定をする。
ax1.set_xticks(np.arange(0, 120, 50))
ax1.set_yticks(np.arange(0, 2000, 200))
ax1.set_xlim(40, 120)
ax1.set_ylim(0, 1200)

# グラフを表示する。
plt.show()
plt.close()
"""""