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
##############################
from scipy.io.wavfile import read
wavf = r'experimental_data\1219_kyosuke\sound\筋肉１\201219_03.WAV'
rate, data = read(wavf)
time = np.arange(0,len(data)/rate,1/rate)
plt.plot(data)
# plt.scatter(time,r_data)
# plt.xticks(np.arange(0, int(len(data)/rate), 50))
# ax1.set_yticks(np.arange(0, 2000, 200))
# plt.xlim(0, len(data))
# ax1.set_ylim(0, 1200)
plt.show()
##################################
# import numpy as np
# import wave
# import struct

# # ファイルを読み出し
# wavf = r'experimental_data\1219_kyosuke\sound\筋肉１\201219_03.WAV'
# wr = wave.open(wavf, 'r')

# # waveファイルが持つ性質を取得
# ch = wr.getnchannels()
# width = wr.getsampwidth()
# fr = wr.getframerate()
# fn = wr.getnframes()

# print("Channel: ", ch)
# print("Sample width: ", width)
# print("Frame Rate: ", fr)
# print("Frame num: ", fn)
# print("Params: ", wr.getparams())
# print("Total time: ", 1.0 * fn / fr)

# # waveの実データを取得し、数値化
# data = wr.readframes(wr.getnframes())
# wr.close()
# x = np.frombuffer(data, dtype=np.int16)
# print(x)
# plt.plot(x)
# plt.show()
# l_channel = data[::2]
# r_channel = data[1::2]
# print(l_channel,r_channel)
# data = l_channel
# x = np.frombuffer(data, dtype=np.int16)

# print(len(x))
# plt.plot(x)
# plt.show()