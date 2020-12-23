
################################
#  wavの読み込み
##############################
from scipy.io.wavfile import read
wavf = r'experimental_data/1219_kyosuke/sound/筋肉１/201219_03.WAV'
rate, data = read(wavf)
time = np.arange(0,len(data)/rate,1/rate)
# plt.plot(data)
# plt.scatter(time,r_data)
# plt.xticks(np.arange(0, int(len(data)/rate), 50))
# ax1.set_yticks(np.arange(0, 2000, 200))
# plt.xlim(0, len(data))
# ax1.set_ylim(0, 1200)
# plt.show()
##################################





# import numpy as np
# import wave
# import struct

# # ファイルを読み出し
# wavf = r'experimental_data/1219_kyosuke/sound/筋肉１/201219_03.WAV'
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
