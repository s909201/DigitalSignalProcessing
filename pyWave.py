# Refer: http://bigsec.net/b52/scipydoc/wave_pyaudio.html
# Plot Wave Form of wave file
# by Morgan

# -*- coding: utf-8 -*-
import wave
import pylab as pl
import numpy as np

# 打開WAV文檔
f = wave.open(r"c:\WINDOWS\Media\ding.wav", "rb")

# 讀取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

# 讀取波形數據
str_data = f.readframes(nframes)
f.close()

# 將波形數據轉換為數組
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
wave_data = wave_data.T
time = np.arange(0, nframes) * (1.0 / framerate)

# 繪制波形
pl.subplot(211)
pl.plot(time, wave_data[0])
pl.subplot(212)
pl.plot(time, wave_data[1], c="g")
pl.xlabel("time (seconds)")
pl.show()

# ---------------------------------------------------------------
# Generate 100Hz~1KHz wave data
"""
# -*- coding: utf-8 -*-
import wave
import numpy as np
import scipy.signal as signal

framerate = 44100
time = 10

# 產生10秒44.1kHz的100Hz - 1kHz的頻率掃描波
t = np.arange(0, time, 1.0/framerate)
wave_data = signal.chirp(t, 100, time, 1000, method='linear') * 10000
wave_data = wave_data.astype(np.short)

# 打開WAV文檔
f = wave.open(r"sweep.wav", "wb")

# 配置聲道數、量化位數和取樣頻率
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(framerate)
# 將wav_data轉換為二進制數據寫入文件
f.writeframes(wave_data.tostring())
f.close()
"""
# ---------------------------------------------------------------
# Play wave data
"""
# -*- coding: utf-8 -*-
import pyaudio
import wave

chunk = 1024

# wf = wave.open(r"ding.wav", "rb")
wf = wave.open(r"c:\WINDOWS\Media\ding.wav", "rb")
# wf = wave.open(r"C:\Pythonwork\sweep.wav", "rb")

p = pyaudio.PyAudio()

# 打開聲音輸出流
stream = p.open(
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True,
)

# 寫聲音輸出流進行播放
while True:
    data = wf.readframes(chunk)
    if data == "":
        break
    stream.write(data)

stream.close()
p.terminate()
"""
# ---------------------------------------------------------------
# Play MP3 Music
# -*- coding: utf-8 -*-
"""
import pygame
import time

mp3file = r'C:\MP3\StarSky.mp3'
pygame.mixer.init()
print("Play Music")
track = pygame.mixer.music.load(mp3file)
pygame.mixer.music.play()
time.sleep(10)
pygame.mixer.music.stop()
"""
# ---------------------------------------------------------------
