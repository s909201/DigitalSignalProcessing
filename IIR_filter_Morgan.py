# Python program for evaluating DSP algorithm
# by Morgan
import sys, os
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import freqs, iirfilter
# ---------------------------------------------------------------
# Create a signal for demonstration.
# ---------------------------------------------------------------
FileName = "ActrosV8Idle.h"  # modify this .h file name, Row = 1783
os.system("cls")  # clean screen
print(os.getcwd())  # show current working path
try:
    file = open(FileName)
except FileNotFoundError:
    print("Can't find %s file" % FileName)
    sys.exit()

data = file.readlines()  # 逐行讀取txt並存成list。每行是list的一個元素，資料型別為str
print("The Number of Row = ", len(data))

eachdata = []
# Parser
for i in range(len(data)):  # len(data): data length
    for j in range(len(list(data[i].split()))):  # len(list(data[i].split()))為資料列數
        eachdata.append(data[i].split(" ")[j])
file.close()
# modify declare word
for i in range(len(eachdata)):
    if (
        (eachdata[i] == "const")
        and (eachdata[i + 1] == "signed")
        and (eachdata[i + 2] == "char")
    ):
        eachdata[i + 1] = " "
        eachdata[i + 2] = "uint32_t"
        break
# print(eachdata) # for debugging

# Get the starter position of number string
idx_start = -1
for i in range(len(eachdata)):
    if eachdata[i] == "{\n":
        # print(i) # for debugging
        idx_start = i
        break

# filter the ',' symbol
RawDataIn = []  # for FFT
for i in range(idx_start + 1, len(eachdata) - 1, 1):
    result1 = eachdata[i][:-1]
    # print(result1) # for debugging
    if result1[-1] == ",":  # sometimes, some data has different format
        result1 = result1[:-1]
    # print(type(result)) # for debugging
    result2 = str((int(result1) + 128) * 16)
    if i % 8 == 0:
        eachdata[i] = result2 + ",\n"
    else:
        eachdata[i] = result2 + ","
    # print(result1, result2, eachdata[i]) # for debugging
    # for FFT, int format
    RawDataIn.append((int(result1) + 128) * 16)  # type: List

# reconstruct new data array
# print(eachdata) # for debugging
# prepare new file name
FileNameOut = FileName[:-2] + "_m2.h"
print("File output Name: ", FileNameOut)
fp = open(FileNameOut, "w")
StrData = " ".join(eachdata)
fp.write(StrData)
fp.close()
# ---------------------------------------------------------------
# Read Raw data
# ---------------------------------------------------------------
RawDataInArray = np.array(RawDataIn)  # List to Int

sample_rate = 22050.0  # Hz
nsamples = 28472  # 400 points = 4 sec
t = arange(nsamples) / sample_rate
# ---------------------------------------------------------------
# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0
# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 40.0 / nyq_rate

# ---------------------------------------------------------------
# Filter Method Selection
# ---------------------------------------------------------------
OrderOfFilter = 17
Wn1 = 50  # 5K Hz
Wn2 = 500  # 10k Hz
# ---------------------------------------------------------------
# butterworth low pass, cut off=100Hz, fs=22KHz
# ---------------------------------------------------------------
# sos = signal.butter(10, 100, 'lp', fs=sample_rate, output='sos')
# filtered_x = signal.sosfilt(sos, RawDataInArray)

# ---------------------------------------------------------------
# butterworth high pass, cut off = 10KHz, fs=22KHz
# ---------------------------------------------------------------
# sos = signal.butter(10, 10000, 'hp', fs=sample_rate, output='sos')
# filtered_x = signal.sosfilt(sos, RawDataInArray)

# ---------------------------------------------------------------
# butterworth band pass (digital), cut off=100~2KHz, fs=22KHz
# ---------------------------------------------------------------
sos = signal.butter(
    OrderOfFilter, [Wn1, Wn2], btype="bp", fs=sample_rate, output="sos"
)  # type: <class 'numpy.ndarray'>, return Digital filter(analog = false)
filtered_x = signal.sosfilt(sos, RawDataInArray)
N = len(sos)  # suppose filter coefficient data length

# ---------------------------------------------------------------
# butterworth band pass (analog), b, a, w, h style
# ---------------------------------------------------------------
# b, a = signal.butter(9, 0.55)
# w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))
# filtered_x = signal.filtfilt(b, a, RawDataInArray)
# figure(2)
# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude response [dB]")
# plt.grid()
# N = 50 # fake data

# ---------------------------------------------------------------
# Level shift
filtered_x_max = np.max(filtered_x)
filtered_x_min = np.min(filtered_x)
filtered_x_avg = (filtered_x_max + filtered_x_min) / 2
filtered_x2 = (filtered_x - filtered_x_avg) + 2048
# print("filtered_x2=", filtered_x2)
filtered_x2_max = np.max(filtered_x2)
filtered_x2_min = np.min(filtered_x2)
# ---------------------------------------------------------------
# Filter Data Array Reconstruction
eachdata1 = []
# filter the ',' symbol
for i in range(idx_start):  # idx_start= 17, 0~16
    eachdata1.append(eachdata[i])
eachdata1.append(eachdata[idx_start])  # for "{"

DigitalNumberStartPosition = idx_start + 1  # DigitalNumberStartPosition=18

# the first number of filter should be copied again
# eachdata1.append(str(int(filtered_x[N])) + ",")
eachdata1.append(str(int(filtered_x2[N])) + ",")

for i in range(
    DigitalNumberStartPosition, len(eachdata) - 2 - N, 1
):  # eachdata = 28472 - 2 - 1995 = 26475
    if (i % 8) == 0:
        eachdata1.append(
            str(int(filtered_x2[i - DigitalNumberStartPosition + N - 1])) + ",\n"
        )
    else:
        # eachdata1.append(str(int(filtered_x[i - DigitalNumberStartPosition])) + ",")
        eachdata1.append(
            str(int(filtered_x2[i - DigitalNumberStartPosition + N - 1])) + ","
        )

# eachdata1.append(eachdata[len(eachdata) - 1])
eachdata1.append("};")
# ---------------------------------------------------------------
# Output filtered_x2 data
NameFilterStr1 = "f2"
FileNameOut = FileName[:-2] + "_" + NameFilterStr1 + ".h"
print("File output Name: ", FileNameOut)
fp = open(FileNameOut, "w")
StrData = []  # Clean Data Buffer
# modify the name of parameters
eachdata1[3] = eachdata1[3] + NameFilterStr1  # sampleRate -> sampleRatefx
eachdata1[9] = eachdata1[9] + NameFilterStr1  # sampleCount -> sampleCountfx
# Change data number
# eachdata1[11] = str(int(len(filtered_x) - N)) + ";\n"
eachdata1[11] = str(int(len(filtered_x2) - N)) + ";\n"
print("filtered_x data len = ", eachdata1[11])

eachdata1[15] = eachdata1[15][:-2] + NameFilterStr1 + "[]"  # sample[] -> samplef1[]
StrData = " ".join(eachdata1)
fp.write(StrData)
fp.close()
# ---------------------------------------------------------------
# Plot the magnitude response of the filter.
# ---------------------------------------------------------------
# The phase delay of the filtered_x signal.
delay = 0.5 * (N - 1) / sample_rate
print("delay= ", delay)
# ---------------------------------------------------------------
# calculate FFT
# ---------------------------------------------------------------
FFTDataShowNum = int(len(RawDataInArray) / 2)
RawDataShowNum = np.arange(len(RawDataIn))
FFTPlotX = np.arange(len(RawDataIn))  # x-axis, frequency
FFTOutputY = abs(fft(RawDataIn)) / len(RawDataIn)  # 歸一化處理
FFTPlotY = FFTOutputY[range(FFTDataShowNum)]
# after Filter
FilterFFTPlotX = np.arange(len(filtered_x2[N:]))  # x-axis, frequency
FilterFFTOutputY = abs(fft(filtered_x2[N:])) / len(filtered_x2[N:])  # 歸一化處理
FilterFFTPlotY = FilterFFTOutputY[range(FFTDataShowNum)]
# ---------------------------------------------------------------
# Plot the original and filtered_x signals.
# ---------------------------------------------------------------
figure(1)
plt.subplot(411)
plt.plot(RawDataShowNum, RawDataIn)  # show all
plt.title("Raw Data")
plt.subplot(412)
# plot FFT
# don't show the first data because it is DC power.
plt.plot(FFTPlotX[1:FFTDataShowNum], FFTPlotY[1:FFTDataShowNum], "b")
plt.title("Raw FFT Result")
# after Filter
FilterDataShowNum = np.arange(len(filtered_x2[N:]))
plt.subplot(413)
plt.plot(FilterDataShowNum, filtered_x2[N:])  # show filtered_x data
plt.title("filtered_x Data")
plt.subplot(414)
plt.plot(FilterFFTPlotX[1:FFTDataShowNum], FilterFFTPlotY[1:FFTDataShowNum], "b")
plt.title("FFT Result for Filtered data")
# ---------------------------------------------------------------
show()
