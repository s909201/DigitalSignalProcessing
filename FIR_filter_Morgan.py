# Python program for evaluating DSP algorithm
# by Morgan
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from scipy.fftpack import fft, ifft

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
for i in range(len(data)):  # len(data)為資料行數
    for j in range(len(list(data[i].split()))):  # len(list(data[i].split()))為資料列數
        eachdata.append(data[i].split(" ")[j])
file.close()
# print(eachdata) # for debugging

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
    RawDataIn.append((int(result1) + 128) * 16)

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
# Create a FIR filter and apply it to x.
# ---------------------------------------------------------------
sample_rate = 22050.0  # Hz
nsamples = 28472  # 400 points = 4 sec
t = arange(nsamples) / sample_rate
# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0
# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 40.0 / nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
print("N=", N)
print("beta=", beta)

# The cutoff frequency of the filter.
cutoff_hz = 50.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz / nyq_rate, window=("kaiser", beta))
# print(type(taps))  #<class 'numpy.ndarray'>
print("Taps Len = ", len(taps))
# Use lfilter to filter x with the FIR filter.
RawDataInArray = np.array(RawDataIn)
filtered_x = lfilter(taps, 1.0, RawDataInArray)
# ---------------------------------------------------------------
# Filter Data Array Reconstruction
# ---------------------------------------------------------------
eachdata1 = []
# filter the ',' symbol
for i in range(idx_start):  # idx_start= 17, 0~16
    eachdata1.append(eachdata[i])
eachdata1.append(eachdata[idx_start])  # for "{"

DigitalNumberStartPosition = idx_start + 1  # DigitalNumberStartPosition=18

# the first number of filter should be copied again
eachdata1.append(str(int(filtered_x[N])) + ",")

for i in range(
    DigitalNumberStartPosition, len(eachdata) - 2 - N, 1
):  # eachdata = 28472 - 2 - 1995 = 26475
    if (i % 8) == 0:
        eachdata1.append(
            # str(int(filtered_x[i - DigitalNumberStartPosition])) + ",\n"
            str(int(filtered_x[i - DigitalNumberStartPosition + N - 1]))
            + ",\n"
        )
    else:
        # eachdata1.append(str(int(filtered_x[i - DigitalNumberStartPosition])) + ",")
        eachdata1.append(
            str(int(filtered_x[i - DigitalNumberStartPosition + N - 1])) + ","
        )
eachdata1.append("};")
# ---------------------------------------------------------------
# Output Filtered data
# ---------------------------------------------------------------
NameFilterStr1 = "f2"
FileNameOut = FileName[:-2] + "_" + NameFilterStr1 + ".h"
print("File output Name: ", FileNameOut)
fp = open(FileNameOut, "w")
StrData = []  # Clean Data Buffer
# modify the name of parameters
eachdata1[3] = eachdata1[3] + NameFilterStr1  # sampleRate -> sampleRatefx
eachdata1[9] = eachdata1[9] + NameFilterStr1  # sampleCount -> sampleCountfx
# Change data number
eachdata1[11] = str(int(len(filtered_x) - N)) + ";\n"
print("filtered data len = ", eachdata1[11])

eachdata1[15] = eachdata1[15][:-2] + NameFilterStr1 + "[]"  # sample[] -> samplef1[]
StrData = " ".join(eachdata1)
fp.write(StrData)
fp.close()
# ---------------------------------------------------------------
# Plot the FIR filter coefficients.
# ---------------------------------------------------------------
figure(1)
plot(taps, "b-", linewidth=2)
title("Filter Coefficients (%d taps)" % N)
grid(True)
# ---------------------------------------------------------------
# Plot the magnitude response of the filter.
# ---------------------------------------------------------------
figure(2)
clf()
w, h = freqz(taps, worN=8000)  # equal to FFT
plot((w / pi) * nyq_rate, absolute(h), linewidth=2)
xlabel("Frequency (Hz)")
ylabel("Gain")
title("Frequency Response")
ylim(-0.05, 1.05)
grid(True)
# ---------------------------------------------------------------
# Plot the original and filtered signals.
# ---------------------------------------------------------------
# The phase delay of the filtered signal.
delay = 0.5 * (N - 1) / sample_rate
print("delay= ", delay)

figure(3)
# Plot the original signal.
plt.subplot(311)
plot(t, RawDataInArray)
# Plot the filtered signal, shifted to compensate for the phase delay.
plt.subplot(312)
plot(t - delay, filtered_x, "r-")
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plt.subplot(313)
plot(t[N - 1 :] - delay, filtered_x[N - 1 :], "g", linewidth=4)
xlabel("t")
grid(True)
# ---------------------------------------------------------------
# calculate FFT
# ---------------------------------------------------------------
FFTDataShowNum = 200
RawDataShowNum = np.arange(len(RawDataIn))
FFTPlotX = np.arange(len(RawDataIn))  # x-axis, frequency
FFTOutputY = abs(fft(RawDataIn)) / len(RawDataIn)  # 歸一化處理
FFTPlotY = FFTOutputY[range(FFTDataShowNum)]
# after Filter
FilterFFTPlotX = np.arange(len(filtered_x[N:]))  # x-axis, frequency
FilterFFTOutputY = abs(fft(filtered_x[N:])) / len(filtered_x[N:])  # 歸一化處理
FilterFFTPlotY = FilterFFTOutputY[range(FFTDataShowNum)]

# plot
figure(4)
plt.subplot(411)
plt.plot(RawDataShowNum, RawDataIn)  # show all
plt.title("Raw Data")
plt.subplot(412)
# don't show the first data because it is DC power.
plt.plot(FFTPlotX[1:FFTDataShowNum], FFTPlotY[1:FFTDataShowNum], "b")
plt.title("Raw FFT Result", color="#F08080")
# after Filter
FilterDataShowNum = np.arange(len(filtered_x[N:]))
plt.subplot(413)
plt.plot(FilterDataShowNum, filtered_x[N:])  # show filtered data
plt.title("Filtered Data")
plt.subplot(414)
plt.plot(FilterFFTPlotX[1:FFTDataShowNum], FilterFFTPlotY[1:FFTDataShowNum], "b")
plt.title("Filter data FFT Result", color="#F08080")
show()
