# IIR Filter Design
# by Morgan
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker
# ---------------------------------------------------------------
# scipy.signal.freqs(b, a, worN=200, plot=None)
# Compute frequency response of analog filter.
# Given the M-order numerator b and N-order denominator a of an analog filter, compute its frequency response:
#         b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
# H(w) = ----------------------------------------------
#         a[0] * (jw)**N + a[1] * (jw)**(N - 1) + ... +a[N]
"""
from scipy.signal import freqs, iirfilter

b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype="cheby1")
w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))
import matplotlib.pyplot as plt

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.xlabel("Frequency")
plt.ylabel("Amplitude response [dB]")
plt.grid()
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.iirfilter
# Design an Nth-order digital or analog filter and return the filter coefficients.
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html
b, a = signal.iirfilter(17, [50, 200], rs=60, btype="band", analog=True, ftype="cheby2")
w, h = signal.freqs(b, a, 1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(w, 20 * np.log10(abs(h)))
ax.set_xscale("log")
ax.set_title("Chebyshev Type II bandpass frequency response")
ax.set_xlabel("Frequency [radians / second]")
ax.set_ylabel("Amplitude [dB]")
ax.axis((10, 1000, -100, 10))
ax.grid(which="both", axis="both")
plt.show()
# ---------------------------------------------------------------
# butter, cheby1, cheby2, ellip, bessel: Filter design using order and critical points
# ---------------------------------------------------------------
# scipy.signal.butter
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
"""
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.ellip
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html
"""
b, a = signal.ellip(4, 5, 40, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Elliptic filter frequency response (rp=5, rs=40)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-40, color='green') # rs
plt.axhline(-5, color='green') # rp
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.cheby1
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
"""
b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Chebyshev Type I frequency response (rp=5)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-5, color='green') # rp
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.cheby2
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html
"""
b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Chebyshev Type II frequency response (rs=40)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-40, color='green') # rs
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.bessel
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html
"""
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(np.abs(h)), color='silver', ls='dashed')
b, a = signal.bessel(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.figure(1)
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.xscale('log')
plt.title('Bessel filter frequency response (with Butterworth)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.figure(2)
plt.plot(w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w))
plt.xscale('log')
plt.title('Bessel filter group delay')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Group delay [seconds]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.show()
"""
# ---------------------------------------------------------------
# buttord, cheb1ord, cheb2ord, ellipord: Find order and critical points from passband and stopband spec
# ---------------------------------------------------------------
# scipy.signal.buttord
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html
"""
N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
b, a = signal.butter(N, Wn, 'band', True)
w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth bandpass filter fit to constraints')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
plt.axis([10, 100, -60, 3])
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.cheb1ord
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ord.html
"""
N, Wn = signal.cheb1ord(0.2, 0.3, 3, 40)
b, a = signal.cheby1(N, 3, Wn, 'low')
w, h = signal.freqz(b, a)
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Chebyshev I lowpass filter fit to constraints')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0) # stop
plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0) # pass
plt.axis([0.08, 1, -60, 3])
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.cheb2ord
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb2ord.html
"""
N, Wn = signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
b, a = signal.cheby2(N, 60, Wn, 'stop')
w, h = signal.freqz(b, a)
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Chebyshev II bandstop filter fit to constraints')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.fill([.01, .1, .1, .01], [-3,  -3, -99, -99], '0.9', lw=0) # stop
plt.fill([.2,  .2, .5,  .5], [ 9, -60, -60,   9], '0.9', lw=0) # pass
plt.fill([.6,  .6,  2,   2], [-99, -3,  -3, -99], '0.9', lw=0) # stop
plt.axis([0.06, 1, -80, 3])
plt.show()
"""
# ---------------------------------------------------------------
# scipy.signal.ellipord
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html
"""
N, Wn = signal.ellipord(30, 10, 3, 60, True)
b, a = signal.ellip(N, 3, 60, Wn, 'high', True)
w, h = signal.freqs(b, a, np.logspace(0, 3, 500))
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Elliptical highpass filter fit to constraints')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.fill([.1, 10,  10,  .1], [1e4, 1e4, -60, -60], '0.9', lw=0) # stop
plt.fill([30, 30, 1e9, 1e9], [-99,  -3,  -3, -99], '0.9', lw=0) # pass
plt.axis([1, 300, -80, 3])
plt.show()
"""
# ---------------------------------------------------------------
# iirdesign: General filter design using passband and stopband spec
# ---------------------------------------------------------------
# scipy.signal.iirdesign
# Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html

# scipy.signal.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba')
# The type of IIR filter to design:
# Butterworth : butter
# Chebyshev I : cheby1
# Chebyshev II : cheby2
# Cauer/elliptic: ellip
# Bessel / Thomson: bessel
#
# example:
# Lowpass: wp = 0.2, ws = 0.3
# Highpass: wp = 0.3, ws = 0.2
# Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]
# Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]
# ---------------------------------------------------------------
"""
# wp = 0.2
# ws = 0.3
wp = [0.1, 0.6]
ws = [0.2, 0.5]
gpass = 1
gstop = 40

system = signal.iirdesign(wp, ws, gpass, gstop, analog=False, ftype = 'cheby1', output='ba')
w, h = signal.freqz(*system)

fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid()
# ax1.set_ylim([-120, 120])
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid()
ax2.axis('tight')
# ax2.set_ylim([-6, 1])
nticks = 8
ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
plt.show()
"""
# ---------------------------------------------------------------
