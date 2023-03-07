import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import scipy as sci
from scipy.fft import fft, ifft, fftfreq
import sys

class Frequencyframe:
  def __init__(self, xf : np.array, yf : np.array):
    self.xf = xf
    self.yf = yf

class PhaseVocoder:
  def __init__(self, N, r, Fs):
    self.N = N
    self.Fs = Fs
    self.HOPa = int(N/4)
    self.DTa = self.HOPa / Fs
    self.HOPs = int(r * self.HOPa)
    self.DTs = self.HOPs / Fs
   

  def stratch_several_chanel(self, data: np.array):
    if (len(data.shape) == 1):
      return self.stratch_one_chanel(data)
    answer = self.stratch_one_chanel(data[:,0]).reshape(-1,1)
    for i in range(1, data.shape[1]):
      answer = np.concatenate([answer, self.stratch_one_chanel(data[:,i]).reshape(-1,1)], axis = 1)
    return answer


  def get_one_chanel(self, data: np.array):
    return self.stratch_one_chanel(data[:,0])

  def stratch_one_chanel(self, data: np.array) -> np.array:
    # разбивание на фреймы
    frames = []
    i = 0
    while(i + self.N < len(data)):
      frame = data[i:i+self.N].copy()
      frames.append(frame)
      i += self.HOPa

    # применение к фреймам rfft с окном Ханна
    frequency_frames = []

    for frame in frames:
      xf = fftfreq(self.N, 1 / self.Fs)
      w = hann(len(frames[0]))
      yf = fft(frame * w)
      frequency_frames.append(Frequencyframe(xf,yf))
    
    # меняем фазу
    frequency_frames[0].fi_s_ = np.angle(frequency_frames[0].yf) 

    for i in range(1,len(frequency_frames)):
      fi_a_i = np.angle(frequency_frames[i].yf)
      fi_a_i_prev  = np.angle(frequency_frames[i-1].yf)
      w = frequency_frames[i].xf
      D_w = ((fi_a_i - fi_a_i_prev) / self.DTa) - w
     # D_w_wrapped = ((D_w + np.pi) % (2*np.pi)) - np.pi
      w_true = w + D_w
      fi_s_i_prev = frequency_frames[i-1].fi_s_
      frequency_frames[i].fi_s_ = fi_s_i_prev + self.DTs * w_true
      
    for i in range(1,len(frequency_frames)): 
      num = np.abs(frequency_frames[i].yf)
      angle = frequency_frames[i].fi_s_
      frequency_frames[i].yf = num * np.cos(angle) + 1j * (num * np.sin(angle))

    # переводим обратно из Фурье в временной ряд.

    frames = []

    for frame in frequency_frames:
      w = hann(self.N)
      yf = np.real(ifft(frame.yf)) * w
      frames.append(yf)

    answer = []
    for i in range(len(frames)):
      for j in range(len(frames[i])):
        index = j + self.HOPs * i
        if (index >= len(answer)):
          answer.append(frames[i][j])
        else:
          answer[index] += frames[i][j]

    answer = np.array(answer)
    m = np.max(np.abs(answer))
    answer32 = (answer/m).astype(np.float32)
    return answer32
    

input_name = sys.argv[1]
output_name = sys.argv[2]
rate = float(sys.argv[3])


Fs,data=read(input_name)
vocoder = PhaseVocoder(2000, rate, Fs)
answer2 = vocoder.stratch_several_chanel(data)
write(output_name,Fs,answer2)

