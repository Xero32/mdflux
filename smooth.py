import math
import numpy
from scipy.special import erf

def Gauss(t, mu, h):
    g = 1. / (numpy.sqrt(2. * numpy.pi * h*h) ) * numpy.exp( (-1.0) * (t-mu)*(t-mu) / (2.*h*h) )
    return g

def GaussSmoothing(N, t, f, dt=1, nu=4, edge='none', X=[]):
    h = nu * dt
    fG = 0
    GaussSum = 0
    for j in range (0,N):
        if math.fabs(t-j) > 5.0*h:
            continue
        if edge == 'positive' or edge == 'pos':
            if X[j] <= 5.*h:                                        #TODO maybe t <= 0
                if X[t] < 0:
                    fG = 0
                    return fG
                else:
                    GaussSum += Gauss(t,j,h) / (0.5 + 0.5 * erf(j / (h*numpy.sqrt(2.) ) ) )
                #end if (omit negatives)
            else:
                GaussSum += Gauss(t, j, h)   # j = time
        elif edge == 'negative' or edge == 'neg':
            if X[j] >= -5.*h:
                if X[t] > 0:                                       #TODO maybe t >= 0
                    fG = 0
                    return fG
                else:
                    GaussSum += Gauss(t,j,h) * 1. / (1. - (0.5 + 0.5 * erf(-j / (h*numpy.sqrt(2.) ) ) ) )
                #end if (omit positives)
            else:
                GaussSum += Gauss(t,j,h)
            #end if (edge gaussians)
        elif edge == 'none' or edge == '':
            GaussSum += Gauss(t,j,h)
        else:
            print('Error in Gaussian Smoothing. Choose one of the following boundary handling methods: "none", "positive" or "negative".')
            return -1
        #end if

    for j in range (0,N):
        if math.fabs(t-j) > 5.0*h:
            continue
        if edge == 'positive' or edge == 'pos':
            if X[j] <= 5.0*h:
                if X[t] < 0:                                       #TODO maybe t <= 0
                    fG = 0
                    return fG
                    continue
                else:
                    fG += Gauss(t,j,h) / GaussSum * f[j] / (0.5 + 0.5 * erf(j / (h*numpy.sqrt(2.) ) ) )
                #end if (omit negatives)
            else:
                fG += Gauss(t, j, h) / GaussSum * f[j]
            #end if (edge gaussians)
        elif edge =='negative' or edge == 'neg':
            if X[j] >= -5.0*h:
                if X[t] > 0:                                       #TODO maybe t >= 0
                    fG = 0
                    return fG
                    continue
                else:
                    fG += Gauss(t,j,h) / GaussSum * f[j] * 1. / (1. - (0.5 + 0.5 * erf(-j / (h*numpy.sqrt(2.) ) ) ) )
                #end if (omit positives)
            else:
                fG += Gauss(t, j, h) / GaussSum * f[j]
            #end if (edge gaussians)
        elif edge =='none' or edge == '':
            fG += Gauss(t, j, h) / GaussSum * f[j]
        else:
            print('Error in Gaussian Smoothing. Choose one of the following boundary handling methods: "none", "positive" or "negative".')
            return -1
        #end if
    return fG # double

def DoSmoothing(length, Arr, dt=1, nu=0):
    for k in range(0,length):
        Arr[k] = GaussSmoothing(length, k, Arr, dt=1, nu=nu)
    return Arr


def smoothing(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def Compress(Val, wl=0, num=0):
    # it is mandatory to set ONE value of wl (window length) or num in function call
    l = len(Val)
    if wl == 0:
        assert(num != 0)
        wl = int(l // num)
    if num == 0:
        assert(wl != 0)
        num = int(l // wl)

    assert(num*wl == l)

    index = 0
    counter = 0
    avg = 0
    while (index+wl <= l):
        for i in range(index, index+wl):
            avg += Val[i]

        Val[counter] = avg / wl
        index += wl
        counter += 1
        avg = 0

    assert(counter == num)
    return Val[:num], wl, num
