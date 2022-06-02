import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import os 

def STFT(array:np.ndarray,window_size:int,step:int,window_func=np.hamming) -> np.ndarray:
    result = []
    step = step
    for i in range((len(array) - window_size) // step):
        tmp = array[i*step:i*step + window_size]
        tmp = tmp * window_func(window_size)    
        amp = ((abs(dft(tmp)))**2)[:window_size // 2 + 1]
        result.append(amp)
    return np.array(result)

def plot_3d_spectrogram(ax_3d,array,N,fs,window_size,step) -> None:
    freq_mesh = [fs*k / window_size for k in range(window_size//2 + 1)]
    time_mesh = np.linspace(0,N*(1/fs),(N-window_size)//step)
    X , Y = np.meshgrid(time_mesh,freq_mesh)
    Z = array.T
    ax_3d.plot_surface(X,Y,Z,cmap='terrain')

def DFT(array):
    N = len(array)
    A = np.arange(N)
    M = np.exp(-1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * np.pi / N)
    return np.sum(array*M,axis=1)

def IDFT(array):
    N = len(array)
    A = np.arange(N)
    M = np.exp(1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * np.pi / N)
    return np.sum(array*M,axis=1) / N

def my_get_peakindex(array,num:int) -> np.ndarray:
    result = []
    cnt = 0
    while True:
        idxes= np.argsort(array)[::-1]
        if array[idxes[cnt]] - array[idxes[cnt]-1] > 0:
            result.append([idxes[cnt]])
            if len(result) == num:
                break
        cnt = cnt + 1
    return np.array(result)

def convolve(f,g,animation_axes=None):
    f_N = len(f)
    g_N = len(g)
    result = []
    img_list = []
    is_move = False
    move = 0
    if animation_axes is not None:
        animation_axes[0].plot(f,'black')
    for tau in range(f_N + g_N):
        if tau < g_N - 1 :
            tmp_f = f[:tau+1]
            tmp_g = g[(g_N-1)-tau:]
        elif tau <= f_N - 1 : 
            is_move = True 
            tmp_f = f[move:tau+1]
            tmp_g = g
        elif tau > f_N - 1:
            tmp_f = f[move:]
            tmp_g = g[:len(tmp_f)]
        result.append(np.sum(tmp_f * tmp_g))
        if animation_axes is not None:
            im = animation_axes[0].plot(list([*[np.nan]*move]) + list(tmp_f) , 'blue')
            im_2 = animation_axes[0].plot(list([*[np.nan]*move]) + list(tmp_g),'red')
            im_3 = animation_axes[1].plot(result,'black')
            img_list.append(im+im_2+im_3)
        if is_move :
            move = move + 1 
    return (np.array(result) , img_list) if animation_axes is not None else np.array(result)

def save_STAC_animation(array:np.ndarray,window:int,output_dir:str) -> None:
    os.makedirs(output_dir,exist_ok=True)
    split_array = array.reshape(-1,window)
    for i,arr in enumerate(split_array):
        fig , axes = plt.subplots(3,1)
        axes[0].set_title(f'Data window{i+1}')
        for j,arr_ in enumerate(split_array):
            axes[0].plot( [*[np.nan]*j*window] + list(arr_) , 'red' if np.all(arr==arr_) else 'black')
        amp , ims = convolve(array,arr,axes[1:])
        axes[0].set_xlim(axes[1].get_xlim())
        axes[1].set_title('Animation')
        axes[2].set_title('Convolution result')
        plt.tight_layout()
        anim = ArtistAnimation(fig,ims,interval=10)
        anim.save(f'window_{i}_AC_result.gif')
    
def STAC(array,window_size,step):
    result = []
    step = step
    fig , axes = plt.subplots(2,1)
    for i in range((len(array) - window_size) // step):
        tmp_array = array[i*step:i*step + window_size]
        amp , _  = convolve(array,tmp_array,axes)
        result.append(amp)
    return np.array(result)

if __name__ == '__main__':
    N = 256
    times = np.linspace(0,10,N)
    y = np.sin(2*np.pi*times)
    STAC(y,N//4,N//4)


