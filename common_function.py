from cmath import sin
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
        amp = ((abs(DFT(tmp)))**2)[:window_size // 2 + 1]
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

def cross_correlate(f:np.ndarray,g:np.ndarray,times:np.ndarray,mode='full',animation_axes=None):
    if mode not in ['full','right','valid']:
        raise ValueError('error')
    if animation_axes is not None:
        animation_axes[0].plot(times,f,'black')
    f_N = len(f)
    g_N = len(g)
    lag_times = np.array(list(reversed(-1*times[1:g_N])) + list(times)) if mode == 'full' else times
    range_ = range(f_N + g_N - 2) if mode  == 'full' else range(g_N-1,f_N+g_N - 2) if mode =='right' else range(g_N-1,f_N-1)
    result = []
    img_list = []
    is_move = False
    move = 0 
    for tau in range_:
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
            tmp_t = times[:tau+1]
            y_1 = list([*[np.nan]*move]) + list(tmp_f)
            y_2 = list([*[np.nan]*move]) + list(tmp_g)
            im = animation_axes[0].plot(tmp_t,y_1, 'blue')
            im_2 = animation_axes[0].plot(tmp_t,y_2,'red')
            lag_title = animation_axes[0].text(0.5, 1.01, f'tau = {tau-(g_N-1)}',ha='center', va='bottom',transform=animation_axes[0].transAxes, fontsize='large')
            im_3 = animation_axes[1].plot(lag_times[:len(result)],result,'black')
            img_list.append(im+im_2+im_3 + [lag_title])
        if is_move :
            move = move + 1 
    return (np.array(result) , img_list) if animation_axes is not None else np.array(result)

def save_STAC_animation(times:np.ndarray,array:np.ndarray,window:int,output_dir:str) -> None:
    os.makedirs(output_dir,exist_ok=True)
    split_array = array.reshape(-1,window)
    for i,arr in enumerate(split_array):
        fig , axes = plt.subplots(3,1)
        axes[0].set_title(f'Data window{i+1}')
        for j,arr_ in enumerate(split_array):
            axes[0].plot(times[:(j+1)*window],[*[np.nan]*j*window] + list(arr_) , 'red' if np.all(arr==arr_) else 'black')
        amp , ims = cross_correlate(array,arr,times,'valid',axes[1:])
        axes[0].set_xlim(axes[1].get_xlim())
        axes[0].set_xlabel('Time [sec]')
        axes[1].set_xlabel('Time [sec]')
        axes[2].set_title('Convolution result')
        axes[2].set_xlabel('Lag [sec]')
        plt.tight_layout()
        anim = ArtistAnimation(fig,ims,interval=10)
        #break
        anim.save(os.path.join(output_dir,f'window{i+1}_convolition.gif'))
        plt.clf()
    #plt.show()
    
def STAC(times:np.ndarray,array:np.ndarray,window_size:int,step:int) -> np.ndarray:
    result = []
    step = step
    for i in range((len(array) - window_size) // step):
        tmp_array = array[i*step:i*step + window_size]
        amp  = cross_correlate(array,tmp_array,times,'right')
        result.append(amp)
    return np.array(result)

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    N = 512
    times = np.linspace(0,30,N)
    y = np.sin(2*np.pi*times*0.5) 
    fig , ax = plt.subplots()
    ax.plot(np.convolve(y,list(reversed(y)),mode='same'))
    plt.show()

    


