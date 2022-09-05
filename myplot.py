from __future__ import annotations
from msilib import sequence
import matplotlib.pyplot as plt 
from typing import Any, Literal, Optional, Sequence, Type, TypeVar, Union
import numpy as np
from mpl_toolkits.mplot3d.art3d import PolyCollection

plt_params = {
    'font':{
            'family':'Arial',
            'size':24,
            'weight':'bold',
        },
    'axes':{
            'spines.right':False,
            'spines.top':False,
            'linewidth':2
        },
    'xtick':{
        'major.width':3,
        'major.size':5
    },
    'ytick':{
        'major.width':3,
        'major.size':5
    }
        
}

T = TypeVar('T', int, float)
def waterfall_plot(ax_3d, dim_2_array:np.ndarray, extent:tuple[T, T, T, T],
                   edgecolors:str|Sequence[str]='k', fill:Optional[bool]=False,
                   facecolors:str|Sequence[str]='w', alpha:float|int=1, zmin:float|int=0):
    y_len , x_len ,  = dim_2_array.shape
    x_min , x_max , y_min , y_max = extent
    xs = np.linspace(x_min,x_max,x_len)
    ys = np.linspace(y_min,y_max,y_len)
    verts:list[Any] = []
    ims:list[Any] = []
    for ydir in range(y_len):
        zs = dim_2_array[ydir]
        if fill:
            verts.append(list(zip(np.hstack((xs[0],xs,xs[-1])),np.hstack((zmin,zs,zmin)))))
        else:
            im = ax_3d.plot(xs,np.full(xs.shape,ys[ydir]),zs,c=edgecolors if type(edgecolors) is str or edgecolors is None else edgecolors[ydir])
            ims.append(*im)
    if fill:
        polygon = PolyCollection(verts,edgecolors=edgecolors,facecolors=facecolors,alpha=alpha)
        ims = ax_3d.add_collection3d(polygon,zs=ys,zdir='y')
        ax_3d.set(xlim=[x_min,x_max],ylim=[y_min,y_max],zlim=[dim_2_array.min(),dim_2_array.max()])
    return ims


KwsType = dict[Literal['bar_kw', 'err_kw', 'scatter_kw'], dict[str, str]]
def error_plot(dataset:Sequence[Sequence[int|float]], ax, colors:str|Sequence[str]='k',
            kws:Optional[KwsType]=None) -> None:
    if kws is not None:
        bar_kw = kws['bar_kw'] if 'bar_kw' in kws else {}
        err_kw = kws['err_kw'] if 'err_kw' in kws else {}
        scatter_kw = kws['scatter_kw'] if 'scatter_kw' in kws else {}
    else:
        bar_kw = {}; err_kw = {}; scatter_kw = {}
        
    row_num = len(dataset); col_num = len(dataset[0])
    bar_xs = range(row_num)
    scatter_xs =  np.array([[x]*col_num for x in bar_xs]).ravel()
    scatter_colors = np.array([[color]*col_num for color in colors]).ravel() if len(colors) > 1 else colors
    aves = np.mean(dataset, axis=1)
    errors = np.std(dataset, axis=1)
    ax.bar(bar_xs, aves, edgecolor=colors, fill=False, **bar_kw)
    ax.errorbar(bar_xs, aves, yerr=errors, capsize=5, fmt='none', ecolor='k', **err_kw)
    ax.scatter(scatter_xs, dataset, color=scatter_colors, **scatter_kw)
    ax.set(xticks=bar_xs)

def plot_3d_spectrogram(ax_3d, array:np.ndarray, N:int, fs:float, window_size:int, step:int) -> None:
    freq_mesh = [fs*k / window_size for k in range(window_size//2 + 1)]
    time_mesh = np.linspace(0,N*(1/fs),(N-window_size)//step)
    X , Y = np.meshgrid(time_mesh,freq_mesh)
    Z = array.T
    ax_3d.plot_surface(X,Y,Z,cmap='terrain')
    
if __name__ == '__main__':
    for group, values in plt_params.items(): plt.rc(group, **values)
    sample_dataset = np.random.rand(3, 12)
    fig, ax = plt.subplots()
    error_plot(sample_dataset, ax, colors=['b', 'g', 'purple'])
    plt.show()
    