from __future__ import annotations
from msilib import sequence
import matplotlib.pyplot as plt 
from matplotlib.collections import PathCollection
from sklearn.neighbors import KernelDensity
from typing import Any, Literal, Optional, Sequence, Type, TypeVar, Union
import numpy as np
from mpl_toolkits.mplot3d.art3d import PolyCollection
import itertools

LINE_WIDTH = 5
rc_params = {
    'figure':{
        'figsize':(19.20,10.80)
    },
    'font':{
            'family':'Arial',
            'size':45,
            'weight':'bold',
        },
    'axes':{
            'spines.right':False,
            'spines.top':False,
            'linewidth':LINE_WIDTH,
            'labelweight':'bold'
        },
    'xtick':{
        'major.width':LINE_WIDTH,
        'major.size':15
    },
    'ytick':{
        'major.width':LINE_WIDTH,
        'major.size':15
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

KwsType = dict[Literal['bar_kw', 'err_kw', 'scatter_kw'], dict[str, Any]]
def error_plot(dataset:Sequence[Sequence[int|float]], ax, colors:str|Sequence[str]='k', scatter_shift_nums:float|Sequence[float]=0.1, 
                error_type:Literal['SD', 'SE']='SD', custom_kws:Optional[KwsType]=None) -> None: #datasetの次元に注意．行の数だけ棒グラフが描画される．
    flat = itertools.chain.from_iterable
    row_num = len(dataset); col_nums = [len(arr) for arr in dataset] #行ごとの列の数
    if isinstance(scatter_shift_nums, float): 
        scatter_shift_nums = [scatter_shift_nums]*row_num #キャスト
    bar_xs = range(row_num)
    scatter_xs =  tuple(flat([[x+shift_num]*x_len for x, x_len, shift_num in zip(bar_xs, col_nums, scatter_shift_nums)])) #散布図はscatter_shift_numだけずらすとエラーバーが見やすい
    scatter_colors = tuple(flat([[color]*x_len for color, x_len in zip(colors, col_nums)])) if not isinstance(colors, str)  else colors
    aves = [np.mean(arr) for arr in dataset] #行ごとにの列の数が異なる可能性があるため，それぞれの行に分解してmean, stdを使う必要がある．np.mean(dataset, axis=1)だとエラー
    errors = [np.std(arr) for arr in dataset]
    if error_type == 'SE':
        errors = [err / np.sqrt(x_len) for err, x_len in zip(errors, col_nums)]
    
    #デフォルトの各プロットオプション
    bar_kw = {
            'edgecolor':colors,
            'fill':False,
            'color':colors,
            'linewidth':LINE_WIDTH,
        }
    err_kw = {
            'yerr':errors,
            'capsize':10,
            'capthick':5,
            'elinewidth':LINE_WIDTH,
            'fmt':'none',
            'ecolor':'k',
        }
    scatter_kw = {
        'color':scatter_colors,
        's':150
        }
    
    #custom_kwsが設定されたときはキーワードの追加または上書き
    if custom_kws is not None:
       bar_kw = {**bar_kw, **custom_kws['bar_kw']} if 'bar_kw' in custom_kws else bar_kw
       err_kw = {**err_kw, **custom_kws['err_kw']} if 'err_kw' in custom_kws else err_kw
       scatter_kw = {**scatter_kw, **custom_kws['scatter_kw']} if 'scatter_kw' in custom_kws else scatter_kw
    
    ax.bar(bar_xs, aves, **bar_kw)
    ax.errorbar(bar_xs, aves, **err_kw)
    ax.scatter(scatter_xs, tuple(flat(dataset)), **scatter_kw)
    ax.set(xticks=bar_xs)

def plot_3d_spectrogram(ax_3d, array:np.ndarray, N:int, fs:float, window_size:int, step:int) -> None:
    freq_mesh = [fs*k / window_size for k in range(window_size//2 + 1)]
    time_mesh = np.linspace(0,N*(1/fs),(N-window_size)//step)
    X , Y = np.meshgrid(time_mesh,freq_mesh)
    Z = array.T
    ax_3d.plot_surface(X,Y,Z,cmap='terrain')
    
if __name__ == '__main__':
    for group, values in rc_params.items(): plt.rc(group, **values)
    sample_dataset = np.random.rand(2, 15)
    fig, axes = plt.subplots(1, 2)
    error_plot(sample_dataset, axes[0], colors='k', error_type='SE', scatter_shift_nums=0.5, custom_kws={
        'bar_kw':{
            'hatch':['//', '**']
        },
        'scatter_kw':{
            'edgecolor':'k',
            'linewidth':2
        },
        'err_kw':{
            'capsize':50
        }
    })
    error_plot(sample_dataset, axes[1], error_type='SE')
    for ax, title in zip(axes, ('custom', 'default')): ax.set(title=title, xlabel='data±SE', xticklabels=('data1', 'data2'))
    fig.tight_layout()
    plt.show()
    
def join_hist(scats:list[PathCollection], bins:int=100, kernel=False):
    fig = scats[0].get_figure()
    fig.clf()
    gs = fig.add_gridspec(2, 2,  width_ratios=(8, 1), height_ratios=(1, 8),)
    hist_x = fig.add_subplot(gs[0, 0:-1])
    hist_y = fig.add_subplot(gs[1:, -1])
    kde_x_ax = hist_x.twinx(); kde_y_ax = hist_y.twiny()
    scatter = fig.add_subplot(gs[1:, 0:-1], )
    for scat in scats:
        data = np.array(scat.get_offsets())
        x = data[:, 0]; y = data[:, 1]
        scatter.scatter(x, y)
        if kernel:
            kde_x = KernelDensity(kernel='gaussian', bandwidth=1).fit(x[:, None])
            kde_y = KernelDensity(kernel='gaussian', bandwidth=1).fit(y[:, None])
            dens_x = np.linspace(*scatter.get_xlim(), 500)[:, None]
            dens_y = np.linspace(*scatter.get_ylim(), 500)[:, None]
            kde_x_ax.plot(dens_x, np.exp(kde_x.score_samples(dens_x)))
            kde_y_ax.plot(np.exp(kde_y.score_samples(dens_y)), dens_y)
        else:
            hist_x.hist(x, bins=bins, alpha=0.7); hist_y.hist(y, bins=bins, orientation='horizontal', alpha=0.7)
    for ax in (hist_x, hist_y, kde_x_ax, kde_y_ax):
        for direction in ('right', 'top', 'left', 'bottom'):
            ax.spines[direction].set_visible(False)
            ax.tick_params(**{direction:False, f'label{direction}':False})
    for ax in (hist_x, kde_x_ax):
        ax.set(xlim=scatter.get_xlim())
        ax.spines['bottom'].set_visible(True)
    for ax in (hist_y, kde_y_ax):
        ax.set(ylim=scatter.get_ylim())
        ax.spines['left'].set_visible(True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)