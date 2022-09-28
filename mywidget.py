from __future__ import annotations
import tkinter as tk
from typing import Sequence, TypeVar, Container, Callable
from abc import ABCMeta, abstractmethod
from PIL import ImageTk , Image
import numpy as np
import cv2

Value = TypeVar('Value')
class ContainerManager(tk.LabelFrame, metaclass=ABCMeta): #抽象クラス
    def __init__(self, master, contents:Container[Value], receive_content_func:Callable[[tk.Event, Value], None], side:str='top', **kw): 
        """
        receive_content_funcに与えられる引数はイベントとcontents内の要素.
        継承先の_create_content_widget内で作成したウィジェットのバインド時に設定する.
        """
        super().__init__(master, **kw)
        self.contents:Container[Value] = contents
        self.receive_content_func:Callable[[Value], None] = receive_content_func
        self.side:str = side
        self.update()       
    def update(self): #クラスの外側やreceive_content_func内でcontentsを更新したときはこのメソッドを呼ぶとウィジェットが更新される
        for widget in self.winfo_children():
            widget.destroy()
        for content in self.contents:
            frame = tk.Frame(self)
            self._create_content_widget(frame, content)
            frame.pack(side=self.side)
    @abstractmethod
    def _create_content_widget(self, frame:tk.Frame, content:Value) -> None: ... #リストの中の表示するコンテンツの説明や操作ボタンをこの抽象メソッドで決める．

class ScrollImageViewer(tk.Canvas):
    def __init__(self,master,img_path_sequence:Sequence[str]=None,img_sequence:Sequence[np.ndarray]=None,index:int=0,cnf={},**kw):
        super().__init__(master,cnf=cnf,**kw)
        def canvas_cmd(e):
            if not self.photos:
                return
            if e.delta > 0:
                if self._index < len(self.photos)-1:
                    self._index += 1
                    self.delete(self._id)
                    self._draw_canvas()
            else:
                if self._index > 0:
                    self._index -= 1
                    self.delete(self._id)
                    self._draw_canvas()
        self.photos = []
        self._index = index
        self._id = 0
        self.bind('<MouseWheel>',canvas_cmd,'+')
        if img_path_sequence is not None:
            self.photos = [ImageTk.PhotoImage(image=Image.open(i), master=self) for i in img_path_sequence]
        if img_sequence is not None:
            self.photos = [ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB)), master=self) for i in img_sequence]
        if self.photos:
            self._draw_canvas()
    def _draw_canvas(self):
        self['width'] , self['height'] = self.photos[self._index]._PhotoImage__size
        if not self._id:
            self.delete(self._id)
        self._id = self.create_image(0,0,anchor='nw',image=self.photos[self._index])
    def append_img_path(self,img_path:str):
        self.photos.append(ImageTk.PhotoImage(image=Image.open(img_path)), master=self)
        self.index = len(self.photos) - 1
    def append_img(self,img:np.ndarray):
        self.photos.append(ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
        self.index = len(self.photos) - 1
    def del_img(self, index:int):
        del self.photos[index]
        self.index = (index - 1) if index > 0 else index + 1
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self,value:int):
        self._index = value
        self._draw_canvas()
        
class ImageCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.img_id:int|None = None
        self.photo:ImageTk.PhotoImage|None = None 
        self.img:np.ndarray|None = None
    def update_img(self, img:np.ndarray):
        if self.photo is not None:
            self.delete(self.img_id)
        self.img = img
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), master=self)
        self.img_id = self.create_image(0, 0, anchor='nw', image=self.photo)

class ImageExtractor(tk.Tk):
    def __init__(self, image:np.ndarray, **kw):
        super().__init__(**kw)
        self.origin_img:np.ndarray = image
        height, width, _ = image.shape
        self.viewer:ImageCanvas = ImageCanvas(self, width=width, height=height)
        self.viewer.update_img(image)
        self.viewer.bind('<1>', lambda e:self._send_two_cor(e, self._extract_img))
        self.viewer.pack()
        self.result:np.ndarray|None = None
    def _send_two_cor(self, e:tk.Event, send_func:Callable[[tuple[int, int]], None],
                      *, __clicked_cor:list[None|int]=[None]):
        if __clicked_cor[0] is None:
            __clicked_cor[0] = (e.x, e.y)
        else:
            second_clicked_cor = (e.x, e.y)
            if __clicked_cor[0] == second_clicked_cor: return
            send_func(__clicked_cor[0], second_clicked_cor)
            __clicked_cor[0] = None
    def _extract_img(self, cor_1:tuple[int, int], cor_2:tuple[int, int]):
        xmin, ymin = cor_1; xmax, ymax = cor_2
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        img_copy = self.origin_img.copy()
        visualized_img = cv2.rectangle(img_copy, cor_1, cor_2, (0, 0, 0), 3)
        self.viewer.update_img(visualized_img)
        extracted_img = self.origin_img[ymin:ymax, xmin:xmax]
        if self.result is None:
            tk.Button(self, text='完了', command=self.destroy).pack()
        self.result = extracted_img
        
class ScrollFrame(tk.Frame):
    def __init__(self,master,x=True,y=True,custom_width=0,custom_height=0,**kw):
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)),"units")
        super().__init__(master,**kw)
        self._f = tk.Frame(self)
        if custom_width:
            canvas = tk.Canvas(self._f,width=custom_width,height=custom_height)
        else:
            canvas = tk.Canvas(self._f)
        self._scrollable_frame = tk.Frame(canvas)
        self._scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox('all'),
            )
        )
        canvas.create_window(0,0,anchor='nw',window=self._scrollable_frame,)
        if y:
            self.scrollbar_y = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
            self.scrollbar_y.pack(side=tk.RIGHT, fill="y")
            canvas.configure(yscrollcommand=self.scrollbar_y.set)
        if x:
            self.scrollbar_x = tk.Scrollbar(self, orient="horizontal", command=canvas.xview)
            self.scrollbar_x.pack(side=tk.BOTTOM, fill="x")
            canvas.configure(xscrollcommand=self.scrollbar_x.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH,expand=True)
        canvas.bind("<MouseWheel>",_on_mousewheel,'+')
    @property
    def f(self): #子ウィジェットを作るときは、子ウィジェットのインスタンス生成時、引数masterにこのプロパティを渡すこと．
        return self._scrollable_frame
    def pack(self,**kw):
        self._f.pack()
        super().pack(**kw)

def color_change_hover(event):
    if event.type == '7':
        event.widget['bg'] = 'gray'
    if event.type == '8':
        event.widget['bg'] = 'SystemButtonFace'