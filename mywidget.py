import tkinter as tk
from typing import Sequence
from PIL import ImageTk , Image
import numpy as np
import cv2

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
            self.photos = [ImageTk.PhotoImage(image=Image.open(i)) for i in img_path_sequence]
        if img_sequence is not None:
            self.photos = [ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))) for i in img_path_sequence]
        if self.photos:
            self._draw_canvas()
    def _draw_canvas(self):
        self['width'] , self['height'] = self.photos[self._index]._PhotoImage__size
        if not self._id:
            self.delete(self._id)
        self._id = self.create_image(0,0,anchor='nw',image=self.photos[self._index])
    def append_img_path(self,img_path:str):
        self.photos.append(ImageTk.PhotoImage(image=Image.open(img_path)))
        self.index = len(self.photos) - 1
    def append_img(self,img:np.ndarray):
        self.photos.append(ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
        self.index = len(self.photos) - 1
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self,value:int):
        self._index = value
        self._draw_canvas()
    
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