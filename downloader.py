from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .mywidget import ScrollImageViewer
import tkinter as tk
import requests 
from bs4 import BeautifulSoup
import os 
import numpy as np 
from PIL import Image
from io import BytesIO
import cv2 
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from typing import Optional

@dataclass
class NamedImage:
    name:str
    img:np.ndarray

@dataclass
class NamedImageFolder:
    name:str
    imgs:list[NamedImage]
    def pop_img(self, index:int):
        named_img = self.imgs.pop(index)
        self._shift_index(index)
        return named_img
    def _shift_index(self, index:int):
        target_imgs = self.imgs[index:]
        for img in target_imgs:
            old_name, ext = os.path.splitext(img.name)
            new_name = str(int(old_name) - 1) + ext
            img.name = new_name
            
TIMEOUT_TIME = 10
class ImageDownloader:
    def __init__(self, webdriver_path:str):
        self.folders:list[NamedImageFolder] = []
        self.driver = webdriver.Chrome(webdriver_path)
        self.driver.implicitly_wait(TIMEOUT_TIME)
    def set_imgs_with_request(self, url:str, imgs_name:str, inner:bool=False) -> None: #inner=True → サムネイルクリック先の画像urlを取り出す
        res = requests.get(url)
        bs = BeautifulSoup(res.text, 'html.parser')
        if inner:
            img_urls = [img_tag.parent['href'] for img_tag in bs.select('img') if img_tag.parent.name == 'a'] 
        else:
            img_urls = [img_tag['src'] for img_tag in bs.select('img')]
            
        img_urls = self._filter_imgs(img_urls)
        self._set_imgs(img_urls, imgs_name)
    def set_imgs_with_selenium(self, url:str, imgs_name:str, inner:bool=False, #loading_css_selectorを設定すると指定したセレクタのdomがなくなるまで待機する（読み込み対策）
                                loading_css_selector:Optional[str]=None,): 

        self.driver.get(url)
        select = self.driver.find_elements_by_css_selector

        if loading_css_selector:
            while True:
                loading_contents = select(loading_css_selector)
                if not loading_contents:
                    break
                cor = loading_contents[0].location
                self.driver.execute_script(f"window.scrollTo(0,{cor['y']});")
        if inner:
            img_urls = [attr(get_parent(img_tag), 'href') for img_tag in select('img') if get_parent(img_tag).tag_name == 'a']
        else:   
            img_urls = [attr(img_tag, 'src') for img_tag in select('img')]
        
        img_urls = self._filter_imgs(img_urls)
        self._set_imgs(img_urls, imgs_name)
    def view(self) -> None:
        def btn_cmd(e, folder_:NamedImageFolder) -> None:
            sub_win = tk.Toplevel(win)
            title_f = tk.LabelFrame(sub_win, text=folder_.name)
            viewer = ScrollImageViewer(title_f, img_sequence=[img.img for img in folder_.imgs])
            del_btn = tk.Button(title_f, text='消去',command=lambda:[folder_.pop_img(viewer.index), viewer.del_img(viewer.index)])
            del_btn.pack()
            viewer.pack()
            title_f.pack()

        win = tk.Tk()
        for folder in self.folders:
            btn = tk.Button(win, text=folder.name)
            btn.bind('<1>', lambda e,arg=folder:btn_cmd(e, arg))
            btn.pack()
        win.mainloop()
    def save(self, dir:str) -> None:
        for folders in self.folders:
            out_dir = os.path.join(os.path.join(dir, folders.name))
            os.makedirs(out_dir, exist_ok=True)
            for img in folders.imgs:    
                imwrite(os.path.join(out_dir, img.name), img.img)
    def _set_imgs(self, img_urls:Sequence[str], imgs_name:str):
        imgs = []
        none_num = 0
        for i, src in enumerate(img_urls, 1):
            ext = src[-4:]
            img = url_to_image(src)
            if img is None:
                print('download error')
                none_num += 1
                continue
            imgs.append(NamedImage(str(i-none_num).zfill(5)+ext, img))
        self.folders.append(NamedImageFolder(imgs_name, imgs))
    def _filter_imgs(self, img_urls:Sequence[str]) -> list[str]: 
        """
        最後3文字が指定した文字列以外は除外する．(ここで，目的のurlが画像の参照ではなく，他のwebページだったり，.pngXXXXXのような文字列を排除するのが目的)
        """
        filter_exts = {'png', 'jpg'}
        return [url for url in img_urls if (url is not None) and (url[-3:] in filter_exts)]
    
def imwrite(filename, img, params=None): #cv2.imwriteが日本語に対応してないので代わり
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
    
def url_to_image(url:str) -> np.ndarray:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        try:
            img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
            img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
        except:
            return
        return img

def get_parent(el:WebElement) -> WebElement:
    return el.find_element_by_xpath('..')

def attr(el:WebElement, attr_name:str) -> str:
    return el.get_attribute(attr_name)

if __name__ == '__main__':
    loader = ImageDownloader('C:/chromedriver_win32/chromedriver.exe')
    loader.set_imgs_with_request('https://www.yahoo.co.jp/', 'Yahoo_images')
    loader.view()
    loader.save(os.path.join(__file__, '..'))

