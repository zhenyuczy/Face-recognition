# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import messagebox as mb
from ui_with_tkinter import face_recognition_ui
from PIL import Image
from PIL import ImageTk


class Login:

    def __init__(self, language='chinese'):
        self.root = Tk()
        self.language = language
        if self.language == 'chinese':
            self.root.title('欢迎来到人脸识别系统！')
        else:
            self.root.title('Welcome to Face Recognition System!')
        # self.root.geometry('450x300')
        """
        Fix window size.
        """
        self.root.minsize(width=1200, height=800)
        self.root.maxsize(width=1200, height=800)
        self.root.attributes("-alpha", 0.9)
        """
        Rewrite exit method.
        """
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        """
        Set welcome image.
        """
        self.canvas = Canvas(self.root, width=1200, height=800)
        self.image = Image.open('../resources/LoginBG.jpg').resize((1200, 800))
        self.canvas.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')
        if self.language == 'chinese':
            self.canvas.create_text(330, 180, text="人脸识别系统", font=('楷体', 64), fill='white', anchor='nw')
            self.canvas.create_text(405, 380, text="制作：陈振宇", font=('楷体', 48), fill='white', anchor='nw')
        else:
            self.canvas.create_text(155, 180, text="Face Recognition System", font=('Times New Roman', 64),
                                    fill='white', anchor='nw')
            self.canvas.create_text(305, 380, text="Producer: ChenZhenyu", font=('Times New Roman', 48),
                                    fill='white', anchor='nw')
        self.canvas.place(x=0, y=0)
        if self.language == 'chinese':
            Button(self.canvas, text='退出', font=('楷体', 18), command=self.on_exit).place(x=500, y=600)
            Button(self.canvas, text='进入', font=('楷体', 18), command=self.on_enter).place(x=640, y=600)
        else:
            Button(self.canvas, text='Exit', font=('Times New Roman', 18), command=self.on_exit).place(x=500, y=600)
            Button(self.canvas, text='Enter', font=('Times New Roman', 18), command=self.on_enter).place(x=640, y=600)

        self.root.mainloop()

    def on_exit(self):
        if self.language == 'chinese':
            if mb.askyesno("退出", '你确定退出应用程序吗？'):
                self.root.destroy()
        else:
            if mb.askyesno("Exit", 'Are you sure to exit?'):
                self.root.destroy()

    def on_enter(self):
        self.root.destroy()
        face_recognition_ui.main(language=self.language)


def main(language='chinese'):
    Login(language=language)


if __name__ == '__main__':
    main()
