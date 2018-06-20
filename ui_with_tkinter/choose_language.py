# -*- coding: utf-8 -*-

from tkinter import *
from ui_with_tkinter import login_ui


class ChooseLanguage:

    def __init__(self):
        self.root = Tk()
        self.root.title('语言选择')
        self.root.minsize(width=400, height=260)
        self.root.maxsize(width=400, height=260)
        self.root.attributes("-alpha", 1)

        self.info = Label(self.root, text='Please Choose Language', font=('Times New Roman', 24), fg='blue')
        self.info.place(x=40, y=50)

        self.chinese_button = Button(self.root, text='Chinese', font=('Times New Roman', 18),
                                     command=self.choose_chinese)
        self.english_button = Button(self.root, text='English', font=('Times New Roman', 18),
                                     command=self.choose_english)
        self.chinese_button.place(x=80, y=150)
        self.english_button.place(x=220, y=150)

    def choose_chinese(self):
        self.root.destroy()
        login_ui.main(language='chinese')

    def choose_english(self):
        self.root.destroy()
        login_ui.main(language='english')


def main():
    ChooseLanguage()
    mainloop()


if __name__ == '__main__':
    main()
