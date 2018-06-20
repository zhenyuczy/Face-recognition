# -*- coding: utf-8 -*-

from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as mb
from face_recognition import test_images
from face_recognition import training_images
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
import os
import pandas as pd


class FaceRecognition:

    def __init__(self, language='chinese'):
        self.root = Tk()
        self.language = language
        if self.language == 'chinese':
            self.root.title('人脸识别系统制作人：陈振宇')
        else:
            self.root.title('Face Recognition System implemented By ChenZhenyu.')
        self.root.maxsize(width=1500, height=800)
        self.root.minsize(width=1500, height=800)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.menuBar = Menu(self.root)

        self.help_menu = Menu(self.menuBar)

        if self.language == 'chinese':
            self.menuBar.add_command(label='训练', font=('楷体', 12), command=self.enable_train_ui)
            self.menuBar.add_command(label='预测', font=('楷体', 12), command=self.enable_test_ui)
            self.menuBar.add_command(label='检索', font=('楷体', 12), command=self.enable_retrieval_ui)
        else:
            self.menuBar.add_command(label='Train', font=('Times New Roman', 12), command=self.enable_train_ui)
            self.menuBar.add_command(label='Predict', font=('Times New Roman', 12), command=self.enable_test_ui)
            self.menuBar.add_command(label='Retrieval', font=('Times New Roman', 12), command=self.enable_retrieval_ui)

        # self.menuBar.add_cascade(label='帮助', font=('楷体', 12), menu=self.help_menu)

        # self.help_menu.add_command(label='训练', font=('楷体', 12))
        # self.help_menu.add_command(label='预测', font=('楷体', 12))

        self.enable_train = False
        self.enable_test = False
        self.enable_retrieval = False

        self.training_dir = None
        self.dir_var = StringVar()

        self.image_size = 144

        self.print_pro_info = True
        self.pro_info_var = StringVar()

        self.fit_all_data = False
        self.fit_data_var = StringVar()

        self.print_clf_info = True
        self.clf_info_var = StringVar()

        self.use_alignment = True
        self.alignment_var = StringVar()

        self.training_parameters = None
        self.training_par_x_scrollbar = None
        self.training_par_y_scrollbar = None

        self.test_parameters = None
        self.test_par_x_scrollbar = None
        self.test_par_y_scrollbar = None

        self.output_frame = None
        self.pro_info_frame = None
        self.clf_info_frame = None

        self.pro_info_box = None
        self.pro_x_scrollbar = None
        self.pro_y_scrollbar = None

        self.clf_info_box = None
        self.clf_x_scrollbar = None
        self.clf_y_scrollbar = None

        self.test_image_path = None
        self.image_path_var = StringVar()

        self.test_state = 'recognition'
        self.test_state_var = StringVar()

        self.choose_ui_widgets = []
        self.choose_labels = []
        self.choose_label_var = StringVar()
        self.delete_label_var = StringVar()
        self.chooses_box = None
        self.chooses_box_x_scrollbar = None
        self.chooses_box_y_scrollbar = None

        self.answer = None
        self.predict_image_data = None

        self.answer_box = None
        self.answer_x_scrollbar = None
        self.answer_y_scrollbar = None

        # images list
        self.display = []

        # retrieval_ui
        self.labels_array = None
        self.paths_array = None
        self.labels_box = None
        self.labels_box_x_scrollbar = None
        self.labels_box_y_scrollbar = None
        self.label_var = StringVar()
        self.label = None
        self.index = None
        self.number = None

        self.root.config(menu=self.menuBar)
        if os.path.exists('../models/info.csv'):
            data_frame = dict(pd.read_csv('../models/info.csv').values)
            self.training_dir = data_frame['training_dir']
            self.dir_var.set(self.training_dir)
            if data_frame['use_alignment'] == 'True':
                self.use_alignment = True
            else:
                self.use_alignment = False

            if data_frame['all_data'] == 'True':
                self.fit_all_data = True
                self.fit_data_var.set(TRUE)
            else:
                self.fit_all_data = False
                self.fit_data_var.set(FALSE)

            self.alignment_var.set(self.use_alignment)
            self.image_size = data_frame['image_size']

        if self.training_dir is not None:
            self.paths_array, self.labels_array = search_dir(self.training_dir)

        if self.enable_train is False and self.enable_test is False and self.enable_retrieval is False:
            self.canvas = Canvas(self.root, width=1500, height=780)
            self.image = Image.open('../resources/MainBG.jpg').resize((1500, 800))
            self.canvas.image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')
            self.canvas.place(x=0, y=0)
        self.root.mainloop()

    def create_train_ui(self):
        self.display = []

        if self.dir_var.get() == '':
            self.dir_var.set('')
        else:
            self.paths_array, self.labels_array = search_dir(self.training_dir)
        if self.language == 'chinese':
            Label(self.root, text='训练集目录', font=('楷体', 12), fg='blue').place(x=10, y=40)
            Button(self.root, text='选择', font=('楷体', 12), command=self.choose_dir).place(x=395, y=36)
        else:
            Label(self.root, text='Training set directory', font=('Times New Roman', 12), fg='blue').place(x=10, y=40)
            Button(self.root, text='Choose', font=('Times New Roman', 12), command=self.choose_dir).place(x=380, y=36)
        Entry(self.root, width=53, textvariable=self.dir_var, font=('Times New Roman', 12)).place(x=11, y=70)

        if self.language == 'chinese':
            Label(self.root, text='图像尺寸', font=('楷体', 12), fg='blue').place(x=10, y=120)
        else:
            Label(self.root, text='Image size', font=('Times New Roman', 12), fg='blue').place(x=10, y=115)
        Scale(self.root, font=('Times New Roman', 12), from_=144, to=256, width=20, orient=HORIZONTAL, length=430,
              showvalue=1, tickinterval=28, resolution=1, command=self.choose_image_size).place(x=10, y=140)

        if self.pro_info_var.get() == '':
            self.pro_info_var.set(TRUE)
        if self.language == 'chinese':
            Label(self.root, text='输出过程信息', font=('楷体', 12), fg='blue').place(x=10, y=230)
        else:
            Label(self.root, text='Print process information', font=('Times New Roman', 12),
                  fg='blue').place(x=10, y=230)
        Radiobutton(self.root, text='Yes', font=('Times New Roman', 12), variable=self.pro_info_var, value=TRUE,
                    command=self.choose_info_list_var).place(x=320, y=230)
        Radiobutton(self.root, text='No', font=('Times New Roman', 12), variable=self.pro_info_var, value=FALSE,
                    command=self.choose_info_list_var).place(x=400, y=230)

        if self.language == 'chinese':
            Label(self.root, text='是否利用全部数据去拟合', font=('楷体', 12), fg='blue').place(x=10, y=280)
        else:
            Label(self.root, text='Use all data to fit', font=('Times New Roman', 12), fg='blue').place(x=10, y=280)
        Radiobutton(self.root, text='Yes', font=('Times New Roman', 12), variable=self.fit_data_var, value=TRUE,
                    command=self.choose_fit_data_var).place(x=320, y=280)
        Radiobutton(self.root, text='No', font=('Times New Roman', 12), variable=self.fit_data_var, value=FALSE,
                    command=self.choose_fit_data_var).place(x=400, y=280)

        if self.clf_info_var.get() == '':
            self.clf_info_var.set(TRUE)

        if self.language == 'chinese':
            Label(self.root, text='输出分类器的准确度', font=('楷体', 12), fg='blue').place(x=10, y=330)
        else:
            Label(self.root, text='Print classifiers\' accuracy', font=('Times New Roman', 12),
                  fg='blue').place(x=10, y=330)
        Radiobutton(self.root, text='Yes', font=('Times New Roman', 12), variable=self.clf_info_var, value=TRUE,
                    command=self.choose_clf_info_var).place(x=320, y=330)
        Radiobutton(self.root, text='No', font=('Times New Roman', 12), variable=self.clf_info_var, value=FALSE,
                    command=self.choose_clf_info_var).place(x=400, y=330)

        if self.alignment_var.get() == '':
            self.alignment_var.set(TRUE)

        if self.language == 'chinese':
            Label(self.root, text='是否使用人脸校正', font=('楷体', 12), fg='blue').place(x=10, y=380)
        else:
            Label(self.root, text='Use face alignment', font=('Times New Roman', 12), fg='blue').place(x=10, y=380)
        Radiobutton(self.root, text='Yes', font=('Times New Roman', 12), variable=self.alignment_var, value=TRUE,
                    command=self.choose_alignment_var).place(x=320, y=380)
        Radiobutton(self.root, text='No', font=('Times New Roman', 12), variable=self.alignment_var, value=FALSE,
                    command=self.choose_alignment_var).place(x=400, y=380)

        if self.language == 'chinese':
            Label(self.root, text='查看训练参数', font=('楷体', 12), fg='blue').place(x=165, y=425)
            self.training_parameters = Listbox(self.root, font=('楷体', 12), width=54, height=12)
        else:
            Label(self.root, text='Check training parameters', font=('Times New Roman', 12),
                  fg='blue').place(x=140, y=425)
            self.training_parameters = Listbox(self.root, font=('Times New Roman', 12), width=54, height=12)
        self.training_parameters.place(x=10, y=480)

        self.training_par_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.training_par_y_scrollbar = Scrollbar(self.root)
        self.training_par_x_scrollbar.place(x=10, y=460, width=435)
        self.training_par_y_scrollbar.place(x=446, y=476, height=214)
        self.training_parameters.config(xscrollcommand=self.training_par_x_scrollbar.set,
                                        yscrollcommand=self.training_par_y_scrollbar.set)
        self.training_par_x_scrollbar.config(command=self.training_parameters.xview)
        self.training_par_y_scrollbar.config(command=self.training_parameters.yview)

        if self.language == 'chinese':
            Button(self.root, text='开始', font=('楷体', 12), fg='blue', command=self.start_train).place(x=145, y=730)
            Button(self.root, text='停止', font=('楷体', 12), fg='blue', command=self.stop_train).place(x=265, y=730)
        else:
            Button(self.root, text='Start', font=('Times New Roman', 12), fg='blue',
                   command=self.start_train).place(x=145, y=730)
            Button(self.root, text='Stop', font=('Times New Roman', 12), fg='blue',
                   command=self.stop_train).place(x=265, y=730)

        self.output_frame = Frame(self.root)
        self.output_frame.pack(side=RIGHT)
        self.pro_info_frame = Frame(self.output_frame)
        self.pro_info_frame.pack(side=TOP)
        self.clf_info_frame = Frame(self.output_frame)
        self.clf_info_frame.pack(side=TOP)

        if self.language == 'chinese':
            Label(self.pro_info_frame, text='输出过程信息', font=('楷体', 12), fg='blue').pack(side=TOP)
        else:
            Label(self.pro_info_frame, text='Print process information', font=('Times New Roman', 12),
                  fg='blue').pack(side=TOP)
        self.pro_y_scrollbar = Scrollbar(self.pro_info_frame)
        self.pro_y_scrollbar.pack(side=RIGHT, fill=Y)
        self.pro_x_scrollbar = Scrollbar(self.pro_info_frame, orient=HORIZONTAL)
        self.pro_x_scrollbar.pack(side=TOP, fill=X)

        if self.language == 'chinese':
            self.pro_info_box = Listbox(self.pro_info_frame, font=('楷体', 12),
                                        yscrollcommand=self.pro_y_scrollbar.set, width=100, height=17)
        else:
            self.pro_info_box = Listbox(self.pro_info_frame, font=('Times New Roman', 12),
                                        yscrollcommand=self.pro_y_scrollbar.set, width=100, height=17)
        self.pro_info_box.pack(side=TOP, fill=BOTH)
        self.pro_y_scrollbar.config(command=self.pro_info_box.yview)
        self.pro_x_scrollbar.config(command=self.pro_info_box.xview)
        self.pro_info_box.config(xscrollcommand=self.pro_x_scrollbar.set)

        if self.language == 'chinese':
            Label(self.clf_info_frame, text='输出分类器准确度', font=('楷体', 12), fg='blue').pack(side=TOP)
        else:
            Label(self.clf_info_frame, text='Print classifiers\' accuracy', font=('Times New Roman', 12),
                  fg='blue').pack(side=TOP)
        self.clf_y_scrollbar = Scrollbar(self.clf_info_frame)
        self.clf_y_scrollbar.pack(side=RIGHT, fill=Y)
        self.clf_x_scrollbar = Scrollbar(self.clf_info_frame, orient=HORIZONTAL)
        self.clf_x_scrollbar.pack(side=TOP, fill=X)
        if self.language == 'chinese':
            self.clf_info_box = Listbox(self.clf_info_frame, font=('楷体', 12),
                                        xscrollcommand=self.clf_x_scrollbar.set,
                                        yscrollcommand=self.clf_y_scrollbar.set, width=100, height=17)
        else:
            self.clf_info_box = Listbox(self.clf_info_frame, font=('Times New Roman', 12),
                                        xscrollcommand=self.clf_x_scrollbar.set,
                                        yscrollcommand=self.clf_y_scrollbar.set, width=100, height=17)
        self.clf_info_box.pack(side=TOP, fill=BOTH)
        self.clf_x_scrollbar.config(command=self.clf_info_box.xview)
        self.clf_y_scrollbar.config(command=self.clf_info_box.yview)

        self.insert_all_train_parameters()

    def destroy_current_ui(self):
        for widget in self.root.winfo_children():
            if widget is not self.menuBar:
                # print(widget)
                widget.destroy()

    def destroy_choose_label_ui(self):
        for widget in self.choose_ui_widgets:
            widget.destroy()

    def create_choose_label_ui(self):
        if self.language == 'chinese':
            self.choose_ui_widgets.append(Label(self.root, text='标签数据库', font=('楷体', 12), fg='blue'))
        else:
            self.choose_ui_widgets.append(Label(self.root, text='All labels', font=('Times New Roman', 12),
                                                fg='blue'))
        self.choose_ui_widgets[-1].place(x=515, y=30)

        if self.language == 'chinese':
            self.labels_box = Listbox(self.root, font=('楷体', 12), width=20, height=15)
        else:
            self.labels_box = Listbox(self.root, font=('Times New Roman', 12), width=20, height=15)
        self.labels_box.place(x=515, y=80)
        self.labels_box_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.labels_box_y_scrollbar = Scrollbar(self.root)
        self.labels_box_x_scrollbar.place(x=515, y=60, width=170)
        self.labels_box_y_scrollbar.place(x=679, y=78, height=263)
        self.labels_box.config(xscrollcommand=self.labels_box_x_scrollbar.set,
                               yscrollcommand=self.labels_box_y_scrollbar.set)
        self.labels_box_x_scrollbar.config(command=self.labels_box.xview)
        self.labels_box_y_scrollbar.config(command=self.labels_box.yview)

        self.choose_ui_widgets.append(self.labels_box)
        self.choose_ui_widgets.append(self.labels_box_x_scrollbar)
        self.choose_ui_widgets.append(self.labels_box_y_scrollbar)

        self.labels_box.bind('<ButtonRelease-1>', self.mouse_update_choose)
        # self.labels_box.bind('<Key>', self.key_update_number)

        if self.language == 'chinese':
            self.choose_ui_widgets.append(Button(self.root, text='选择', font=('楷体', 12), command=self.on_choose))
        else:
            self.choose_ui_widgets.append(Button(self.root, text='Choose', font=('Times New Roman', 12),
                                                 command=self.on_choose))
        self.choose_ui_widgets[-1].place(x=615, y=25)

        if self.labels_array:
            self.insert_labels(self.labels_box, self.labels_array)

        if self.language == 'chinese':
            self.choose_ui_widgets.append(Label(self.root, text='被选中标签', font=('楷体', 12), fg='blue'))
        else:
            self.choose_ui_widgets.append(Label(self.root, text='Chosen labels',
                                                font=('Times New Roman', 12), fg='blue'))
        self.choose_ui_widgets[-1].place(x=515, y=400)

        self.choose_labels = []
        if self.language == 'chinese':
            self.chooses_box = Listbox(self.root, font=('楷体', 12), width=20, height=15)
        else:
            self.chooses_box = Listbox(self.root, font=('Times New Roman', 12), width=20, height=15)
        self.chooses_box.place(x=515, y=450)
        self.chooses_box_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.chooses_box_y_scrollbar = Scrollbar(self.root)
        self.chooses_box_x_scrollbar.place(x=515, y=430, width=170)
        self.chooses_box_y_scrollbar.place(x=679, y=448, height=263)
        self.chooses_box.config(xscrollcommand=self.chooses_box_x_scrollbar.set,
                                yscrollcommand=self.chooses_box_y_scrollbar.set)
        self.chooses_box_x_scrollbar.config(command=self.chooses_box.xview)
        self.chooses_box_y_scrollbar.config(command=self.chooses_box.yview)

        self.choose_ui_widgets.append(self.chooses_box)
        self.choose_ui_widgets.append(self.chooses_box_x_scrollbar)
        self.choose_ui_widgets.append(self.chooses_box_y_scrollbar)

        self.chooses_box.bind('<ButtonRelease-1>', self.mouse_update_delete)
        if self.language == 'chinese':
            self.choose_ui_widgets.append(Button(self.root, text='删除', font=('楷体', 12), command=self.on_delete))
            self.choose_ui_widgets[-1].place(x=615, y=395)
        else:
            self.choose_ui_widgets.append(Button(self.root, text='Delete', font=('Times New Roman', 12),
                                                 command=self.on_delete))
            self.choose_ui_widgets[-1].place(x=620, y=395)

    def create_test_ui(self):

        if self.image_path_var.get() == '':
            self.image_path_var.set('')

        if self.language == 'chinese':
            Label(self.root, text='预测图像的路径', font=('楷体', 12), fg='blue').place(x=10, y=30)
            Button(self.root, text='选择', font=('楷体', 12), command=self.choose_file).place(x=418, y=26)
        else:
            Label(self.root, text='The path to the image', font=('Times New Roman', 12), fg='blue').place(x=10, y=30)
            Button(self.root, text='Choose', font=('Times New Roman', 12), command=self.choose_file).place(x=400, y=26)
        Entry(self.root, width=56, textvariable=self.image_path_var, font=('Times New Roman', 12)).place(x=10, y=60)

        if self.test_state_var.get() == '':
            self.test_state_var.set('recognition')
        if self.test_state_var.get() != 'recognition':
            self.create_choose_label_ui()

        if self.language == 'chinese':
            Label(self.root, text='模式', font=('楷体', 12), fg='blue').place(x=10, y=95)
            Checkbutton(self.root, text='人脸验证', font=('楷体', 12), variable=self.test_state_var,
                        onvalue='verification', offvalue=0, command=self.choose_state).place(x=10, y=115)
            Checkbutton(self.root, text='人脸识别', font=('楷体', 12), variable=self.test_state_var,
                        onvalue='recognition', offvalue=0, command=self.choose_state).place(x=200, y=115)
            Checkbutton(self.root, text='人脸查找', font=('楷体', 12), variable=self.test_state_var,
                        onvalue='search', offvalue=0, command=self.choose_state).place(x=390, y=115)
        else:
            Label(self.root, text='Mode', font=('Times New Roman', 12), fg='blue').place(x=10, y=95)
            Checkbutton(self.root, text='Face Verification', font=('Times New Roman', 12), variable=self.test_state_var,
                        onvalue='verification', offvalue=0, command=self.choose_state).place(x=10, y=115)
            Checkbutton(self.root, text='Face Recognition', font=('Times New Roman', 12), variable=self.test_state_var,
                        onvalue='recognition', offvalue=0, command=self.choose_state).place(x=190, y=115)
            Checkbutton(self.root, text='Face Search', font=('Times New Roman', 12), variable=self.test_state_var,
                        onvalue='search', offvalue=0, command=self.choose_state).place(x=370, y=115)

        if self.image_path_var.get() != '':
            self.display = []
            if self.test_state_var.get() == 'recognition':
                self.show_image(self.test_image_path, size=(800, 700), x=640, y=40)
            else:
                self.show_image(self.test_image_path, size=(700, 700), x=740, y=40)

        if self.language == 'chinese':
            Label(self.root, text='图像尺寸', font=('楷体', 12), fg='blue').place(x=10, y=150)
            Scale(self.root, font=('Times New Roman', 12), from_=144, to=256, width=20, orient=HORIZONTAL, length=450,
                  showvalue=1, tickinterval=28, resolution=1, command=self.choose_image_size).place(x=10, y=170)
        else:
            Label(self.root, text='Image size', font=('Times New Roman', 12), fg='blue').place(x=10, y=150)
            Scale(self.root, font=('Times New Roman', 12), from_=144, to=256, width=20, orient=HORIZONTAL, length=450,
                  showvalue=1, tickinterval=28, resolution=1, command=self.choose_image_size).place(x=10, y=170)

        if self.fit_data_var.get() == '':
            self.fit_data_var.set(FALSE)

        if self.language == 'chinese':
            Label(self.root, text='使用通过全部数据训练出的模型', font=('楷体', 12), fg='blue').place(x=10, y=245)
        else:
            Label(self.root, text='Use model that trained by all data',
                  font=('Times New Roman', 12), fg='blue').place(x=10, y=245)
        Radiobutton(self.root, text='Yes', font=('Times New Roman', 12), variable=self.fit_data_var, value=TRUE,
                    command=self.choose_fit_data_var).place(x=340, y=245)
        Radiobutton(self.root, text='No', font=('Times New Roman', 12), variable=self.fit_data_var, value=FALSE,
                    command=self.choose_fit_data_var).place(x=400, y=245)

        if self.language == 'chinese':
            Label(self.root, text='检查模型参数', font=('楷体', 12), fg='blue').place(x=176, y=270)
            self.test_parameters = Listbox(self.root, font=('楷体', 12), width=56, height=8)
        else:
            Label(self.root, text='Check model\'s parameters', font=('Times New Roman', 12),
                  fg='blue').place(x=150, y=270)
            self.test_parameters = Listbox(self.root, font=('Times New Roman', 12), width=56, height=8)

        self.test_parameters.place(x=10, y=320)
        self.test_par_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.test_par_y_scrollbar = Scrollbar(self.root)
        self.test_par_x_scrollbar.place(x=10, y=300, width=458)
        self.test_par_y_scrollbar.place(x=462, y=315, height=150)
        self.test_parameters.config(xscrollcommand=self.test_par_x_scrollbar.set,
                                    yscrollcommand=self.test_par_y_scrollbar.set)
        self.test_par_x_scrollbar.config(command=self.test_parameters.xview)
        self.test_par_y_scrollbar.config(command=self.test_parameters.yview)

        if self.language == 'chinese':
            Button(self.root, text='开始', font=('楷体', 12), fg='blue', command=self.start_test).place(x=150, y=490)
            Button(self.root, text='停止', font=('楷体', 12), fg='blue', command=self.stop_test).place(x=270, y=490)
            Label(self.root, text='预测结果', font=('楷体', 12), fg='blue').place(x=200, y=550)
            self.answer_box = Listbox(self.root, font=('楷体', 12), width=56, height=8)
        else:
            Button(self.root, text='Start', font=('Times New Roman', 12), fg='blue',
                   command=self.start_test).place(x=150, y=490)
            Button(self.root, text='Stop', font=('Times New Roman', 12), fg='blue',
                   command=self.stop_test).place(x=270, y=490)
            Label(self.root, text='Prediction', font=('Times New Roman', 12), fg='blue').place(x=200, y=550)
            self.answer_box = Listbox(self.root, font=('Times New Roman', 12), width=56, height=8)

        self.answer_box.place(x=10, y=600)
        self.answer_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.answer_y_scrollbar = Scrollbar(self.root)
        self.answer_x_scrollbar.place(x=10, y=578, width=458)
        self.answer_y_scrollbar.place(x=462, y=596, height=150)
        self.answer_box.config(xscrollcommand=self.answer_x_scrollbar.set, yscrollcommand=self.answer_y_scrollbar.set)
        self.answer_x_scrollbar.config(command=self.answer_box.xview)
        self.answer_y_scrollbar.config(command=self.answer_box.yview)

        self.insert_all_test_parameters()

    def create_retrieval_ui(self):
        self.display = []

        if self.language == 'chinese':
            Label(self.root, text='样本标签', font=('楷体', 12), fg='blue').place(x=55, y=20)
            self.labels_box = Listbox(self.root, font=('楷体', 12), width=20, height=20)
        else:
            Label(self.root, text='All labels', font=('Times New Roman', 12), fg='blue').place(x=60, y=20)
            self.labels_box = Listbox(self.root, font=('Times New Roman', 12), width=20, height=20)

        self.labels_box.place(x=10, y=80)
        self.labels_box_x_scrollbar = Scrollbar(self.root, orient=HORIZONTAL)
        self.labels_box_y_scrollbar = Scrollbar(self.root)
        self.labels_box_x_scrollbar.place(x=10, y=60, width=170)
        self.labels_box_y_scrollbar.place(x=174, y=78, height=350)
        self.labels_box.config(xscrollcommand=self.labels_box_x_scrollbar.set,
                               yscrollcommand=self.labels_box_y_scrollbar.set)
        self.labels_box_x_scrollbar.config(command=self.labels_box.xview)
        self.labels_box_y_scrollbar.config(command=self.labels_box.yview)

        self.labels_box.bind('<ButtonRelease-1>', self.mouse_update_number)
        # self.labels_box.bind('<Key>', self.key_update_number)

        if self.labels_array:
            self.insert_labels(self.labels_box, self.labels_array)

        if self.language == 'chinese':
            Label(self.root, text='标签搜索', font=('楷体', 12), fg='blue').place(x=50, y=520)
            Entry(self.root, width=12, textvariable=self.label_var, font=('楷体', 12)).place(x=35, y=560)
            Button(self.root, text='查找', font=('楷体', 12), command=self.on_search).place(x=30, y=640)
            Button(self.root, text='显示', font=('楷体', 12), command=self.on_show).place(x=100, y=640)
        else:
            Label(self.root, text='Search label', font=('Times New Roman', 12), fg='blue').place(x=45, y=520)
            Entry(self.root, width=12, textvariable=self.label_var, font=('Times New Roman', 12)).place(x=40, y=560)
            Button(self.root, text='Search', font=('Times New Roman', 12), command=self.on_search).place(x=20, y=640)
            Button(self.root, text='Display', font=('Times New Roman', 12), command=self.on_show).place(x=100, y=640)

    def mouse_update_number(self, event):
        if self.labels_array is not None:
            self.label_var.set(self.labels_box.get(self.labels_box.curselection()))
        pass

    def enable_train_ui(self):
        self.enable_test = False
        self.enable_retrieval = False
        if self.enable_train:
            if self.language == 'chinese':
                mb.showwarning('警告', '当前的用户界面已为训练界面！')
            else:
                mb.showwarning('Warning', 'The current user interface is training interface!')
        else:
            self.destroy_current_ui()
            self.enable_train = True
            self.create_train_ui()

    def enable_test_ui(self):
        self.enable_train = False
        self.enable_retrieval = False
        if self.enable_test:
            if self.language == 'chinese':
                mb.showwarning('警告', '当前的用户界面已为预测界面！')
            else:
                mb.showwarning('Warning', 'The current user interface is predict interface!')
        else:
            self.destroy_current_ui()
            self.enable_test = True
            self.create_test_ui()

    def enable_retrieval_ui(self):
        self.enable_test = False
        self.enable_train = False
        if self.enable_retrieval:
            if self.language == 'chinese':
                mb.showwarning('警告', '当前的用户界面已为检索界面！')
            else:
                mb.showwarning('Warning', 'The current user interface is retrieval interface!')
        else:
            if self.training_dir is None:
                if self.language == 'chinese':
                    mb.showwarning('警告', '数据库为空！')
                else:
                    mb.showwarning('Warning', 'The database is empty!')
            else:
                self.paths_array, self.labels_array = search_dir(self.training_dir)
            self.destroy_current_ui()
            self.enable_retrieval = True
            self.create_retrieval_ui()

    def start_train(self):
        # print(type(int(self.image_size)), int(self.image_size)) self.image_size -> str
        if self.training_dir is None or self.training_dir == '':
            if self.language == 'chinese':
                mb.showerror("错误", '训练集目录非法！')
            else:
                mb.showerror("Error", 'Training set directory is not valid!')

        else:
            if self.language == 'chinese':
                if mb.askyesno("提示", '你要开始训练数据吗？'):
                    self.pro_info_box.delete(0, END)
                    self.clf_info_box.delete(0, END)
                    _, _, _, _ = training_images.train(self.training_dir, fr=self, image_size=int(self.image_size),
                                                       process_output=self.print_pro_info,
                                                       all_data=self.fit_all_data, clf_output=self.print_clf_info,
                                                       use_alignment=self.use_alignment, language=self.language)
            else:
                if mb.askyesno("Info", 'Do you want to start training?'):
                    self.pro_info_box.delete(0, END)
                    self.clf_info_box.delete(0, END)
                    _, _, _, _ = training_images.train(self.training_dir, fr=self, image_size=int(self.image_size),
                                                       process_output=self.print_pro_info,
                                                       all_data=self.fit_all_data, clf_output=self.print_clf_info,
                                                       use_alignment=self.use_alignment, language=self.language)

    def stop_train(self):
        pass

    def check_error(self):
        if self.test_image_path is None or self.test_image_path == '':
            if self.language == 'chinese':
                mb.showerror("错误", '测试集图像路径非法！')
            else:
                mb.showerror("Error", 'The path to the image is not valid!')
            return False

        if self.test_state == 'verification':
            if len(self.choose_labels) > 1:
                if self.language == 'chinese':
                    mb.showerror("错误", "人脸验证只能选择一个标签！")
                else:
                    mb.showerror("Error", "Face Verification only choose one label!")
                return False
            elif len(self.choose_labels) == 0:
                if self.language == 'chinese':
                    mb.showerror("错误", "人脸识别必须选择至少一个标签！")
                else:
                    mb.showerror("Error", "Face Recognition must choose at least one label!")
                return False

        if self.test_state == 'search':
            if len(self.choose_labels) == 0:
                if self.language == 'chinese':
                    mb.showerror("错误", "人脸查找必须选择至少一个标签！")
                else:
                    mb.showerror("Error", "Face Search must choose at least one label!")
                return False
        return True

    def start_test(self):
        if self.check_error():
            if self.language == 'chinese':
                if mb.askyesno("预测", '你要开始预测吗？'):
                    self.answer_box.delete(0, END)
                    self.answer, self.predict_image_data = test_images.recognition(self.test_image_path,
                                                                                   self,
                                                                                   state=self.test_state,
                                                                                   used_labels=self.choose_labels,
                                                                                   image_size=int(self.image_size),
                                                                                   all_data=self.fit_all_data,
                                                                                   language=self.language)
                    cv2.imwrite('../datasets/answers/answer.jpg', cv2.cvtColor(self.predict_image_data,
                                                                  cv2.COLOR_RGB2BGR))
                    self.display = []
                    if self.test_state == 'recognition':
                        self.show_image('../datasets/answers/answer.jpg', size=(800, 700), x=640, y=40)
                    else:
                        self.show_image('../datasets/answers/answer.jpg', size=(700, 700), x=740, y=40)
            else:
                if mb.askyesno("Predict", 'Do you want to start predicting?'):
                    self.answer_box.delete(0, END)
                    self.answer, self.predict_image_data = test_images.recognition(self.test_image_path,
                                                                                   self,
                                                                                   state=self.test_state,
                                                                                   used_labels=self.choose_labels,
                                                                                   image_size=int(self.image_size),
                                                                                   all_data=self.fit_all_data,
                                                                                   language=self.language)
                    cv2.imwrite('../datasets/answers/answer.jpg', cv2.cvtColor(self.predict_image_data,
                                                                  cv2.COLOR_RGB2BGR))
                    self.display = []
                    if self.test_state == 'recognition':
                        self.show_image('../datasets/answers/answer.jpg', size=(800, 700), x=640, y=40)
                    else:
                        self.show_image('../datasets/answers/answer.jpg', size=(700, 700), x=740, y=40)

    def stop_test(self):
        pass

    def choose_state(self):
        self.test_state = self.test_state_var.get()
        if self.test_state == 'recognition':
            self.destroy_choose_label_ui()
        else:
            self.create_choose_label_ui()

        if self.test_image_path != '' and self.test_image_path is not None:
            self.display = []
            if self.test_state_var.get() == 'recognition':
                self.show_image(self.test_image_path, size=(800, 700), x=640, y=40)
            else:
                self.show_image(self.test_image_path, size=(700, 700), x=740, y=40)

        self.insert_all_test_parameters()

    def mouse_update_choose(self, event):
        if self.labels_array is not None:
            self.choose_label_var.set(self.labels_box.get(self.labels_box.curselection()))
        pass

    def mouse_update_delete(self, event):
        if self.labels_array is not None:
            self.delete_label_var.set(self.chooses_box.get(self.chooses_box.curselection()))
        pass

    def on_choose(self):
        if self.choose_label_var.get() not in self.choose_labels:
            self.choose_labels.append(self.choose_label_var.get())
        else:
            if self.language == 'chinese':
                mb.showinfo('提示', '该标签已被选中！')
            else:
                mb.showinfo('Info', 'This label has been chosen!')
        self.insert_labels(self.chooses_box, self.choose_labels)

    def on_delete(self):
        if self.delete_label_var.get() in self.choose_labels:
            del self.choose_labels[self.choose_labels.index(self.delete_label_var.get())]
        self.insert_labels(self.chooses_box, self.choose_labels)

    @ staticmethod
    def insert_labels(box, labels):
        box.delete(0, END)
        for label in labels:
            box.insert(END, label)

    def choose_dir(self):
        self.training_dir = askdirectory()
        self.dir_var.set(self.training_dir)
        self.paths_array, self.labels_array = search_dir(self.training_dir)
        self.insert_all_train_parameters()

    def choose_file(self):
        temp_path = askopenfilename()
        if temp_path is not None and temp_path != '':
            if temp_path[-4:] in ['.jpg', '.JPG', '.PNG', '.png']:
                self.test_image_path = temp_path
                self.image_path_var.set(self.test_image_path)
                if self.test_image_path != '':
                    self.display = []
                    if self.test_state_var.get() == 'recognition':
                        self.show_image(self.test_image_path, size=(800, 700), x=640, y=40)
                    else:
                        self.show_image(self.test_image_path, size=(700, 700), x=740, y=40)
                self.insert_all_test_parameters()
            else:
                if self.language == 'chinese':
                    mb.showerror('错误', '请选择一个图像文件！')
                else:
                    mb.showerror('Error', 'Please choose an image file!')

    def choose_image_size(self, image_size):
        self.image_size = image_size
        if self.enable_train:
            self.insert_all_train_parameters()
        else:
            self.insert_all_test_parameters()

    def choose_info_list_var(self):
        # print(type(self.pro_info_var.get())) -> str
        if self.pro_info_var.get() == '1':
            self.print_pro_info = True

        else:
            self.print_pro_info = False
        if self.enable_train:
            self.insert_all_train_parameters()
        else:
            self.insert_all_test_parameters()

    def choose_fit_data_var(self):
        if self.fit_data_var.get() == '1':
            self.fit_all_data = True

        else:
            self.fit_all_data = False
        if self.enable_train:
            self.insert_all_train_parameters()
        else:
            self.insert_all_test_parameters()

    def choose_clf_info_var(self):
        if self.clf_info_var.get() == '1':
            self.print_clf_info = True

        else:
            self.print_clf_info = False
        if self.enable_train:
            self.insert_all_train_parameters()
        else:
            self.insert_all_test_parameters()

    def choose_alignment_var(self):
        if self.alignment_var.get() == '1':
            self.use_alignment = True

        else:
            self.use_alignment = False
        if self.enable_train:
            self.insert_all_train_parameters()
        else:
            self.insert_all_test_parameters()

    def insert_all_train_parameters(self):
        self.training_parameters.delete(0, END)
        if self.language == 'chinese':
            self.training_parameters.insert(0, '训练参数如下：')
            self.training_parameters.itemconfig(0, fg='red')
            self.training_parameters.insert(1, '')

            self.training_parameters.insert(2, '训练集目录：')
            self.training_parameters.itemconfig(2, fg='blue')
            self.training_parameters.insert(3, '{}'.format(self.training_dir))

            self.training_parameters.insert(4, '图像尺寸：')
            self.training_parameters.itemconfig(4, fg='blue')
            self.training_parameters.insert(5, '{}'.format(self.image_size))

            self.training_parameters.insert(6, '是否输出过程信息：')
            self.training_parameters.itemconfig(6, fg='blue')
            self.training_parameters.insert(7, '{}'.format(self.print_pro_info))

            self.training_parameters.insert(8, '是否用全部数据集去训练：')
            self.training_parameters.itemconfig(8, fg='blue')
            self.training_parameters.insert(9, '{}'.format(self.fit_all_data))

            self.training_parameters.insert(10, '是否输出分类器信息：')
            self.training_parameters.itemconfig(10, fg='blue')
            self.training_parameters.insert(11, '{}'.format(self.print_clf_info))

            self.training_parameters.insert(12, '是否使用人脸校正：')
            self.training_parameters.itemconfig(12, fg='blue')
            self.training_parameters.insert(13, '{}'.format(self.use_alignment))
            self.training_parameters.insert(14, '')

            self.training_parameters.insert(15, '请检查训练模型的参数！')
            self.training_parameters.insert(16, '如果参数无误，请点击\'开始\'按钮！')
            self.training_parameters.itemconfig(15, fg='red')
            self.training_parameters.itemconfig(16, fg='red')
        else:
            self.training_parameters.insert(0, 'Training parameters are as follows:')
            self.training_parameters.itemconfig(0, fg='red')
            self.training_parameters.insert(1, '')

            self.training_parameters.insert(2, 'Training set directory:')
            self.training_parameters.itemconfig(2, fg='blue')
            self.training_parameters.insert(3, '{}'.format(self.training_dir))

            self.training_parameters.insert(4, 'Image size:')
            self.training_parameters.itemconfig(4, fg='blue')
            self.training_parameters.insert(5, '{}'.format(self.image_size))

            self.training_parameters.insert(6, 'Print process information:')
            self.training_parameters.itemconfig(6, fg='blue')
            self.training_parameters.insert(7, '{}'.format(self.print_pro_info))

            self.training_parameters.insert(8, 'Use all data to fit:')
            self.training_parameters.itemconfig(8, fg='blue')
            self.training_parameters.insert(9, '{}'.format(self.fit_all_data))

            self.training_parameters.insert(10, 'Print classifiers\' accuracy:')
            self.training_parameters.itemconfig(10, fg='blue')
            self.training_parameters.insert(11, '{}'.format(self.print_clf_info))

            self.training_parameters.insert(12, 'Use face alignment:')
            self.training_parameters.itemconfig(12, fg='blue')
            self.training_parameters.insert(13, '{}'.format(self.use_alignment))
            self.training_parameters.insert(14, '')

            self.training_parameters.insert(15, 'Please check model\'s parameters!')
            self.training_parameters.insert(16, 'If the parameters are correct, click the \'Start\' button!')
            self.training_parameters.itemconfig(15, fg='red')
            self.training_parameters.itemconfig(16, fg='red')

    def output_all_train_parameters(self):
        print('Training data directory: {}'.format(self.training_dir))
        print('Image size: {}'.format(self.image_size))
        print('Process information variable: {}'.format(self.print_pro_info))
        print('Fit data variable: {}'.format(self.fit_all_data))
        print('Classifier information variable: {}'.format(self.print_clf_info))

    def insert_all_test_parameters(self):
        self.test_parameters.delete(0, END)
        if self.language == 'chinese':
            self.test_parameters.insert(0, '预测参数如下：')
            self.test_parameters.itemconfig(0, fg='red')
            self.test_parameters.insert(1, '')

            self.test_parameters.insert(2, '图像路径：')
            self.test_parameters.itemconfig(2, fg='blue')
            self.test_parameters.insert(3, '{}'.format(self.test_image_path))

            self.test_parameters.insert(4, '模式：')
            self.test_parameters.itemconfig(4, fg='blue')
            self.test_parameters.insert(5, '{}'.format(self.test_state))

            self.test_parameters.insert(6, '图像尺寸：')
            self.test_parameters.itemconfig(6, fg='blue')
            self.test_parameters.insert(7, '{}'.format(self.image_size))

            self.test_parameters.insert(8, '是否使用通过全部数据训练得到的模型：')
            self.test_parameters.itemconfig(8, fg='blue')
            self.test_parameters.insert(9, '{}'.format(self.fit_all_data))

            self.test_parameters.insert(10, '')

            self.test_parameters.insert(11, '请检查模型的参数！')
            self.test_parameters.insert(12, '如果参数无误，请点击\'开始\'按钮！')
            self.test_parameters.itemconfig(11, fg='red')
            self.test_parameters.itemconfig(12, fg='red')
        else:
            self.test_parameters.insert(0, 'Prediction parameters are as follows:')
            self.test_parameters.itemconfig(0, fg='red')
            self.test_parameters.insert(1, '')

            self.test_parameters.insert(2, 'Image path:')
            self.test_parameters.itemconfig(2, fg='blue')
            self.test_parameters.insert(3, '{}'.format(self.test_image_path))

            self.test_parameters.insert(4, 'Mode: ')
            self.test_parameters.itemconfig(4, fg='blue')
            self.test_parameters.insert(5, '{}'.format(self.test_state))

            self.test_parameters.insert(6, 'Image size:')
            self.test_parameters.itemconfig(6, fg='blue')
            self.test_parameters.insert(7, '{}'.format(self.image_size))

            self.test_parameters.insert(8, 'Use model that trained by all data:')
            self.test_parameters.itemconfig(8, fg='blue')
            self.test_parameters.insert(9, '{}'.format(self.fit_all_data))

            self.test_parameters.insert(10, '')

            self.test_parameters.insert(11, 'Please check the model\'s parameters!')
            self.test_parameters.insert(12, 'If the parameters are correct, click the \'Start\' button!')
            self.test_parameters.itemconfig(11, fg='red')
            self.test_parameters.itemconfig(12, fg='red')

    def output_all_test_parameters(self):
        print('Test image path: {}'.format(self.test_image_path))
        print('Test state: {}'.format(self.test_state))
        print('Image size: {}'.format(self.image_size))
        print('Use model which trained by all data: {}'.format(self.fit_all_data))

    def show_image(self, image_path, size=(960, 760), x=520, y=5):
        image = Image.open(image_path)
        image = image.resize(size)
        self.display.append(ImageTk.PhotoImage(image))
        Label(self.root, image=self.display[-1]).place(x=x, y=y)

    def on_exit(self):
        if self.language == 'chinese':
            if mb.askyesno("退出", '你要退出该应用程序吗？'):
                self.root.destroy()
        else:
            if mb.askyesno("Exit", 'Are you sure to exit?'):
                self.root.destroy()

    def show_information(self, info_list, pro_output=False, clf_output=False, predict=False):
        if pro_output:
            self.pro_info_box.delete(0, END)
            for info in info_list:
                self.pro_info_box.insert(END, info)

        if clf_output:
            self.clf_info_box.delete(0, END)
            for info in info_list:
                self.clf_info_box.insert(END, info)

        if predict:
            self.answer_box.delete(0, END)
            for info in info_list:
                self.answer_box.insert(END, info)

    def on_show(self):
        self.get_label_and_number()
        if self.number is not 0:
            self.index = self.labels_array.index(self.label)
            if self.number > 8:
                mask = np.random.choice(self.number, 8)
            else:
                mask = np.arange(0, self.number)

            self.display = []
            for i, index in enumerate(mask):
                if i < 4:
                    self.show_image(self.paths_array[self.index][index],
                                    size=(300, 360), x=300+(i-0)*300, y=20)
                else:
                    self.show_image(self.paths_array[self.index][index],
                                    size=(300, 360), x=300+(i-4)*300, y=400)
        pass

    def get_label_and_number(self):
        self.label = self.label_var.get()
        self.number = 0
        if self.labels_array is None:
            if self.language == 'chinese':
                mb.showerror('错误', '数据集为空！')
            else:
                mb.showerror('Error', 'Data set is empty!')
        elif self.label in self.labels_array:
            self.number = get_number_of_samples(self.label, paths_array=self.paths_array,
                                                labels_array=self.labels_array)
        else:
            if self.language == 'chinese':
                mb.showerror('错误', '数据集没有这个标签！')
            else:
                mb.showerror('Error', 'Data set does not have this label!')

    def on_search(self):
        self.get_label_and_number()
        if self.number is not 0:
            if self.language == 'chinese':
                mb.showinfo('结果', '在数据库中共找到{1}个标签为{0}的样本！'.format(
                    self.label, self.number))
            else:
                mb.showinfo('Answer', 'Found {1} samples with {0} label in the database!'.format(
                    self.label, self.number))
        pass


def search_dir(root_dir):
    paths_array, labels_array = [], []
    for (root, sub_dirs, files) in os.walk(root_dir):
        if len(sub_dirs) is 0:
            label = os.path.basename(root)
            sub_array = []
            for file in files:
                path = os.path.join(root, file)
                sub_array.append(path)

            paths_array.append(sub_array)
            labels_array.append(label)

    return paths_array, labels_array


def get_number_of_samples(label, paths_array, labels_array):
    index = labels_array.index(label)
    number = len(paths_array[index])
    return number


def main(language='chinese'):
    FaceRecognition(language=language)


if __name__ == '__main__':
    main()
