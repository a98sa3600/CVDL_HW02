from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from UI import Ui_MainWindow
from keras.models import load_model
from keras.utils.image_utils import load_img
from keras.utils.image_utils import img_to_array
from keras.applications import ResNet50V2
import tensorflow_addons as tfa
from random import randint

IMAGE_SIZE = (224, 224) 

class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__() 
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        
    def setup_control(self):
        self.ui.load_image_button.clicked.connect(self.load_image)
        self.ui.show_image_button.clicked.connect(self.show_images)
        self.ui.show_distribution_button.clicked.connect(self.distribution)
        self.ui.show_model_structure_button.clicked.connect(self.model_structure)
        self.ui.show_comparision_button.clicked.connect(self.comparison)
        self.ui.inference_button.clicked.connect(self.inference)
        
    def load_image(self):
        self.img_path, _ = QFileDialog.getOpenFileName(self,filter='Image Files (*.png *.jpg *.jpeg *.bmp)')# start path

        self.img1 = cv2.imread(self.img_path)
        self.img1 = cv2.resize(self.img1, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        height, width, channel = self.img1.shape
        bytesPerline = 3 * width
        qimg = QImage(self.img1.tobytes(), width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        qpixmap = QPixmap.fromImage(qimg)
        qpixmap_height = qpixmap.height()
        qpixmap_height += 100
        scaled_pixmap = qpixmap.scaledToHeight(qpixmap_height)
        self.ui.label_image.setPixmap(scaled_pixmap)
        
    def get_picture_dir(self,file_dir):
        filelist = os.listdir(file_dir) #获取文件路径
        total_num=len(filelist)
        idx = randint(0,total_num)
        image_path=filelist[idx]
        picture_path = os.path.join(os.path.abspath(file_dir), image_path)
        filelist = ''
        return picture_path
        
        
    def show_images(self):
        self.class_list = ["Cat", "Dog"]
        self.folder_path = QFileDialog.getExistingDirectory(self,"Open folder","./")  
        folder_path_cat = self.folder_path + '/Cat/' 
        folder_path_dog = self.folder_path + '/Dog/'     

        cat_path = self.get_picture_dir(folder_path_cat)
        dog_path = self.get_picture_dir(folder_path_dog)
        print(cat_path)
        print(dog_path)
        cat1 = cv2.imread(cat_path)
        dog1 = cv2.imread(dog_path)
        ##
        
        title_list = []
        image_list = []
        fig = plt.figure(figsize=(1, 2))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        # cat1 = cv2.imread("./Dataset_OpenCvDl_Hw2_Q5/inference_dataset/Cat/8043.jpg")
        # dog1 = cv2.imread("./Dataset_OpenCvDl_Hw2_Q5/inference_dataset/Dog/12053.jpg")
        cat = cv2.resize(cat1, IMAGE_SIZE)
        dog = cv2.resize(dog1, IMAGE_SIZE)
        cat = cat[:,:,::-1]
        dog = dog[:,:,::-1]
        image_list.append(cat)
        image_list.append(dog)
        for i in range(1,3):
            title_list.append(self.class_list[i-1])
            plt.subplot(120 + i )
            plt.imshow(image_list[i-1])
            plt.title(title_list[i-1])
            plt.axis('off')
        plt.show()

    def distribution(self):
        # trcat = './Dataset_OpenCvDl_Hw2_Q5/training_dataset/Cat/'
        # trdog = './Dataset_OpenCvDl_Hw2_Q5/training_dataset/Dog/'
        # valcat = './Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Cat/'
        # valdog = './Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Dog/'
        # num = [len(os.listdir(trcat))+len(os.listdir(valcat)),len(os.listdir(trdog))+len(os.listdir(valdog))]
        # x_label = ["cat", "dog"]
        # plt.xticks( range(2),x_label)  
        # plt.bar(range(2),num, align = 'center',color= 'steelblue', alpha = 0.8)
        # plt.ylabel('number of images')
        # plt.title('Class Distribution')
        # for x,y in enumerate(num):plt.text(x,y,'%s'%y,ha='center')
        # plt.show()
        img_src = cv2.imread('distribution.png')
        cv2.imshow("result", img_src)
        cv2.waitKey(0)
        
    def model_structure(self):
        model = load_model('Q5_Model_loss2_third.h5')
        model.summary()
        
    
    def comparison(self):
        # model1 = load_model('Q5_Model_loss1_third.h5')
        # model2 = load_model('Q5_Model_loss2_third.h5') 
        # validation = tf.keras.utils.image_dataset_from_directory('./Dataset_OpenCvDl_Hw2_Q5/validation_dataset',image_size=IMAGE_SIZE, batch_size=128)
        # result1 = model1.evaluate(validation)
        # print(result1[1])
        # result2 = model2.evaluate(validation)
        # print(result2[1])
        # num = [   int(result1[1]*100),    int(result2[1]*100) ]
        # x_label = ["Focal Loss", "Binary Cross Entropy"]
        # plt.xticks( range(2),x_label)  
        # plt.bar(range(2),num, align = 'center',color= 'steelblue', alpha = 0.8)
        # plt.ylabel('Accuray(%)')
        # plt.title('Accuracy Comparison')
        # for x,y in enumerate(num):plt.text(x,y,'%s'%y,ha='center')
        # plt.savefig('Accuracy.png') 
        # plt.show()        
        img_src = cv2.imread('Accuracy.png')
        cv2.imshow("Accuracy", img_src)
        cv2.waitKey(0)
                
                 
    def inference(self): 
        img = load_img(self.img_path, target_size=(224, 224))
        if img is None:
            print('Please corrently input data') 
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        model = load_model('Q5_Model_loss2_third.h5')
        pred = model.predict(x)[0]
        
        if(pred<0.5):
            self.ui.label_predict.setText("Prediction:Cat")
            # print(pred,'Prediction:Cat')
        else:
            self.ui.label_predict.setText("Prediction:Dog")  
            # print(pred,'Prediction:Dog')   


                