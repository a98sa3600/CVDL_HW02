from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() 
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.number = 1
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        self.corners_vertical = 8
        self.corners_horizontal = 11
        self. pattern_size = (self.corners_horizontal, self.corners_vertical)
        self.world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        self.world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        self.world_points = []
        self.img_points = []
        self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1,3)
                
    def setup_control(self):
        self.ui.load_folder_button.clicked.connect(self.load_folder)
        self.ui.load_image_l_button.clicked.connect(self.load_image_L)
        self.ui.load_image_r_button.clicked.connect(self.load_image_R)
        
        self.ui.draw_contour_button.clicked.connect(self.draw_counter)
        self.ui.count_rings_button.clicked.connect(self.count_rings)
        
        self.ui.number_button.currentIndexChanged.connect(self.numbers)
        self.ui.find_corners_button.clicked.connect(self.find_corners)
        self.ui.find_intrinsic_button.clicked.connect(self.find_intrinsic)
        self.ui.find_extrinsic_button.clicked.connect(self.find_extrinsic)
        self.ui.find_distortion_button.clicked.connect(self.find_distortion)
        self.ui.show_result_button.clicked.connect(self.show_result)
        
        self.ui.show_onboard_button.clicked.connect(self.words_on_board)
        self.ui.show_vertically_button.clicked.connect(self.words_vertically)
        self.ui.stereo_disparity_map_button.clicked.connect(self.stereo_disparity_map)


    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder","./")                 
        foldername= os.path.basename(self.folder_path) 
        self.ui.folder_label.setText(foldername)
        print(self.folder_path)

    def load_image_L(self):
        filepathL,_= QFileDialog.getOpenFileName(self, "Open file", " ") 
        if len(filepathL) == 0: 
            print("\n Please input correct picture")
        self.img_L = cv2.imread('{}'.format(filepathL))
        filenameL= os.path.basename(filepathL) 
        self.ui.label_imageL.setText(filenameL)
        self.filepathL= filepathL

        
    def load_image_R(self):
        filepathR,_= QFileDialog.getOpenFileName(self, "Open file", " ") 
        if len(filepathR) == 0: 
            print("\n Please input correct picture")
        self.img_R = cv2.imread('{}'.format(filepathR))
        filenameR= os.path.basename(filepathR) 
        self.ui.label_imageR.setText(filenameR)
        self.filepathR = filepathR
        
        
    def draw_counter(self):
        # gray-> binary -> gaussian -> edge detection -> draw     
        img1 = self.img_L
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        #ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        gaussian1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        edge1 = cv2.Canny(gaussian1,30,200)
        contours1, hierarchy1 = cv2.findContours(edge1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img1,contours1,-1,(0,0,255),3)
        cv2.imshow("img_L", img1)  
        contour_nmm1 = hierarchy1.shape[1]
        self.rings1 = int(contour_nmm1/4) 
                

        img2 = self.img_R
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        #ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        gaussian2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        edge2 = cv2.Canny(gaussian2,30,200)
        contours2, hierarchy2 = cv2.findContours(edge2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img2,contours2,-1,(0,0,255),3)
        cv2.imshow("img_R", img2)  
        contour_nmm2 = hierarchy2.shape[1]
        self.rings2 = int(contour_nmm2/4)    

        cv2.waitKey(0)
    
    def count_rings(self):  
        text = "img1: {} rings \n img2: {} rings \n".format(self.rings1,self.rings2)
        self.ui.label_rings.setText(text)
        
        
    def find_corners(self):
        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:
                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(img_src, self.pattern_size, corners2, ret)
                cv2.namedWindow("img", 0)
                cv2.resizeWindow("img", 1075, 900)
                cv2.imshow("img", img_src)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def find_intrinsic(self):
        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    self.world_points.append(self.world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if  [corners2]:
                    self.img_points.append(corners2)
                else:
                    self.img_points.append(corners)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.world_points, self.img_points , size,None,None)
        print("instrinsic:\n", self.mtx)

    def numbers(self):
        self.number = self.ui.number_button.currentText()
        print(self.number)

    def find_extrinsic(self):
        file_path = ('{}/{}.bmp'.format(self.folder_path,self.number) )
        img_src = cv2.imread(file_path)

        if img_src is not None:
            gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                _, self.rvec, self.tvec, inliers = cv2.solvePnPRansac(self.world_point, corners2, self.mtx, self.dist)

                rotation_m, _ = cv2.Rodrigues(self.rvec)
                rotation_t = np.hstack([rotation_m, self.tvec])
                print("extrinsic:\n",rotation_t)

    def find_distortion(self):
        print("distortion:\n", self.dist)

    def show_result(self):
        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, size, 0, size)

                dst = cv2.undistort(img_src, self.mtx, self.dist, None, newcameramtx)

            imgs = np.hstack([img_src,dst])
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("img", 1920, 1080)
            cv2.imshow("img", imgs)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def words_on_board(self):
        self.words = self.ui.words_input.toPlainText()
        world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        world_points = []
        img_points = []

        position = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
        for i in range(1,6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    world_points.append(world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, size, None, None)
        for i in range(1, 6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:
                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, corners2, mtx,dist)

                    fs = cv2.FileStorage('{}/Q2_lib/alphabet_lib_onboard.txt'.format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(self.words)):
                        if self.words[j] is not None:
                            w = []
                            wi = []
                            ch = fs.getNode('{}'.format(self.words[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+position[j])
                                wi.append(ch[k][1]+position[j])
                            word = np.float32(w).reshape(-1,3)
                            imgpts, jac = cv2.projectPoints(word, rvec, tvec, mtx, dist)
                            word_i = np.float32(wi).reshape(-1,3)
                            imgpts_i =cv2.projectPoints(word_i, rvec, tvec, mtx, dist)
                            for a in range(len(ch)):
                                corners2[0][0][0] = imgpts_i[0][a][0][0]
                                corners2[0][0][1] = imgpts_i[0][a][0][1]
                                corner = tuple( (corners2[0].astype(int)).ravel())                        
                                end = tuple(  ((imgpts[a]).astype(int)).ravel())                                                           
                                cv2.line(img_src, corner,end, (0, 0, 255), 5)
 
                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", img_src)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def words_vertically(self):
        self.words = self.ui.words_input.toPlainText()
        world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        world_points = []
        img_points = []

        position = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
        for i in range(1,6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
                
                if ret:
                    world_points.append(world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, size, None, None)
        for i in range(1, 6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, corners2, mtx,dist)

                    fs = cv2.FileStorage('{}/Q2_lib/alphabet_lib_vertical.txt'.format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(self.words)):
                        if self.words[j] is not None:
                            w = []
                            wi = []
                            ch = fs.getNode('{}'.format(self.words[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+position[j])
                                wi.append(ch[k][1]+position[j])


                            word = np.float32(w).reshape(-1,3)
                            imgpts, jac = cv2.projectPoints(word, rvec, tvec, mtx, dist)
                            word_i = np.float32(wi).reshape(-1,3)
                            imgpts_i =cv2.projectPoints(word_i, rvec, tvec, mtx, dist)
                            for a in range(len(ch)):
                                corners2[0][0][0] = imgpts_i[0][a][0][0]
                                corners2[0][0][1] = imgpts_i[0][a][0][1]
                                corner = tuple(corners2[0].ravel())
                                corner = tuple( (corners2[0].astype(int)).ravel())                        
                                end = tuple(  ((imgpts[a]).astype(int)).ravel())                                                           
                                cv2.line(img_src, corner,end, (0, 0, 255), 5)

                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", img_src)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def stereo_disparity_map(self):
        
        img_L = cv2.imread('{}'.format(self.filepathL), 0)
        img_R = cv2.imread('{}'.format(self.filepathR), 0)
        stereo = cv2.StereoBM_create(numDisparities=16*16, blockSize=25)
        disparity = stereo.compute(img_L,img_R)
        disparity = (disparity - np.min(disparity)) * (255 / (np.max(disparity) - np.min(disparity)))
        
        img_L = cv2.imread('{}'.format(self.filepathL))
        img_R = cv2.imread('{}'.format(self.filepathR))
        height = disparity.shape[0]
        width = disparity.shape[1]
        img_L = cv2.resize(img_L, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)
        img_R = cv2.resize(img_R, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)
        disparity = cv2.resize(disparity, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)

        baseline=343
        focal_length=4019
        Cx = 279

        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                if disparity[y][x] > 0 :
                    dist = disparity[y][x] - Cx
                    depth = int(focal_length * baseline / abs(dist))
                    cv2.circle(img_R, ( x - int( disparity[y][x] / 2 )  , y ), 5 , (0, 0, 255), -1)
                    cv2.imshow('imgr', img_R)
                    
        while (True):
            cv2.namedWindow('imgl')              
            cv2.namedWindow('imgr')
            cv2.setMouseCallback('imgl', draw_circle, None)
            cv2.imshow('imgl', img_L)
            cv2.imshow('imgr', img_R)
            plt.imshow(disparity, 'gray')
            plt.show()
            if cv2.waitKey(20)  :
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        