import wx
import os
import cv2
import numpy as np
from PIL import Image

class MyFrame(wx.Frame):
    def OnEraseBAck(self, event):
        dc = event.GetDC()
        if not dc:
            dc = wx.ClientDC(self)
            rect = self.GetUpdateRegion().GetBox()
            dc.SetClippingRect(rect)

        dc.Clear()
        bmp = wx.Bitmap("Zhangwei.jpg")
        dc.DrawBitmap(bmp, 0, 0)

    def __init__(self,parent):
        wx.Frame.__init__(self,parent,title='人脸识别',size=(466,325))

        panel=wx.Panel(self)
        panel.Bind(wx.EVT_ERASE_BACKGROUND,self.OnEraseBAck)
        #创建按钮
        self.btn1=wx.Button(parent=panel,id=10,label='图像采集',pos=(70,50),size=(100,50))
        self.btn2=wx.Button(parent=panel,id=11,label='开始训练',pos=(70,100),size=(100,50))
        self.btn3=wx.Button(parent=panel,id=12,label='人脸识别',pos=(70,150),size=(100,50))
        self.btn4=wx.Button(parent=panel,id=13,label='人脸检测',pos=(70,200),size=(100,50))
        #创建捆绑事件
        self.btn1.Bind(wx.EVT_BUTTON,self.on_btn1,self.btn1)
        self.btn2.Bind(wx.EVT_BUTTON,self.on_btn2, self.btn2)
        self.btn3.Bind(wx.EVT_BUTTON,self.on_btn3, self.btn3)
        self.btn4.Bind(wx.EVT_BUTTON,self.on_btn4,self.btn4)



    #创建按钮点击事件
    def on_btn1(self,event):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        #输入id
        face_id = input('\n enter user id end press <return> ==>  ')

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")

        count = 0

        while (True):

            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # 保存图片
                cv2.imwrite("data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(200) & 0xff  # ESC退出and每0.2s拍一张
            if k == 27:
                break
            elif count >= 40:  # 拍40张照片
                break

        # cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    def on_btn2(self,event):
        path = 'data'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # 保存.yml训练集
        recognizer.write('trainer/trainer.yml')

        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def on_btn3(self,event):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX


        id = 0

        names = ['None', 'pengyuyan', 'xsy', 'rhy', 'Z', 'W']

        cam = cv2.VideoCapture(0)  #调用摄像头
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            img = cv2.flip(img, 1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # 判断成功概率>45时，输出id,否则输出unknow
                if (confidence < 55):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    def on_btn4(self,event):
        faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # set Weight
        cap.set(4, 480)  # set Height

        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,

                scaleFactor=1.2,
                minNeighbors=5
                ,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

            cv2.imshow('video', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # Esc for quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    app=wx.App()
    MyFrame(None).Show()
    app.MainLoop()
