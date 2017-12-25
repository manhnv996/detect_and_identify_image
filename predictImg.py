import numpy as np
import os, sys
import cv2


# faceDetect = cv2.CascadeClassifier("cascade.xml")
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")


def writeFile(Xdata, Ylabel, filename):
    fileWrite = open(filename, 'w')
    for i in range(0, len(Ylabel)):
        fileWrite.write(str(Ylabel[i]) + ' ')
        for i2 in range(0, 1024):
            fileWrite.write(str(i2) + ':' + str(Xdata[i][i2] / 255.0) + ' ')

        fileWrite.write('\n')
    fileWrite.close()



def resizeFace(fileVector, img_file, scale_factor, min_neighbors):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    faces = faceDetect.detectMultiScale(img, scale_factor, min_neighbors)

    listFace = []
    listLabel = []

    num = ''
    for x, y, w, h in faces:
        if (faces.shape != (1, 4)):
            num = num + 'a'
        # if (h < 100 or w < 100):  # face detect maybe wrong --> not save!
        #     continue

        faceCrop = img[y:y + h, x:x + w]
        croppedImg = cv2.resize(
            faceCrop, (32, 32))  # resize to 32x32 px


        listFace.append(croppedImg)
        listLabel.append(0)

    XdataTemp = np.array(listFace)
    Xdata = XdataTemp.reshape(len(XdataTemp), 32 * 32)
    Ylabel = np.array(listLabel)

    print('   WRITING VECTOR...')
    writeFile(Xdata, Ylabel, fileVector)


def detectImg(fileOutput, img_file, scale_factor, min_neighbors):
    outlabel = np.genfromtxt(fileOutput, delimiter='\n').astype(int)
    if (open(fileOutput, 'r').read().count('\n') <= 1):
        outlabel = [outlabel]
    realLabel = ['adele', 'bruno_mars', 'jennifer_lopez', 'justin_bieber', 'lady_gaga', 'rihanna', 'shakira']

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray, scale_factor, min_neighbors)
    i = 0
    for (x, y, w, h) in faces:
        # if (h < 100 or w < 100):  # face detect maybe wrong --> not save!
        #     continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        putText(img, realLabel[outlabel[i]], (x, y-3), 0.5)

        print realLabel[outlabel[i]]
        i += 1

    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



font                   = cv2.FONT_ITALIC
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 1
def putText(img, text, bottomLeftCornerOfText, fontScale):
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def predictImg(img_file):
    resizeFace('TEMPP/testShowImg/vectorOutFrame', img_file, 1.3, 5)

    os.system('svm-predict TEMPP/testShowImg/vectorOutFrame model TEMPP/testShowImg/testout > TEMPP/testShowImg/log')

    detectImg('TEMPP/testShowImg/testout', img_file, 1.3, 5)




def showDetectImg(img_file, scale_factor, min_neighbors):

    img = cv2.imread(img_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showWebcam(scale_factor, min_neighbors):
    faceDetect = cv2.CascadeClassifier("cascade.xml")

    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceDetect.detectMultiScale(gray, scale_factor, min_neighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            break

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()



if __name__ == "__main__":

    # showWebcam(1.5, 3)
    # showDetectImg('test/img/img0.jpg', 1.1, 3)
    predictImg('test/img/img.jpg')
