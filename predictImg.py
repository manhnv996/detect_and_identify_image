import numpy as np
import os, sys
import cv2


# from skimage import io
# from svm import *
# from svmutil import *
# from preProcess.py import faceDetectAndResizeImg, writeVector, writeFile


faceDetect = cv2.CascadeClassifier("cascade.xml")

# Detect face and resize image to (32x32) px.
def faceDetectAndResizeImg(inputFolder, outFaceFolder):

    print('\n   STARTING DETECT AND RESIZE FACE...')
    listDirs = os.listdir(inputFolder)
    for subDir in listDirs:
        path = inputFolder + subDir + '/'
        for imgfile in os.listdir(path):
            if os.path.isfile(path + imgfile):
                img = cv2.imread(path + imgfile)
                # Detect Faces
                faces = faceDetect.detectMultiScale(img)

                # IF CAN'T DETECT FACE --> CONTINUE!
                if (isinstance(faces, tuple)):
                    continue

                # print(path + imgfile)
                num = ''
                for x, y, w, h in faces:
                    if (faces.shape != (1, 4)):
                        num = num + 'a'
                    if (h < 100 or w < 100):  # face detect maybe wrong --> not save!
                        continue

                    faceCrop = img[y:y + h, x:x + w]
                    croppedImg = cv2.resize(
                        faceCrop, (32, 32))  # resize to 32x32 px
                    if not os.path.exists(outFaceFolder + subDir):
                        os.makedirs(outFaceFolder + subDir)
                    cv2.imwrite(outFaceFolder + subDir + '/' +
                                num + imgfile, croppedImg)

    print('---FACE DETECT COMPLETED!---\n')


def writeVector(faceFolder, fileVector):
    listFace = []
    listLabel = []

    listDirs = os.listdir(faceFolder)
    print('   READING IMAGE...')
    for subDir in listDirs:
        path = faceFolder + subDir + '/'
        for imgfile in os.listdir(path):

            # img = io.imread(path + imgfile, as_grey=True)
            img = cv2.imread(path + imgfile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # print(path + imgfile)
            label = listDirs.index(subDir)
            if (img.shape != (32, 32)):
                continue

            listFace.append(img)
            listLabel.append(label)
    # print('---READ IMAGE COMPLETED---\n')

    XdataTemp = np.array(listFace)
    Xdata = XdataTemp.reshape(len(XdataTemp), 32 * 32)
    Ylabel = np.array(listLabel)

    # print('dataShape: ', Xdata.shape)

    print('   WRITING VECTOR...')
    writeFile(Xdata, Ylabel, fileVector)

    print('---SAVE VECTOR COMPLETE!---\n')


def writeFile(Xdata, Ylabel, filename):
    fileWrite = open(filename, 'w')
    for i in range(0, len(Ylabel)):
        fileWrite.write(str(Ylabel[i]) + ' ')
        for i2 in range(0, 1024):
            fileWrite.write(str(i2) + ':' + str(Xdata[i][i2] / 255.0) + ' ')

        fileWrite.write('\n')
    fileWrite.close()


def showResult(fileOutput):
    outlabel = np.genfromtxt(fileOutput, delimiter='\n').astype(int)
    # print(outlabel)
    numlabel = [0, 0, 0, 0, 0, 0, 0]
    realLabel = ['adele', 'bruno_mars', 'jennifer_lopez', 'justin_bieber', 'lady_gaga', 'rihanna', 'shakira']
    for x in outlabel:
        numlabel[x] += 1
    maxIndex = max(numlabel)
    labelInt = numlabel.index(maxIndex)
    labelPredict = realLabel[labelInt]
    print('\nRESULT: ', labelPredict)


def predictVideo(fileVideo):
    # command = 'ffmpeg -i ' + fileVideo + ' -vf fps=1 TEMPP/outFrame/img/%d.jpg'
    # if not os.path.exists('TEMPP/outFrame/img'):
    #     os.makedirs('TEMPP/outFrame/img')
    # os.system(command)
    # #
    # faceDetectAndResizeImg('TEMPP/outFrame/', 'TEMPP/outFaceVideo/')
    # writeVector('TEMPP/outFaceVideo/', 'TEMPP/vectorOutFrame')
    # os.system('libsvm/svm-predict TEMPP/vectorOutFrame model testout > TEMPP/log')
    # os.system('svm-predict ../TEMPP/vectorOutFrame ../model ../testout > ../TEMPP/log')
    # m = svm_load_model('model')
    # y = svm_load_model('TEMPP/vectorOutFrame')
    # x = svm_load_model('testout')
    # svm_predict(y, m, x)
    showResult('testout')

    # os.system('rm -rf TEMPP')


# predictVideo('Boyfriend.mp4')
# predictVideo('hello.mp4')
# predictVideo('wakawaka.mp4')





def showImg(faceFolder):

    listDirs = os.listdir(faceFolder)
    print('   READING IMAGE...')
    for subDir in listDirs:
        path = faceFolder + subDir + '/'
        for imgfile in os.listdir(path):

            img = cv2.imread(path + imgfile)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceDetect.detectMultiScale(gray)
            for (x, y, w, h) in faces:
                if (h < 100 or w < 100):  # face detect maybe wrong --> not save!
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.imshow('img', img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print('---READ IMAGE COMPLETED---\n')




def resizeFace(fileVector, img_file, scale_factor, min_neighbors):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    faces = faceDetect.detectMultiScale(img, scale_factor, min_neighbors)

    listFace = []
    listLabel = []
    # print(path + imgfile)
    num = ''
    for x, y, w, h in faces:
        if (faces.shape != (1, 4)):
            num = num + 'a'
        # if (h < 100 or w < 100):  # face detect maybe wrong --> not save!
        #     continue

        faceCrop = img[y:y + h, x:x + w]
        croppedImg = cv2.resize(
            faceCrop, (32, 32))  # resize to 32x32 px

        # if not os.path.exists(outFaceFolder + subDir):
        #     os.makedirs(outFaceFolder + subDir)
        # cv2.imwrite(outFaceFolder + subDir + '/' +
        #             num + imgfile, croppedImg)


        listFace.append(croppedImg)
        listLabel.append(0)

    XdataTemp = np.array(listFace)
    Xdata = XdataTemp.reshape(len(XdataTemp), 32 * 32)
    Ylabel = np.array(listLabel)
    # print('dataShape: ', Xdata.shape)

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


def testImg(img_file):
    resizeFace('TEMPP/testShowImg/vectorOutFrame', img_file, 1.3, 5)
    # os.system('cd libsvm/windows/')
    os.system('svm-predict TEMPP/testShowImg/vectorOutFrame model TEMPP/testShowImg/testout > TEMPP/testShowImg/log')
    # os.system('svm-predict')
    # showResult('../../TEMPP/testImg/testout')

    # showResult('testout')

    detectImg('TEMPP/testShowImg/testout', img_file, 1.3, 5)
    # os.system('svm-predict vectorOutFrame ../model ../testout > ../TEMPP/log')


testImg('test/img/img.jpg')