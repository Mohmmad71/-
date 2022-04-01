import os
import cv2
import numpy as np
import sys
from PIL import Image
 
 
detector = cv2.CascadeClassifier("D:\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
 
def getImagesAndLabels(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faceSamples=[]
	ids = []
	for imagePath in imagePaths:
		# İÊÍ ÇáÕæÑÉ
		PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
		 # ÊÍæíá ÇáÕæÑÉ Åáì ãÌãæÚÉ
		img_numpy = np.array(PIL_img,'uint8')
		 # ÇÍÕá Úáì ãÚÑİ ßá ÕæÑÉ
		#print(os.path.split(imagePath))
		id = int(os.path.split(imagePath)[-1].split(".")[0])
		faces = detector.detectMultiScale(img_numpy)
		for x,y,w,h in faces:
			faceSamples.append(img_numpy[y:y+h,x:x+w])
			ids.append(id)
	return faceSamples,ids
 
if __name__ == '__main__':
	 # ãÓÇÑ ÇáÕæÑÉ
	path='data/jm/'
	 # ÇÍÕá Úáì ãÕİæİÉ ÇáÕæÑ æãÌãæÚÉ ÚáÇãÇÊ ÇáåæíÉ
	faces, ids = getImagesAndLabels(path)
	 # ÇÍÕá Úáì ßÇÆä ÇáÊÏÑíÈ
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	 Recognizer.train (faces¡ np.array (ids)) # áÊÍæíá ÇáãÚÑİÇÊ ãä ŞÇÆãÉ Åáì ãÕİæİÉ
	# Save the model into trainer/trainer.yml
	recognizer.write('trainer/trainer.yml')