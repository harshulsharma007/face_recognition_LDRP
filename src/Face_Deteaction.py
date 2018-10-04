import cv2

error_img = cv2.imread('data/test2.jpg')

haar_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

def detectFaces(face_cascade, color_image, scaleFactor = 1.1):

    img_copy = color_image.copy()

    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
        #return img_copy  [y:y+h, x:x+w]

    return img_copy

test1_img = cv2.imread('data/s2/1.jpg')
#test_img = cv2.resize(test1_img,None,fx=1/2,fy=1/2,interpolation=cv2.INTER_AREA)


result_image = detectFaces(haar_face_cascade, test1_img)

cv2.imshow('grayScale_IMG', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
