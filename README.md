# face_recognition_LDRP

----------Local Directional Relation Pattern--------------

The program is an implementation of the Local Directional Relation Pattern algorithm for facial recognition.
It uses OpenCV's built in LBP and haar cascade classifier for detecting faces from an image. The cascade classifier return a bounding box which contains the face in the input image.
The bounding box(face image) is fed to the LDRP feature vector generating algorithm which creates a feature vector using the input face image.
The feature vector is then sent to be stored in a database, which was a .csv file in my case. Thus, following this procedure, all the images in the training set are converted into a feature vectors and are stored in the database.

Now for any new input image, a feature vector is generated. The generated feature vector is now compared with the ones stored in the database using KNN.
The closest k feature vectors are selected and then are voted upon, the subject which gets the maximum number of votes is declared to be the result.
