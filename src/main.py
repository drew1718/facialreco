# this opencv librayr
import cv2

#getting the file face_train to use its function
from faces_train import *

# pickle is used to store and load data from the system in here we load labels
import pickle

# this is used to erase directories of the database in case we tried to 
# delete some one from the system
import shutil


# this is a classifier that detects front facing faces
face_cascades = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')
# face_cascades_2 = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')


# this is a recognizer model from opencv that classifies which face in the database
# recognizer = cv2.face.createLBPHFaceRecognizer.create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# here we are telling the recognizer to load what it learned
# from the training so we don't have to train the model each time
# just load a pretrained model


# this is a video capture to read images from the camera
cap = cv2.VideoCapture(0)

labels = {}

while(True):
    # here we present the choices to the user 

    choice = input("1- Register a new person.\n2- Delete a person.\n3- Test.\n4- Exit.\nyour choice: ")
    choice = int(choice)
    
    # if one then the user must give the name of the new user
    # and the store_images from the faces_train images will be called
    # more details there
    if(choice == 1):
        cap.release()
        store_images()
        break
        
    
    # 2 is where a user is deleted from the database and the name of that
    # user should be input and then the directory will be deleted 
    elif(choice == 2):
        name = input("please, input the name:")
        if(os.path.isdir("database/{}".format(name))):
            shutil.rmtree("database/{}".format(name))
            train()
        else:
            print("not found!\n")
        
        break

    # to test
    # the labels of all the users will be loaded from the pkl file
    elif(choice == 3):
        try:
            recognizer.read("trainer.yml")
        except: 
            print("no one to detect! please register a person first")
            break
        i = 0
        with open("labels.pkl", 'rb') as f:
            og_labels = pickle.load(f)
            print(og_labels.items())
            labels = {v:k for k,v in og_labels.items()}


        # and the 
        while(True):
            # here frame is read from the camera
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # then find the faces in the current frame using the face detector 
            # we previously defined in line 16
            faces = face_cascades.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
            # faces_2 = face_cascades_2.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)

            

            # now we loop over each face we found in the current frame
            for (x, y, w, h) in faces:
                
                # we cut the face  from the image
                gray_cut = gray[y:y+h, x:x+w]
                
                # pass the cut part to the recognizer to classify it.
                # id_ is the name of user detected
                # conf is the confidence and how we sure the model is 
                # from this prediction
                id_, conf = recognizer.predict(gray_cut)

                # from experience 45 is a good threshold
                if conf>=70:
                    print(id_)
                    print(labels[id_])

                    #from line 92 to 105 we are just putting a text on detected face 
                    # with the recongnized user's name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (x,y)
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2

                    cv2.putText(frame, labels[id_], 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)


            #     # now we loop over each face we found in the current frame
            # for (x, y, w, h) in faces_2:
                
            #     # we cut the face  from the image
            #     gray_cut = gray[y:y+h, x:x+w]
                
            #     # pass the cut part to the recognizer to classify it.
            #     # id_ is the name of user detected
            #     # conf is the confidence and how we sure the model is 
            #     # from this prediction
            #     id_, conf = recognizer.predict(gray_cut)

            #     # from experience 45 is a good threshold
            #     if conf>=55:
            #         print(id_)
            #         print(labels[id_])

            #         #from line 92 to 105 we are just putting a text on detected face 
            #         # with the recongnized user's name
            #         font = cv2.FONT_HERSHEY_SIMPLEX
            #         bottomLeftCornerOfText = (x,y)
            #         fontScale              = 1
            #         fontColor              = (255,255,255)
            #         lineType               = 2

            #         cv2.putText(frame, labels[id_], 
            #             bottomLeftCornerOfText, 
            #             font, 
            #             fontScale,
            #             fontColor,
            #             lineType)

                
                # img_item = "database/Moustafa/moustafa-{}.png".format(i) 
                # cv2.imwrite(img_item, gray_cut)
                # i += 1

                # in the next couple of lines we are just putting a rectangle around
                # the detected face
                color = (255,0,0)
                # how thick the rectangle side is
                stroke = 2
                end_cord_x = x+w
                end_cord_y = y+h
                cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y),color, stroke)

            # here we show a real time
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    elif(choice == 4):
        break