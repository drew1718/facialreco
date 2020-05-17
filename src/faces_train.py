import os
from PIL import Image
import numpy as np 
import pickle
import cv2
import time
import shutil


# this function does the following:
# 1- it asks for new user's name
# 2- captures 150 images and stores them in the database folder 
# 3- train the model so that it can recognize it in the test
def store_images():
    # ask for the user's name
    name = input("Please, Enter the new user name: ")
    
    # face detector
    face_cascades = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')
    
    # capture camera images
    cap = cv2.VideoCapture(0)
    
    # simple counter to count images
    i = 0
    
    # check if the user name is already there if so delete it and 
    # record new photos of him
    if(os.path.isdir("database/{}".format(name))):
        shutil.rmtree("database/{}".format(name))
        
    # create a directory
    os.mkdir("database/{}".format(name))
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascades.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
            
            # same as in main.py
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h,x:x+w]

                img_item = "database/{}/{}.png".format(name,i) 
                cv2.imwrite(img_item, roi_gray)
                i += 1
                color = (255,0,0)
                # how thick the rectangle side is
                stroke = 2
                end_cord_x = x+w
                end_cord_y = y+h
                cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y),color, stroke)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            if i == 400:
                break

    cap.release()
    cv2.destroyAllWindows()
    train()



def train():
    # here we go through all the stored images and we feed them to
    # the recognizer
    # this is the base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # here we store the image directory
    image_dir = os.path.join(BASE_DIR, "database/")


    # create a recognizer object
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # have the labels list and train list to be filled
    y_labels = []
    x_train = []

    # the current id is an id we give to each user
    current_id = 0
    label_ids = {}

    # loop over the image dir 
    for root, dirs, files in os.walk(image_dir):
        # loop over the files (images) that ends with png and jpg
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                
                # replace each space with a '-' so no error happens 
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                
                # here we check if we have a new one 
                if not(label in label_ids):
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                    
                # read the image as a gray image
                pil_image = Image.open(path).convert("L")
                
                # put it in a numpy array as unsigned 8-bit
                # because image pixels have only values from 0 to 255
                image_arr = np.array(pil_image, "uint8")


                # append the image_array and the id
                x_train.append(image_arr)
                y_labels.append(id_)


    # save the labels list to use at testing
    with open("labels.pkl", "wb") as f:
        pickle.dump(label_ids, f)

    try:
        # train the recognizer
        recognizer.train(x_train, np.array(y_labels))
        
        # save what it has learned in a yml file and load it when testing.
        recognizer.save("trainer.yml")
        print("Done!!")
    except:
        os.remove("trainer.yml")
        print("No data found!!")