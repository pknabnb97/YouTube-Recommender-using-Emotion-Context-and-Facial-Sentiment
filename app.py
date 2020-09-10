'''

This is a simple recommender that uses YouTube-Api to recommend videos.

The app first takes in Emotion and keeps it as context. If recommended video gets negative sentiment feedback, 
new category of video is recommended, keeping the Emotion it encountered first as context for the next recommendation.

YOUTUBE VIDEO_CATEGORY_ID LIST CAN BE FOUND ON THE PROJECT DIRECTORY.
    
#=========CONTEXT======================MAPPED-TO========================#
#=========EMOTION==================VIDEO_CATEGORY_ID====================#
           SAD                      23, 15, 10
           HAPPY                    24, 40, 28
           ANGRY/STRESSED           24, 39, 15
           SURPRISE/FEAR            1, 10, 39

SURPRISE and NEUTRAL are not good features for this first part of the recommender,
but SURPRISE is useful in SENTIMENT FEEDBACK.

FOR SENTIMENT FEEDBACK: *If user is SAD/HAPPY/SURPRISE mostly while watching the video, it means user is engaged with
                         the recommended media. So a POSITIVE SENTIMENT is taken.
                         
                        *ANGRY and DISGUST generally gives misclassification error in most FER tests,
                         as they look almost the same. In this recommendation, we can treat both anger and disgust
                         as NEGATIVE SENTIMENT as Anger is bad for health and very few people like Disgusting videos.

#===================KEEP WEB-BROWSER RESTORED DOWN TO OBSERVE GUI AND OPENCV WINDOW===================================#


'''


max_mood = ''
max_mood_2 = ''

    
def onClick(x):
    webbrowser.open(x,new=1)
    
def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values()) #no identation error..do not change
     k=list(d.keys())
     return k[v.index(max(v))]

def TakeMood():
    label_ids = {}
    current_id = 1
   
    camera = cv2.VideoCapture(0) # 0 means 1st Webcam
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    settings = {
        'scaleFactor': 1.3, 
        'minNeighbors': 5, 
        'minSize': (50, 50)
    }

    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_detection.detectMultiScale(gray, **settings)

        for x, y, w, h in detected:
            cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
            cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
            face = gray[y+5:y+h-5, x+20:x+w-20]
            face = cv2.resize(face, (48,48))
            face = face/255.0

            predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
            state = labels[predictions]
            if state != 'Neutral':
                if state in label_ids:
                    current_id += 1   # count of emotion increased. to be used as feature for recommender
                    label_ids[state] = current_id
                else:
                    label_ids[state] = current_id

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Show Mood. Press q to Quit", (20, 60) , font,1.2, (0,255,0), 3, cv2.LINE_AA)
            cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Facial Expression', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #writer.writeFrame(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    
    global max_mood  
    ''' GLOBAL VARIABLE CAN BE REPLACED EASILY WITH CLASS-BASED APPROACH, WHERE VALUE ENTERED IN BUTTON CAN
        BE KEPT IN "self.value"
    '''
    max_mood = keywithmaxval(label_ids)
    
    print(keywithmaxval(label_ids))
    print(max_mood)
    
    if max_mood == 'Angry':
        msgs.insert(END, "UI: Anger is bad for health, Dear User. Click on Play Recommended to play recommended media.")
        msgs.insert(END,"     REMEMBER TO QUIT THE OPENCV FRAME AFTER WATCHING VIDEO")
        msgs.insert(END,"     TO STOP GATHERING SENTIMENT.")
    
    elif max_mood == 'Sad':
        msgs.insert(END, "UI: AWW, you are SAD! Click on Play Recommended to play recommended media.")
        msgs.insert(END,"     REMEMBER TO QUIT THE OPENCV FRAME AFTER WATCHING VIDEO")
        msgs.insert(END,"     TO STOP GATHERING SENTIMENT.")
    
    elif max_mood == 'Happy':
        msgs.insert(END, "UI: You are a Happy User! Click on Play Recommended to play recommended media.")
        msgs.insert(END,"     REMEMBER TO QUIT THE OPENCV FRAME AFTER WATCHING VIDEO")
        msgs.insert(END,"     TO STOP GATHERING SENTIMENT.")
    elif max_mood == 'Surprise':
        msgs.insert(END, "UI: You are either Surprised or Very Scared, Dear User.")
        msgs.insert(END,"     Click on Play Recommended to play recommended media.")
        msgs.insert(END,"     You'll be pranked if you dont like our suggestion for the third time.")
        msgs.insert(END,"     REMEMBER TO QUIT THE OPENCV FRAME AFTER WATCHING VIDEO.")
        msgs.insert(END,"     TO STOP GATHERING SENTIMENT.")
        
    
        
    
        
    
         
def Recommend():
    
    
    global max_mood_2
    if not max_mood_2:
        if max_mood == 'Sad':
            url = "https://www.youtube.com/watch?v=" + response_23['items'][1]['id']
            onClick(url)
        elif max_mood == 'Happy':
            url = "https://www.youtube.com/watch?v=" + response_24['items'][1]['id']
            onClick(url)
        
        elif max_mood == 'Anger':
            url = "https://www.youtube.com/watch?v=" + response_24['items'][1]['id']
            onClick(url)
            
        elif max_mood == 'Surprise':
            url = "https://www.youtube.com/watch?v=" + response_1['items'][1]['id']
            onClick(url)
    
    elif max_mood_2 == 'Positive':
        global count_max_mood
        count_max_mood += 1  # will increase with every positive sentiment
        if max_mood == 'Happy':
            url = "https://www.youtube.com/watch?v=" + response_24['items'][count_max_mood]['id']
            onClick(url)
            
        elif max_mood == 'Sad':
            url = "https://www.youtube.com/watch?v=" + response_23['items'][count_max_mood]['id']
            onClick(url)
        
        elif max_mood == 'Anger':
            url = "https://www.youtube.com/watch?v=" + response_24['items'][count_max_mood]['id']
            onClick(url)
            
        elif max_mood == 'Surprise':
            url = "https://www.youtube.com/watch?v=" + response_1['items'][count_max_mood]['id']
            onClick(url)
    
    elif max_mood_2 == 'Negative':
        global count_max_mood_2
        count_max_mood_2 += 1  
        
        if count_max_mood_2 == 1:
            if max_mood == 'Happy':
                url = "https://www.youtube.com/watch?v=" + response_40['items'][count_max_mood]['id']
                onClick(url)
            
            elif max_mood == 'Sad':
                url = "https://www.youtube.com/watch?v=" + response_15['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Anger':
                url = "https://www.youtube.com/watch?v=" + response_39['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Surprise':
                url = "https://www.youtube.com/watch?v=" + response_10['items'][count_max_mood]['id']
                onClick(url)
                
        elif count_max_mood_2 == 2: # max available categories = 3, so maximum count_max_mood_2 will be 2
            
            if max_mood == 'Happy':
                url = "https://www.youtube.com/watch?v=" + response_28['items'][count_max_mood]['id']
                onClick(url)
            
            elif max_mood == 'Sad':
                url = "https://www.youtube.com/watch?v=" + response_10['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Anger':
                url = "https://www.youtube.com/watch?v=" + response_15['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Surprise':
                url = "https://www.youtube.com/watch?v=" + response_39['items'][count_max_mood]['id']
                onClick(url)
                
        else:
            
            if max_mood == 'Happy':
                url = "https://www.youtube.com/watch?v=" + response_40['items'][count_max_mood]['id']
                onClick(url)
            
            elif max_mood == 'Sad':
                url = "https://www.youtube.com/watch?v=" + response_15['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Anger':
                url = "https://www.youtube.com/watch?v=" + response_39['items'][count_max_mood]['id']
                onClick(url)

            elif max_mood == 'Surprise':
                url = "https://www.youtube.com/watch?v=" + response_10['items'][count_max_mood]['id']
                onClick(url)
            
            
#==============================================================================================#        
#============================COLLECTING FACIAL SENTIMENT=======================================#        

    
    label_ids = {}
    current_id = 1
   
    camera = cv2.VideoCapture(0) # 0 means 1st Webcam
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    settings = {
        'scaleFactor': 1.3, 
        'minNeighbors': 5, 
        'minSize': (50, 50)
    }

    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_detection.detectMultiScale(gray, **settings)

        for x, y, w, h in detected:
            cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
            cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
            face = gray[y+5:y+h-5, x+20:x+w-20]
            face = cv2.resize(face, (48,48))
            face = face/255.0

            predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
            state = labels[predictions]
            if state != 'Neutral':
                if state in label_ids:
                    current_id += 1   # count of emotion increased. to be used as feature for recommender
                    label_ids[state] = current_id
                else:
                    label_ids[state] = current_id

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Show Mood. Press q to Quit", (20, 60) , font,1.2, (0,255,0), 3, cv2.LINE_AA)
            cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Facial Expression', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #writer.writeFrame(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    
    #print(keywithmaxval(label_ids))
    
    #global max_mood_2
    if keywithmaxval(label_ids) == 'Angry':
        max_mood_2 = 'Negative'
        msgs.insert(END, "UI: You had a Negative Sentiment about the suggestion.")
        msgs.insert(END, "UI: So Recommending different category of videos.")
        msgs.insert(END, "UI: Click Play Recommended Button again. And remember to close opencv.")
        print(max_mood_2)
    elif keywithmaxval(label_ids) in ['Happy', 'Sad', 'Surprise']:
        
        max_mood_2 = 'Positive'
        msgs.insert(END, "UI: You had a Positive Sentiment about the suggestion.")
        msgs.insert(END, "UI: So Recommending similar category videos.")
        msgs.insert(END, "UI: Click Play Recommended Button again. And remember to close opencv.")



#=======================================================================================================#

import numpy as np
import cv2
import tensorflow as tf

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')

labels = ["Neutral","Happy","Sad","Surprise","Angry"]

model = tf.keras.models.load_model('expression.model')

#==========================================================================================#
#====================================GUI_FUNCTIONS=========================================#

import webbrowser  #built-in python
    
def Quit():
    main.destroy()  #destroy gui
    
def GoTo():
    onClick('https://console.developers.google.com/apis/dashboard')
    
def Instructions():
    messagebox.showinfo("Instructions to Create API-key", '''\
    1. Go to Google Developer Console with the link Above.
    2. On the Dashboard, Click on Create Project
    3. Give Project Name and Click on Create
    4. Click on go to API library
    5. Search for YouTube Data API v3 and Click on that
    6. Click on Enable and next Click on Create Credentials
    7. Choose YouTube Data api from the Drop-Down list of 'Which api are you using?'
    8. Choose Other Non-UI from the Drop-Down list of 'Where will be calling from?
    9. Select Public Data and Click on Which Credentials do I need.
    10. Copy the API-Key and Paste it on the Entry Field of our GUI''')

    
def BuildApi():
    from googleapiclient.discovery import build
    api_key = txt.get() #please enter your own api key with the executed app
    if api_key == 'Please put your own valid YouTube API-Key HERE and click ENTER':
        messagebox.showerror("ERROR","Please Enter Valid API-KEY")
        a = False
    else:
        messagebox.showinfo("SUCCESS","Key successfully entered!")
        a = True
    if a:
        print("IF ERROR OCCURS, THE ENTERED API KEY WAS NOT VALID OR NOT CONNECTED TO INTERNET!")
        youtube = build('youtube','v3',developerKey=api_key)
        #
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular",videoCategoryId = "20"  # chart="mostPopular" is bugged
            )
        
        global response_20
        response_20 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "15"
            )
        
        global response_15
        response_15 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "10"
            )
        global response_10
        response_10 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "29"
            )
        global response_23
        response_23 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "24"
            )
        global response_24
        response_24 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "20"   # mostPopular chart is bugged
            )
        global response_40
        response_40 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "28"
            )
        global response_28
        response_28 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "10"
            )
        global response_39
        response_39 = request.execute()
        
        request = youtube.videos().list(
            part = 'id', chart = "mostPopular", videoCategoryId = "1"
            )
        global response_1
        response_1 = request.execute()
        
        
        

#======================TKINTER GUI========================================================#
#=========================================================================================#

from tkinter import *
from tkinter import messagebox

main = Tk()
main.geometry("500x560")
main.title("SENTREC")

video = StringVar()
txt = Entry(main, textvariable = video, width = 60, borderwidth = 3) 
video.set('Please put your own valid YouTube API-Key HERE and click ENTER')
txt.pack()

btn = Button(main, text = "ENTER",command=BuildApi)
btn.pack()

btn = Button(main, text = "GO TO GOOGLE DEVELOPER CONSOLE",command=GoTo)
btn.pack()

btn = Button(main, text = "INTRUCTIONS TO CREATE API-Key",command=Instructions)
btn.pack()

frame = Frame(main)
sc = Scrollbar(frame)
msgs = Listbox(frame, width=80,height=20)
sc.pack(side=RIGHT, fill = Y)
msgs.pack(side=LEFT, fill = BOTH, pady = 10)
frame.pack()



msgs.insert(END, "UI: PLACE/CREATE API-Key:")
msgs.itemconfig(0, {'fg': 'blue'})
msgs.insert(END, "UI: Hello User, First Enter already created API-key in the Entry field above and Click ENTER.")
msgs.insert(END, "UI: MAKE SURE YOU HAVE INSTALLED google-api-python-client")
msgs.itemconfig(2, {'fg': 'red'})
msgs.insert(END, "UI: To Create new KEY click on GO TO GOOGLE DEVELOPER CONSOLE.")
msgs.insert(END, "UI: Click on INTRUCTIONS TO CREATE API-Key for instructions.")


msgs.insert(END, "UI: HOW TO USE APP:")
msgs.itemconfig(5, {'fg': 'blue'})
msgs.insert(END, "UI: CLICK ON SHOW YOUR MOOD TO SHOW YOUR CURRENT EMOTIONAL STATE...")
msgs.insert(END,"     AFTER SHOWING MOOD, CLOSE THE OPENCV FRAME USING 'q' KEY")

count_max_mood = 1 # must be given a range like len() to remove List Out Of Bounds error
count_max_mood_2 = 0


btn_show = Button(main, text = "SHOW YOUR MOOD (First-Time Use Only)", font = ('verdana',15), command = TakeMood) #IMP
btn_show.pack()

btn_rec = Button(main, text = "PLAY RECOMMENDED", font = ('verdana',15), command = Recommend)
btn_rec.pack()

btn_quit = Button(main, text = "QUIT", command = Quit)
btn_quit.pack()

main.mainloop()
