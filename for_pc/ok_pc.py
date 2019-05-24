#importing all files
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import imutils
import dlib
import math
import cv2
import time
########################################################
class stack:# round-robin to save old month statuses

    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
    def push(self,obj):
    ###save a transiton###
        if len(self.memory)<self.capacity:
            self.memory.append(1)
        tmp=self.position
        self.memory[self.position]=obj
        self.position=(self.position+1)%self.capacity
        return tmp
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def ready(self,batch_size):
        return len(self.memory)>batch_size
    def _len_(self):
        return len(self.memory)
    def min(self,leng,con):
        a=np.array(self.memory)
        for i in range(self.capacity-1):
            if(abs(i-con)<leng or abs(con-i+self.capacity)<leng):
                a[i]=10000
        print(a)
        return a.min()
#####################################################
thresh_0=0.4
thresh_2=0.72

thresh = 0.25
frame_check = 5
mStart=48
mEnd=60
xStart=60
xEnd=68

drowsy=0
priva=1
memtime=30
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#priva = input("Please input your sleepy intend: (1-5,and 5 means the most) ")
thresh -= priva/5*0.05
flag=0
flag_1=0
flag_2=0
ALPHA = 1.8
BETA = 60

# g1=stack(memtime)
# for i in range(memtime):
#     g1.push(10000)
global flag
flag1=0
t1=0
global t1
con_light=0


########################################################
#facial landmarks key points 
faciallandmarks =[
    ("jaw_keypoints", (0, 17)),
    ("righteyebrow_keypoints", (17, 22)),
    ("lefteyebrow_keypoints", (22, 27)),
    ("nose_keypoints", (27, 36)),
    ("righteye_keypoints", (36, 42)),
    ("lefteye_keypoints", (42, 48)),
    ("mouth_keypoints", (48, 68))
    
]


# def light():
#     bus = smbus.SMBus(1)
#     addr = 0x23
#     data = bus.read_i2c_block_data(addr,0x11)
#     light=(data[1]+256*data[0])/1.2
#     return light
import mp3play   
def playmusic(path):
    clip = mp3play.load(path)
    clip.play()
    time.sleep(3)
    clip.stop()

def playmusic1(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#rectangle points
def rectbb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

#numpy array conversion
def shapenp(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
#yawn detection
def yawn_detection1(shape,dtype="int"):
    global t1,flag
    t1+=1
    top_lips=[]
    bottom_lips=[]
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if 50<=i<=53 or 61<=i<=64:
            top_lips.append(coords[i])
        
        elif 65<=i<=68 or 56<=i<=59:
            bottom_lips.append(coords[i])
    mouth_wid=coords[55]-coords[49]
    toplipsall=np.squeeze(np.asarray(top_lips))
    bottomlipsall=np.squeeze(np.asarray(bottom_lips))
    top_lips_mean=np.array(np.mean(toplipsall,axis=0),dtype=dtype)
    bottom_lips_mean=np.array(np.mean(bottomlipsall,axis=0),dtype=dtype)
    top_lips_mean = top_lips_mean.reshape(-1) 
    bottom_lips_mean=bottom_lips_mean.reshape(-1) 
    
    #distance=math.sqrt((bottom_lips_mean[0] - top_lips_mean[0])**2 + (bottom_lips_mean[-1] - top_lips_mean[-1])**2)
    distance1=bottom_lips_mean[-1] - top_lips_mean[-1]
    radio=distance1*1.0/mouth_wid[0]
    print(bottom_lips_mean[-1],top_lips_mean[-1])
    yawn=False
    
    if(radio>thresh_2):
        print("yawn ratio",radio)
        flag+=1
    if(t1>memtime):
        flag=0
        t1=0
    if(flag>10):
        yawn=True
        flag=0
    
    return yawn
def yawn_detection(shape,dtype="int"):
    yawn=False
    global t1,distance_pre_pre_pre,distance_pre_pre,distance_pre,flag
    t1+=1
    top_lips=[]
    bottom_lips=[]
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if 50<=i<=53 or 61<=i<=64:
            top_lips.append(coords[i])
        
        elif 65<=i<=68 or 56<=i<=59:
            bottom_lips.append(coords[i])
    mouth_wid=coords[55]-coords[49]
    print("wide",mouth_wid)
    toplipsall=np.squeeze(np.asarray(top_lips))
    bottomlipsall=np.squeeze(np.asarray(bottom_lips))
    top_lips_mean=np.array(np.mean(toplipsall,axis=0),dtype=dtype)
    bottom_lips_mean=np.array(np.mean(bottomlipsall,axis=0),dtype=dtype)
    top_lips_mean = top_lips_mean.reshape(-1) 
    bottom_lips_mean=bottom_lips_mean.reshape(-1) 
        
    distance1=bottom_lips_mean[-1] - top_lips_mean[-1]
    # print("distance",distance)
    radio=distance1*1.0/mouth_wid[0]
    print("radio",radio)
    tmp=g1.push(radio)
    if(radio>thresh_2):
        print("yawn ratio",radio)
        if(g1.min(10,tmp)<thresh_0):
            flag+=1
            # print("flag",flag)
    if(t1>memtime):
        flag=0
        t1=0
    if(flag>10):
        yawn=True
        flag=0
    return yawn
def yawn_init(shape,dtype="int"):
    top_lips=[]
    bottom_lips=[]
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if 50<=i<=53 or 61<=i<=64:
            top_lips.append(coords[i])
        
        elif 65<=i<=68 or 56<=i<=59:
            bottom_lips.append(coords[i])
        toplipsall=np.squeeze(np.asarray(top_lips))
        bottomlipsall=np.squeeze(np.asarray(bottom_lips))
        top_lips_mean=np.array(np.mean(toplipsall,axis=0),dtype=dtype)
        bottom_lips_mean=np.array(np.mean(bottomlipsall,axis=0),dtype=dtype)
        top_lips_mean = top_lips_mean.reshape(-1) 
        bottom_lips_mean=bottom_lips_mean.reshape(-1) 
        
        #distance=math.sqrt((bottom_lips_mean[0] - top_lips_mean[0])**2 + (bottom_lips_mean[-1] - top_lips_mean[-1])**2)
        distance=bottom_lips_mean[-1] - top_lips_mean[-1]
        print(bottom_lips_mean[-1],top_lips_mean[-1])
        #distance=math.sqrt((bottom_lips_mean[0] - top_lips_mean[0])**2 + (bottom_lips_mean[-1] - top_lips_mean[-1])**2)
        distance=bottom_lips_mean[-1] - top_lips_mean[-1]
        return distance
#pout and smile detection
def pout_detection(shape,dtype="int"):
    left_corner=[]
    right_corner=[]
    coords = np.zeros((68, 2), dtype=dtype)
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if i==65 or i==55:
            right_corner.append(coords[i])
        
        elif i==60 or i==48:
            left_corner.append(coords[i])
        leftcornerall=np.squeeze(np.asarray(left_corner))
        rightcornerall=np.squeeze(np.asarray(right_corner))
        leftmean=np.array(np.mean(leftcornerall,axis=0),dtype=dtype)
        rightmean=np.array(np.mean(rightcornerall,axis=0),dtype=dtype)
        leftmean_flat = leftmean.reshape(-1) 
        rightmean_flat=rightmean.reshape(-1)
        distance=abs(leftmean_flat[0]-rightmean_flat[0])  
        pout=False
        smile=False
        
        if distance<35:
            pout=True
        
        elif distance>41:
            smile=True
    
    return (pout,smile)
def warn(x,y,w,h):
    if(x<50):
        print("move left")
        #playmusic("left.mp3")
        
    if(y<50):
        print("move down")
        #playmusic("down.mp3")
    if(x+w>250):
        print("move right")
        #playmusic("right.mp3")
        
    if(y+h>200):
        print("move up")
        #fplaymusic("up.mp3")



#front face dectetor
detect = dlib.get_frontal_face_detector()
#key points predictor
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#video capturing
cam=cv2.VideoCapture(0)
time.sleep(2.0)

# playmusic("traffic.mp3")

#frame capturing
while True:
    flag_month=False
    flag_eye=False
    # con_light=light()
    # if(con_light<100):
    #     playmusic("lightdown.mp3")
    # elif(con_light>5000):
    #     playmusic("lighttoomuch.mp3")

    a,image = cam.read()
    image = cv2.resize(image,(300,250))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 1)
    shape=[]
    for (index, rect) in enumerate(rects):
    	print(1)
        shape = predict(gray, rect)
        shape_new = shapenp(shape)
        (x, y, w, h) =rectbb(rect)
        print("x{},y{},w{},h{}".format(x,y,w,h))
        warn(x,y,w,h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face No.{}".format(index + 1), (x - 8, y - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        count=0
        
        for (x, y) in shape_new:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(image,"{}".format(count + 1), (x - 1, y - 1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),1)
            count+=1
    print("shape",type(shape))
    if(isinstance(shape, list)):
        # playmusic("move.mp3")
        continue
    # value=yawn_detection1(shape)
    # print("yawn",value)
    # if value==True:
    #     flag_month=True
    #     cv2.putText(image,"Yawning",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),2)
    #     cv2.putText(image, "****************month!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    leftEye = shape_new[lStart:lEnd] 
    rightEye = shape_new[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    print("ear",ear)
    # leftEyeHull = cv2.convexHull(leftEye)
    # rightEyeHull = cv2.convexHull(rightEye)
    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    if ear < thresh:
        flag1 += 1
        if flag1 >= frame_check:
            flag1=0
            cv2.putText(image, "****************Eye!****************", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "****************Eye!****************", (10,325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #print ("Drowsy")
    # if flag_month==True and flag_eye==True:
    #     playmusic("pilao3.mp3")
    # elif flag_month==True:
    #     playmusic("pilao2.mp3")
    # elif flag_eye==True:
    #     playmusic("pilao1.mp3")
    cv2.imshow("result", image)
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        break

#webcam release
cam.release()
#all windows destroying
cv2.destroyAllWindows()
