import cv2
import sys
import imutils
import time
import numpy as np

#neighborhood area scale parameter
nsp = 2.0

start_time = time.time()
upper_img = cv2.imread("pic/pig.png", -1)
# Get user supplied values
#imagePath = sys.argv[1]
imagePath = "pic/campus.png"
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#load cascades
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")


def templateMatching(frame,previous_face):
    # ref: http://docs.opencv.org/trunk/d4/dc6/tutorial_py_template_matching.html
    faces = []
    res = cv2.matchTemplate(frame,previous_face,cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        faces.append([pt[0],pt[1],previous_face.shape[1],previous_face.shape[0]])
        #only append 1 face
        break

    return faces

def detectFace(frame,face_cascade,previous_frame,previous_faces):
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(5, 5),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if (len(faces) == 0):
        faces = []
        for (x,y,w,h) in previous_faces:
            #border of neighborhood area: frame[y1:y2,x1:x2]
            x1 = max(0, int(x+(1-nsp)*w/2))
            x2 = min(frame.shape[1]-1, int(x+(1+nsp)*w/2))
            y1 = max(0, int(y+(1-nsp)*h/2))
            y2 = min(frame.shape[0]-1, int(y+(1+nsp)*h/2))
            
            temp_faces = templateMatching(frame[y1:y2,x1:x2],previous_frame[y:y+h,x:x+w])
            
            for (tx,ty,tw,th) in temp_faces:
                faces.append([tx+x1,ty+y1,tw,th])
    return faces
    # algorithm ref: https://github.com/hrastnik/face_detect_n_track
    # if face found in previous frame
    '''
    if type(previous_faces) != type(0) and len(previous_faces):
        faces = []
        for (x,y,w,h) in previous_faces:
            #border of neighborhood area: frame[y1:y2,x1:x2]
            x1 = max(0, int(x+(1-nsp)*w/2))
            x2 = min(frame.shape[1]-1, int(x+(1+nsp)*w/2))
            y1 = max(0, int(y+(1-nsp)*h/2))
            y2 = min(frame.shape[0]-1, int(y+(1+nsp)*h/2))
            
            temp_faces = (face_cascade.detectMultiScale(
                frame[y1:y2,x1:x2],
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(5, 5),
                flags = cv2.CASCADE_SCALE_IMAGE
            ))
            if (len(temp_faces)):
                for (tx,ty,tw,th) in temp_faces:
                    faces.append([tx+x1,ty+y1,tw,th])
                return faces
            else:
                return templateMatching(frame,previous_frame,previous_faces)
    else:
        #search whole frame
        return face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(5, 5),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    '''    

def drawHead(frame, faces):
    '''
    uw: upper width, uh: upper height
    x,y,w,h: horizontal to right, vertical to bottom
    frame shape: (450, 800, 3) <= (height, width, dimensions)
    '''
    #height and width of frame
    fh = frame.shape[0]
    fw = frame.shape[1]

    for (x, y, w, h) in faces:
        
        #draw rectangle
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #center of face
        cy = y+h/2
        cx = x+w/2
        #make upper bigger to cover head
        scale_upper = 2 
        upper2 = imutils.resize(upper_img, width=int(min(scale_upper*h, scale_upper*upper_img.shape[1])))
        #upper2 = imutils.resize(upper1, width=int(min(w, upper1.shape[0])))
        uh = upper2.shape[0]
        uw = upper2.shape[1]
        '''
        print('upper2.shape: ',upper2.shape,' uh ',uh,' uw ',uw)
        print('x,y,w,h: ',(x,y,w,h))
        print('frame resized shape: ',frame_resized.shape)
        '''

        # deal with alpha
        #frame[y:y+uh,x:x+uw] = upper2

        # in case face move out of border
        # use cy-uh/2+uh instead of cy+uh/2 to keep the length

        #range of coordinates of upper in frame
        x1 = int(cx-uw/2)
        x2 = int(x1+uw)
        y1 = int(cy-uh/2)
        y2 = int(y1+uh)
        #the range of coordinates of upper laying outside border of frame
        ox1 = max(0-x1,0)
        ox2 = max(x2-(fw-1),0)
        oy1 = max(0-y1,0)
        oy2 = max(y2-(fh-1),0)
        #range of coordinates of upper within frame
        x11 = int(x1 + ox1)
        x21 = int(x2 - ox2)
        y11 = int(y1 + oy1)
        y21 = int(y2 - oy2)
        #range of coordinates of upper itself within frame
        x12 = int(ox1)
        x22 = int(uw - ox2)
        y12 = int(oy1)
        y22 = int(uh - oy2)

        mh = min(h,upper2.shape[1])
        mw = min(w,upper2.shape[0])

        for c in range(0,3):
            frame[y11:y21,x11:x21, c] =  \
                upper2[y12:y22,x12:x22,c] * (upper2[y12:y22,x12:x22,3]/255.0) +\
                frame[y11:y21,x11:x21,c] * \
                (1.0 - upper2[y12:y22,x12:x22,3]/255.0)

    return frame




if __name__ == "__main__":

    
    video_path = sys.argv[1]# "video/1.mp4"
    sample_rate = int(sys.argv[2])
    cursor = 0
    
    #print sys.argv[1]
    camera = cv2.VideoCapture(video_path)
    while not camera.isOpened():
        camera = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print ("wait for header")
    if camera.isOpened():
        print ("camera opened")

    #setup videowriter properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')#'M','J','P','G')
    fps = camera.get(cv2.CAP_PROP_FPS)  #?20.0
    print('source video fps = ',fps)

    #get one frame
    grabbed, frame = camera.read()

    frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
    out_video = cv2.VideoWriter('output.avi',fourcc, fps, (frame_resized.shape[1],frame_resized.shape[0]), True)
    last_faces = 0
    
    while True:
        previous_frame = frame_resized_grayscale
        previous_faces = last_faces
        grabbed, frame = camera.read()
        if not grabbed:
            break
        
        #print(frame.shape)
        frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
        #print(frame_resized.shape)
        frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        #print('frame_resized: ',frame_resized.shape)
        #print('frame_resized_grayscale: ',frame_resized_grayscale.shape)
            
        

        if cursor == 0:
            # detect faces
            faces = detectFace(frame_resized_grayscale, faceCascade, previous_frame, previous_faces)
            #print(faces)
            #eyes = eyeCascade.detectMultiScale(frame_resized_grayscale)

            
            if len(faces):
                #cv2.imshow("frame_resized",frame_resized)
                #important to show image
                frame_processed = drawHead(frame_resized, faces)
                #cv2.waitKey(0)
                #cv2.imshow("Detected Human and face", frame_processed)
                last_faces = faces
            #elif type(last_faces) != type(0):
                #frame_processed = drawHead(frame_resized, last_faces, eyes)
            else:
                frame_processed = frame_resized
        #elif type(last_faces) != type(0):
        #    frame_processed = drawHead(frame_resized, last_faces)
        else:
            frame_processed = frame_resized            
        key = cv2.waitKey(1) & 0xFF
        out_video.write(frame_processed)
        cursor = (cursor + 1)%sample_rate
    print('over')
    #camera.release()
    cv2.destroyAllWindows()
    out_video.release()

    print('time spent: ', time.time()-start_time)