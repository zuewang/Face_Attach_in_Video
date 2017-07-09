import cv2
import sys
import imutils
import time

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



def drawHead(frame, faces, eyes):
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    # Draw a picture over the faces
    for (x, y, w, h) in faces:
        
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx = y+h/2
        cy = x+w/2
        upper2 = imutils.resize(upper_img, width=int(min(h, upper_img.shape[1])))
        #upper2 = imutils.resize(upper1, width=int(min(w, upper1.shape[0])))
        uh = upper2.shape[0]
        uw = upper2.shape[1]
        '''
        print('(x,y,w,h): ',(x,y,w,h))
        print('frame: ',frame.shape,', upper: ', upper2.shape)
        print('cy, cx: ', cy, ', ',cx)
        print(cy-uh/2,' ',cy-uh/2+uh)
        print(frame[cy-uh/2:cy-uh/2+uh, cx-uw/2:cx-uw/2+uw, 0].shape)
        print(frame[cy-uh/2:cy-uh/2+uh, cx-uw/2:cx-uw/2+uw, 1].shape)
        print(frame[cy-uh/2:cy-uh/2+uh, cx-uw/2:cx-uw/2+uw, 2].shape)
        '''
        # deal with alpha
        #frame[y:y+uh,x:x+uw] = upper2
        
        for c in range(0,3):
            #print('c= ',c)
            # in case face move out of border
            # use cy-uh/2+uh instead of cy+uh/2 to keep the length
            x1 = cx-uw/2
            x2 = cx-uw/2+uw
            y1 = cy-uh/2
            y2 = cy-uh/2+uh
            ox1 = max(-x1,0)
            ox2 = max(x2-w+1,0)
            oy1 = max(-y1,0)
            oy2 = max(y2-h+1,0)

            mh = min(h,upper2.shape[1])
            mw = min(w,upper2.shape[0])
            frame[y:y+mh,x:x+mw, c] =  \
                upper2[:mh,:mw,c] * (upper2[:mh,:mw,3]/255.0) +\
                frame[y:y+mh,x:x+mw,c] * \
                (1.0 - upper2[:mh,:mw,3]/255.0)
        
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
        print "wait for header"
    if camera.isOpened():
        print "camera opened"

    #setup videowriter properties
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
    fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)  #?20.0
    print('fps = ',fps)

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
            faces = faceCascade.detectMultiScale(
                frame_resized_grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(5, 5),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            eyes = eyeCascade.detectMultiScale(frame_resized_grayscale)

            
            if len(faces):
                #cv2.imshow("frame_resized",frame_resized)
                #important to show image
                #cv2.waitKey(0)
                frame_processed = drawHead(frame_resized, faces, eyes)
                #cv2.imshow("Detected Human and face", frame_processed)
                last_faces = faces
            #elif type(last_faces) != type(0):
                #frame_processed = drawHead(frame_resized, last_faces, eyes)
            else:
                frame_processed = frame_resized

        elif type(last_faces) != type(0):
            frame_processed = drawHead(frame_resized, last_faces, eyes)
        else:
            frame_processed = frame_resized            
        key = cv2.waitKey(1) & 0xFF
        out_video.write(frame_processed)
        cursor = (cursor + 1)%sample_rate
        
    #camera.release()
    cv2.destroyAllWindows()
    out_video.release()

    print('time spent: ', time.time()-start_time)

'''

# Detect faces in the image
# faces = faceCascade.detectMultiScale(image, 1.1, 2, 0, (20, 20) )


print("Found {0} faces!".format(len(faces)))
drawHead(image,faces)
        #image[y:y+upper_img.shape[0],x:x+upper_img.shape[1]]=upper_img
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
'''
