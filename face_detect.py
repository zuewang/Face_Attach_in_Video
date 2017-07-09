import cv2
import sys
import imutils
import time

start_time = time.time()

upper_img = cv2.imread("pic/cirno_face.png", -1)
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
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx = y+h/2
        cy = x+w/2
        upper2 = imutils.resize(upper_img, width=int(min(1.5*w, 1.5*upper_img.shape[1])))
       
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
        '''
        for c in range(0,3):
            #print('c= ',c)
            # in case face move out of border
            # use cy-uh/2+uh instead of cy+uh/2 to keep the length
            x1 = cx-uw/2
            x2 = cx-uw/2+uw
            y1 = cy-uh/2
            y2 = cy-uh/2+uh
            outrangex1 = max(-x1,0)
            outrangex2 = max(x2-w+1,0)
            outrangey1 = max(-y1,0)
            outrangey2 = max(y2-h+1,0)
            frame[y:y+uh,x:x+uw, c] =  \
                upper2[:,:,c] * (upper2[:,:,3]/255.0) +\
                frame[y:y+uh,x:x+uw, c] * \
                (1.0 - upper2[:,:,3]/255.0)
        '''
    return frame






video_path = "video/1.mp4"
camera = cv2.VideoCapture(video_path)
while not camera.isOpened():
    camera = cv2.VideoCapture(video_path)
    cv2.waitKey(1000)
    print "wait for header"
if camera.isOpened():
    print "camera opened"
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
fps=20.0
out_video = cv2.VideoWriter('output.avi',fourcc, fps, (800,450), True)
while True:
    #previous_frame = frame_resized_grayscale
    grabbed, frame = camera.read()
    if not grabbed:
        break
    
    #print(frame.shape)
    frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
    #print(frame_resized.shape)
    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    #print('frame_resized: ',frame_resized.shape)
    #print('frame_resized_grayscale: ',frame_resized_grayscale.shape)
        
    


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
        key = cv2.waitKey(1) & 0xFF
        #print(frame_processed.shape)
        out_video.write(frame_processed)
    
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
