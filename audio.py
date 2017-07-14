import pygame
import sys
import time 
import wave

from mido import MidiFile


start_time = time.time()


import cv2
import sys
import imutils
import time
import numpy as np


start_time = time.time()
#upper image to cover detected head
uimgs = []
for i in ["pic/yushenmu.png"]:
	uimgs.append(cv2.imread(i))

def drawHead(frame, channel):
	'''
	uw: upper width, uh: upper height
	x,y,w,h: horizontal to right, vertical to bottom
	frame shape: (450, 800, 3) <= (height, width, dimensions)
	'''
	#height and width of frame
	fh = frame.shape[0]
	fw = frame.shape[1]

	temp_up = imutils.resize(uimgs[0], width=int(min(fh/2, uimgs[0].shape[1])))
	if (True):
		frame[0:temp_up.shape[0],0:temp_up.shape[1]] = temp_up
		print('0')
	print(channel)
	return frame




if __name__ == "__main__":

	print('time spent: ', time.time() - start_time)
    
	video_path = sys.argv[1]
	bgm_mid = sys.argv[2]
	time_cursor = 0.0

	on_frame = []

    
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

		#transform time cursor to corresponding frames
	for msg in MidiFile(bgm_mid):
		#time.sleep(msg.time)
		if not msg.is_meta:
			time_cursor += msg.time
			if msg.type == 'note_on':
				on_frame.append([msg.channel,int(time_cursor*fps)])


	#get one frame
	grabbed, frame = camera.read()

	frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
	scaled_ratio = frame_resized.shape[1]/800
	frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
	    
	out_video = cv2.VideoWriter('output_au.avi',fourcc, fps, (frame_resized.shape[1],frame_resized.shape[0]), True)

	frame_cursor = 1
	on_frame_cursor = 0

	while True:
	    grabbed, frame = camera.read()
	    if not grabbed:
	        break
	    
	    frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
	    frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
	    
	    
	    if on_frame_cursor < len(on_frame) and frame_cursor >= on_frame[on_frame_cursor][1]:
	    	frame_processed = drawHead(frame_resized, on_frame[on_frame_cursor][0])
	    	on_frame_cursor += 1
	    else:
	    	frame_processed = frame_resized
	    cv2.waitKey(0)
	    cv2.imshow(" ", frame_processed)
	    #cv2.imwrite('sword.png',frame_resized)
	    key = cv2.waitKey(1) & 0xFF
	    out_video.write(frame_processed)
	    
	    frame_cursor += 1
	    #cursor = (cursor + 1)%sample_rate
	print('over')
	#camera.release()
	cv2.destroyAllWindows()
	out_video.release()

	print('time spent: ', time.time()-start_time)