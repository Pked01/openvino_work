import sys,argparse
import cv2,os
import pandas as pd

def create_frame_classification_csv(video_path,class_label):
	"""
	press F to choose frame as fire
	F :[70,102]
	left arrow: 37
	right arrow: 39
	"""
	frame_number = 0
	cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
	cap = cv2.VideoCapture(video_path)
	df = pd.DataFrame(columns = ["frame_number","label"])
	filename, file_extension = os.path.splitext(video_path)
	data_path = filename+'.csv'
	df.to_csv(data_path,index = False)

	while True:
	    ret,frame = cap.read()
	    if not ret:
	        print("break : End of video or incorrect Video file")
	        break
	    cv2.putText(frame,"Frame_number = "+str(frame_number),(10,50),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),2)
	    cv2.putText(frame,"Press F to set frame as fire",(10,80),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,255,0),2)
	    cv2.putText(frame,"Press P for previous frame",(10,110),cv2.FONT_HERSHEY_SIMPLEX ,1,(120,0,120),2)
	    
	    cv2.imshow("preview",frame)
	    k = cv2.waitKey(0)
	    if k==27:
	        break
	    if k in [70,102]:
	        ### setting frame as fire
	        df = pd.DataFrame({"frame_number":frame_number,"label":class_label[0]},index = [0])
	        df.to_csv(data_path,index=False,header=False,mode = 'a')
	    else:
	        df = pd.DataFrame({"frame_number":frame_number,"label":class_label[1]},index = [0])
	        df.to_csv(data_path,index=False,header=False,mode = 'a')        
	    if k in [80,112]:
	        ### previous_frame press p
	        frame_number = frame_number-2
	        cap.set(1,frame_number)
	    frame_number+=1
	    
	df = pd.read_csv(data_path)
	df.drop_duplicates(subset="frame_number",keep="last",inplace = True)
	df.to_csv(data_path,index=False)

	cap.release()
	cv2.destroyAllWindows()





if __name__=='__main__':
	parser = argparse.ArgumentParser()
	class_label = ["fire", "no_fire"]
	parser.add_argument('--video_path','-i', dest='video_path', required=True, help='path of video file')
	args = parser.parse_args()
	create_frame_classification_csv(args.video_path,class_label)

