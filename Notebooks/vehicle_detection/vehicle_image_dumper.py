from argparse import ArgumentParser, SUPPRESS
import os,cv2,sys,pdb,time
os.sys.path.append("../")
from utilities import support_utility_openvino
from datetime import datetime
import numpy as np


def build_argparser():
  parser = ArgumentParser(add_help=False)
  args = parser.add_argument_group('Options')
  args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
  args.add_argument("-m", "--model", help="Path to an .xml file with  vehicle detection model.", required=True, type=str)
  args.add_argument("-i", "--input", help="Path to the input rtsp feed", required=True,
                    type=str)
  args.add_argument("-o", "--output", help="Path to a folder to save images", required=True,
                    type=str)
  args.add_argument("-sl", "--sleep_time", help="Time(in seconds) for which we want to sleep the script", required=True,
                    type=int)

  args.add_argument("-d", "--device",
                    help="Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Sample "
                         "will look for a suitable plugin for device specified. Default value is CPU", default="CPU",
                    type=str)
  args.add_argument("-t", "--threshold",
                    help="threshold for vehicle detection model", default=0.5,
                    type=float)
  args.add_argument("-p","--plot",help="to plot result or not",type=bool,default=True)
  args.add_argument("-ret","--retry_times",default=100,  help = "number of times connection retries to rstp have to be made before terminating the program",
                    type=str)


  return parser

def main():
  res_thresh = [.005,.5]
  args = build_argparser().parse_args()
  orig_frame_path = os.path.join(args.output,"original")
  cropped_frame_path = os.path.join(args.output,"cropped")
  os.makedirs(orig_frame_path,exist_ok=True)
  os.makedirs(cropped_frame_path,exist_ok=True)

  output_support = support_utility_openvino.create_plot()
  vehicle_detection = support_utility_openvino.async_infer(2)
  vehicle_detection.load_model(model_path=args.model,device=args.device)

  # print(args.input)
  cap = cv2.VideoCapture(args.input)
  # cap.set(1,500)  
  img_idx_1 = 0
  img_idx_2 = 0
  while cap.isOpened():
    
    for i in range(int(args.sleep_time*cap.get(cv2.CAP_PROP_FPS))):
      ret,frame = cap.read()
    # cap.set(1,int(cap.get(1)+30*args.sleep_time))

    if not ret:
      while True:
        args.retry_times-=1
        cap = cv2.VideoCapture(args.input)
        ret,frame  = cap.read()
        if ret or args.retry_times<=0:
          
          break
    bboxes = []
    vehicle_detection.predict(frame)
    if vehicle_detection.frame_processed>=vehicle_detection.num_requests:
      frame,attr,res = vehicle_detection.postprocess_op()
      
      # result = res[0]
      # res_filt =  result[np.where(result[:,:,:,2]>args.threshold)]
      # res_filt = output_support.resolution_filter(res_filt,resolution_thresh_range=[.001,.03])
      # pdb.set_trace()
      # print(res_filt.shape)
      # print(res_filt)
      # bboxes = output_support.filter_result_bbox(frame,res[0],args.threshold)
      if args.plot:
        op_frame=None
        op_frame = output_support.label_obj_detection(frame, res[0],thresh=args.threshold,labels=["vehicle"],resolution_thresh_range=res_thresh)
        cv2.imshow("preview",op_frame)
        k = cv2.waitKey(1)
        if k==ord('q'):
          cv2.destroyAllWindows()
          args.plot = False
      cropped_frames,updated_bbox,res_filt = output_support.trim_frame_with_result(frame,res[0],args.threshold, resolution_thresh_range=res_thresh\
        ,return_results=True,offset=10)
      time_now = datetime.now()
      
      if res_filt.shape[0]>0:
        # print(res_filt)
        filename_1 = os.path.join(orig_frame_path,str(time_now.year)+"_"+str(time_now.month)+"_"+str(time_now.day)+"_"+str(time_now.hour)+"_"+\
          str(time_now.minute)+"_"+str(time_now.second)+"_"+str(img_idx_1).zfill(6)+".jpg")
        cv2.imwrite(filename_1,frame)
        img_idx_1+=1
        for cropped_frame in cropped_frames:
          filename_2 = os.path.join(cropped_frame_path,str(time_now.year)+"_"+str(time_now.month)+"_"+str(time_now.day)+"_"+str(time_now.hour)+"_"+\
          str(time_now.minute)+"_"+str(time_now.second)+"_"+str(img_idx_2).zfill(6)+".jpg")
          cv2.imwrite(filename_2,cropped_frame)
          img_idx_2+=1
        print("number of original images = {}, number of cropped images = {}".format(img_idx_1,img_idx_2))

    time.sleep(args.sleep_time)


if __name__ == '__main__':
    sys.exit(main() or 0)
  



