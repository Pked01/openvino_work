from time import time_ns
import requests,os
from datetime import datetime
from PIL import Image
from io import StringIO,BytesIO
urls = {
    "tumkur_station" : "http://117.192.41.97:81/jpgmulreq/1/image.jpg?key=1516975535684&lq=1&1629105119",
    # "vijaywada_store_1" : "http://49.205.165.10:81/webcapture.jpg?command=snap&channel=1?1629104380",
    "vijaywada_store_2" : "http://49.205.165.10:81/webcapture.jpg?command=snap&channel=2?1629104380",
    "chennai_store" : "http://106.51.155.198:60001/cgi-bin/snapshot.cgi?chn=0&u=admin&p=&q=0&1629104492",

}
# os.chdir("/home/prateek/shared_space/Notebooks/abg/openvino/insec_cams_data_collect/dataset")
dump_path = "/home/prateek/shared_space/Notebooks/abg/openvino/Notebooks/insec_cams_data_collect/dataset/vehicle/"

def response2img(response,filepath):

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)


image_folder_dict = {}
for k,v in urls.items():
    image_folder_location = os.path.join(dump_path,k)
    image_folder_dict[k] = image_folder_location
    os.makedirs(image_folder_location,exist_ok=True)

while True: 

    time_now = datetime.now()
    current_time = str(time_now.year)+"_"+str(time_now.month)+"_"+str(time_now.day)+"_"+str(time_now.hour)+"_"+\
            str(time_now.minute)+"_"+str(time_now.second)

    for k,v in urls.items():
        r = requests.get(v, stream=True)
        image_path = os.path.join(image_folder_dict[k],current_time+".jpg")
        response2img(r,image_path)
    

 


