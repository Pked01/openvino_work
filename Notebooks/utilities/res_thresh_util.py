import cv2
def get_resolution_thresh(image):
"""
return resolution thresholds for an image
image : image 
"""
    cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (20, 20) 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2
    image1 = cv2.putText(image.copy(), "draw MINIMUM sized ROI", org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
    r1 = cv2.selectROI("preview",image1,False,False)
    image2 = cv2.putText(image.copy(), "draw MAXIMUM sized ROI", org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
    r2 = cv2.selectROI("preview",image2,False,False)
    cv2.destroyAllWindows()
    return (r1[2]*r1[3])/(image.shape[0]*image.shape[1]),(r2[2]*r2[3])/(image.shape[0]*image.shape[1])

frame = cv2.imread("path")
get_resolution_thresh(frame)