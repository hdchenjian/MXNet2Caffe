from operator import itemgetter
try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/opt/ego/caffe-rcnn-face-ssd/python"))
    import caffe
import numpy as np
import cv2

'''
Function:
	apply NMS(non-maximum suppression) on ROIs in same scale
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is scale, rectangles[i][5] is score
Output:
	rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    sorted(rectangles,key=itemgetter(4),reverse=True)
    result_rectangles = rectangles
    number_of_rects = len(result_rectangles)
    cur_rect = 0
    while cur_rect < number_of_rects : 
        rects_to_compare = number_of_rects - cur_rect - 1 
        cur_rect_to_compare = cur_rect + 1 
        while rects_to_compare > 0:
	    score = 0
	    if type == 'iou':
		score =  IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare])
	    else:
		score =  IoM(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare])
            if score >= threshold:
                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle            
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle
    return result_rectangles

'''
Function:
	Filter face position and calibrate bounding box on 12net's output
Input:
	cls_prob  : cls_prob[1] is face possibility
	roi       : roi offset
	pts       : 5 landmark
	rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
	width     : image's origin width
	height    : image's origin height
	threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
	rectangles: face positions and landmarks
'''
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    boundingBox = []
    rect_num = len(rectangles)
    for i in range(rect_num):
	if cls_prob[i][1]>threshold:
	    rect = [rectangles[i][0],rectangles[i][1],rectangles[i][2],rectangles[i][3],cls_prob[i][1],
		   roi[i][0],roi[i][1],roi[i][2],roi[i][3],
		   pts[i][0],pts[i][5],pts[i][1],pts[i][6],pts[i][2],pts[i][7],pts[i][3],pts[i][8],pts[i][4],pts[i][9]]
	    boundingBox.append(rect)
    rectangles = NMS(boundingBox,0.7,'iom')
    rect = []
    
    for rectangle in rectangles:
	roi_w = rectangle[2]-rectangle[0]+1
	roi_h = rectangle[3]-rectangle[1]+1

  	x1 = round(max(0     , rectangle[0]+rectangle[5]*roi_w))
        y1 = round(max(0     , rectangle[1]+rectangle[6]*roi_h))
        x2 = round(min(width , rectangle[2]+rectangle[7]*roi_w))
        y2 = round(min(height, rectangle[3]+rectangle[8]*roi_h))
	pt0 = rectangle[ 9]*roi_w + rectangle[0] -1
	pt1 = rectangle[10]*roi_h + rectangle[1] -1
	pt2 = rectangle[11]*roi_w + rectangle[0] -1
	pt3 = rectangle[12]*roi_h + rectangle[1] -1
	pt4 = rectangle[13]*roi_w + rectangle[0] -1
	pt5 = rectangle[14]*roi_h + rectangle[1] -1
	pt6 = rectangle[15]*roi_w + rectangle[0] -1
	pt7 = rectangle[16]*roi_h + rectangle[1] -1
	pt8 = rectangle[17]*roi_w + rectangle[0] -1
	pt9 = rectangle[18]*roi_h + rectangle[1] -1
	score = rectangle[4]
	rect_ = np.round([x1,y1,x2,y2,pt0,pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8,pt9]).astype(int)
	rect_= np.append(rect_,score)
	rect.append(rect_)
    return rect

def compare_models(prefix_caffe):
    threshold = [0.6,0.6,0.7]
    deploy = prefix_caffe + '48net.prototxt'
    caffemodel = prefix_caffe + '48net.caffemodel'
    deploy = prefix_caffe + 'det3_permute.prototxt'
    caffemodel = prefix_caffe + 'det3_bgr.caffemodel'
    net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

    img = cv2.imread('face1.jpg')
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    rectangles = [[127.0, 118.0, 481.0, 473.0, 0.9980586171150208], [353.0, 198.0, 394.0, 239.0, 0.6727402210235596]]
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] = scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
    print 'real: ', [178.0, 117.0, 479.0, 498.0]
    print rectangles
    print "done"
    draw = cv2.imread('face1.jpg')
    for rectangle in rectangles:
        cv2.putText(draw,str(rectangle[14]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    for i in range(4,14,2):
    	cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("test",draw)
    cv2.waitKey()
    cv2.imwrite('test.jpg',draw)

    
if __name__ == "__main__":
    prefix_caffe = "/var/darknet/MXNet2Caffe/model_caffe/"
    compare_models(prefix_caffe)
