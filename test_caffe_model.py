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

def compare_diff_sum(tensor1, tensor2):
    pass

def compare_cosin_dist(tensor1, tensor2):
    pass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

'''
Function:
	apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
	rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
	return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
	xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
	w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
	if type == 'iom':
	    o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
	else:
	    o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
	pick.append(I[-1])
	I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle
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
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    pick = []
    for i in range(len(rectangles)):
	x1 = int(max(0     ,rectangles[i][0]))
	y1 = int(max(0     ,rectangles[i][1]))
	x2 = int(min(width ,rectangles[i][2]))
	y2 = int(min(height,rectangles[i][3]))
	if x2>x1 and y2>y1:
	    pick.append([x1,y1,x2,y2,rectangles[i][4],
			 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.7,'iom')

def compare_models(prefix_caffe, size):
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
        cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    for i in range(5,15,2):
    	cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("test",draw)
    cv2.waitKey()
    cv2.imwrite('test.jpg',draw)

    
if __name__ == "__main__":
    prefix_caffe = "/var/darknet/MXNet2Caffe/model_caffe/"
    size = (1, 3, 112, 112)
    compare_models(prefix_caffe, size)
