#Instructions
# make sure there is good contrast between the background and your document
# python scan_document.py --image ~/some/image/file/path.jpg

import argparse
import cv2
import numpy as np
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="File path to image of document to be scanned")
ap.add_argument("-d","--debug",help="flag for debug mode", action='store_true')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#resize for contour finding
w_resize = 500
ratio = h/w

dim = (w_resize, int(w_resize * ratio))
resized = cv2.resize(gray, dim)

w_ratio = w / resized.shape[1]
h_ratio = h / resized.shape[0]

if args["debug"]:
    cv2.imshow("original", image)
    cv2.imshow("prepped image", resized)
    cv2.waitKey(0)

#find edges
blurred = cv2.GaussianBlur(resized, (11, 11), 0)
sigma = 0.33
v = np.median(blurred)
lower = int(max(0,(1.0-sigma)*v))
upper = int(min(255,(1.0+sigma)*v))
edges = cv2.Canny(blurred, lower, upper)

#debug
if args["debug"]:
    cv2.imshow("Edges View", edges)
    cv2.waitKey(0)

#find contours of document
#relax contours until there are 4 edges

contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]

#document contour should be the largest contour area seen
contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

if args["debug"]:
    debug_edge = resized.copy()
    cv2.drawContours(debug_edge, contour, -1, (0, 255, 0), 2)
    cv2.imshow("document edge",debug_edge)
    cv2.waitKey(0)

peri = cv2.arcLength(contour[0], True)
peri_factor = 0.01

approx_cnt = cv2.approxPolyDP(contour[0], peri_factor * peri, True)

while len(approx_cnt) != 4 and peri_factor < 0.03:
    peri_factor += 0.005
    approx_cnt = cv2.approxPolyDP(contour[0], peri_factor * peri, True)

#exit with error if document cant be found
if len(approx_cnt) != 4 :
    print("Error: could not find 4 corners of the document: {} corners found".format(len(approx_cnt)))
    sys.exit(1)

#reduce dims from 4,1,2 to 4,2
approx_cnt = np.squeeze(approx_cnt, axis = 1)

#order document vertices TL,TR,BL,BR

#order vertical
approx_cnt = sorted(approx_cnt, key= lambda x: x[1])

#then horizontal for top and bottom sets of vertices
approx_cnt[:2] = sorted(approx_cnt[:2], key=lambda x: x[0])
approx_cnt[2:] = sorted(approx_cnt[2:], key=lambda x: x[0])

if args["debug"]:
    debug_edge = resized.copy()
    if len(approx_cnt) == 4:
        for coord in approx_cnt:
            print(coord)
            coord = np.array([coord])
            cv2.drawContours(debug_edge, [coord], -1, (255, 0, 0), 2)
            cv2.imshow("approx doc edge",debug_edge)
            cv2.waitKey(0)

#convert to float32
approx_cnt = np.array(approx_cnt, dtype =np.float32)

#scale contour vertices to match original image dimensions in order to scan at higher resolution
for pt in approx_cnt:
    pt[0] = pt[0]*w_ratio
    pt[1] = pt[1]*h_ratio

doc_w = approx_cnt[1][0] - approx_cnt[0][0]
doc_h = approx_cnt[3][1] - approx_cnt[0][1]


#determine dimensions of output image
ratio_output = 11/8.5
if doc_w > doc_h:
    h_output= doc_w / ratio_output
else:
    h_output= doc_w * ratio_output

dims_out = (int(doc_w), int(h_output))
scanned = np.zeros(dims_out, dtype=np.uint8)
scanned_corners = np.array([[0, 0], [dims_out[0] - 1, 0],
                            [0, dims_out[1] - 1], [dims_out[0] - 1, dims_out[1] - 1]], dtype =np.float32)

#compute transfrom between input image document corners and output image major dimensions
#apply transform
transform = cv2.getPerspectiveTransform(approx_cnt, scanned_corners)
scanned = cv2.warpPerspective(gray, transform, dims_out)

#write to output
filename = args["image"][ : args["image"].rfind(".")] + "_scanned.jpg"
cv2.imshow(filename, scanned)
cv2.waitKey(0)
cv2.imwrite(filename, scanned)
