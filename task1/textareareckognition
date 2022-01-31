import cv2
import numpy as np

filename = '2_noise.png'
imgcol = cv2.imread(filename, cv2.IMREAD_COLOR)
img = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)
x,y = img.shape


#απαλοιφη θορυβου

if filename == '2_noise.png' or filename == '3_noise.png' or filename == '4_noise.png' or filename == '5_noise.png':
    for i in range(0, x):
        for j in range(0, y):
            pixels = []
            if i > 0 and i <= x- 2 and j > 0 and j <= y - 2:
                for k in range(i - 1, i + 2):
                    for m in range(j - 1, j + 2):
                        pixels.append(img[k, m])

                pixels.sort()
                # img[i, j] = st.median(pixels)
                img[i, j] = pixels[4]
            if i == 0 and j == 0:
                pixels = [255, 255, 255, 255, 255]
                pixels.append(img[i, j])
                pixels.append(img[i + 1, j])
                pixels.append(img[i, j + 1])
                pixels.append(img[i + 1, j + 1])
                pixels.sort()
                img[i, j] = pixels[4]
            if i == 0 and j == y - 1:
                pixels = [255, 255, 255, 255, 255]
                pixels.append(img[i, j - 1])
                pixels.append(img[i + 1, j - 1])
                pixels.append(img[i, j - 2])
                pixels.append(img[i + 1, j - 2])
                pixels.sort()
                img[i, j] = pixels[4]
            if i == x - 1 and j == 0:
                pixels = [255, 255, 255, 255, 255]
                pixels.append(img[i - 1, j])
                pixels.append(img[i - 2, j])
                pixels.append(img[i - 1, j + 1])
                pixels.append(img[i - 2, j + 1])
                pixels.sort()
                img[i, j] = pixels[4]
            if i == x - 1 and j == y - 1:
                pixels = [255, 255, 255, 255, 255]
                pixels.append(img[i - 1, j - 1])
                pixels.append(img[i - 2, j - 1])
                pixels.append(img[i - 1, j - 2])
                pixels.append(img[i - 2, j - 2])
                pixels.sort()
                img[i, j] = pixels[4]
            if i == 0:
                pixels = [255,255,255]
                pixels.append(img[0,j-2])
                pixels.append(img[0,j-1])
                pixels.append(img[0,j])
                pixels.append(img[1,j-2])
                pixels.append(img[1,j-1])
                pixels.append(img[1,j])
                pixels.sort()
                img[i, j-1] = pixels[4]
            if j == 0:
                pixels = [255,255,255]
                pixels.append(img[i-2,0])
                pixels.append(img[i-1,0])
                pixels.append(img[i,0])
                pixels.append(img[i-2,1])
                pixels.append(img[i-1,1])
                pixels.append(img[i,1])
                pixels.sort()
                img[i-1, j] = pixels[4]
            if i == x-1:
                pixels = [255,255,255]
                pixels.append(img[i-2,j-2])
                pixels.append(img[i-2,j-1])
                pixels.append(img[i-2,j])
                pixels.append(img[i-1,j-2])
                pixels.append(img[i-1,j-1])
                pixels.append(img[i-1,j])
                pixels.sort()
                img[i, j-1] = pixels[4]
            if j == y-1:
                pixels = [0,0,0]
                pixels.append(img[i-2,j-2])
                pixels.append(img[i-1,j-2])
                pixels.append(img[i,j-2])
                pixels.append(img[i-2,j-1])
                pixels.append(img[i-1,j-1])
                pixels.append(img[i,j-1])
                pixels.sort()
                img[i-1, j] = pixels[4]



#μορφολογικες πραξεις στις εικονες για διαχωρισμο περιοχων και λεξεων

if filename =='2_noise.png' or filename == '2_original.png':
    ret, th1 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((45, 45), np.uint8)
    dilation = cv2.dilate(opening, kernel, iterations=1)

    kernel = np.ones((11, 11), np.uint8)
    wordsim = cv2.dilate(opening, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    wordsimg= cv2.morphologyEx(wordsim, cv2.MORPH_OPEN, kernel, iterations=2)

if filename =='3_noise.png' or filename == '3_original.png':
    ret, th1 = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(th1, kernel, iterations=1)

    kernel = np.ones((41, 41), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    kernel = np.ones((13, 13), np.uint8)
    wordsim = cv2.dilate(erosion, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    wordsimg= cv2.morphologyEx(wordsim, cv2.MORPH_OPEN, kernel, iterations=1)


if filename =='4_noise.png' or filename == '4_original.png':
    ret, th1 = cv2.threshold(img, 108, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(th1, kernel, iterations=1)

    kernel = np.ones((51, 51), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    kernel = np.ones((11, 11), np.uint8)
    wordsim = cv2.dilate(th1, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wordsimg = cv2.morphologyEx(wordsim, cv2.MORPH_OPEN, kernel, iterations=2)

if filename=='5_noise.png' or filename=='5_original.png':
    ret, th1 = cv2.threshold(img, 197, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(th1, kernel, iterations=1)

    kernel = np.ones((51, 51), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    kernel = np.ones((15, 15), np.uint8)
    wordsim = cv2.dilate(erosion, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    wordsimg= cv2.morphologyEx(wordsim, cv2.MORPH_OPEN, kernel, iterations=1)

#δημιουργια περιοχων και υπολογισμος ερωτηματων στις υπο-περιοχες

output = cv2.connectedComponentsWithStats(dilation)
num=output[0]
stats=output[2]
for l in range(1, num ):
    x = stats[l, cv2.CC_STAT_LEFT]
    y = stats[l, cv2.CC_STAT_TOP]
    w = stats[l, cv2.CC_STAT_WIDTH]
    h = stats[l, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(imgcol, (x, y), (x + w, y + h), (0,0,255), 2)
    cv2.putText(imgcol, str(l), (x+5 , y + 37), cv2.FONT_HERSHEY_SIMPLEX, 1.8 , (255, 0, 0), 5)
    area = w*h
    n_word_pxl = np.sum( th1[y:y+h, x:x+w] == 255)
    words = cv2.connectedComponents(wordsimg[y:y+h, x:x+w])
    intimg = cv2.integral(img[y:y+h, x:x+w])
    sum = (intimg[0, 0] + intimg[-1, -1] - intimg[0, -1] - intimg[-1, 0])
    graylevel = sum / area
    print('area No:', l, '\n pixels:', n_word_pxl, '\n box area:', area, '\n number of words:', words[0], "\n gray level:", graylevel)


if filename == '2_noise.png' : cv2.imwrite('newnoise1.png',imgcol)
if filename == '3_noise.png' : cv2.imwrite('newnoise2.png',imgcol)
if filename == '4_noise.png' : cv2.imwrite('newnoise3.png',imgcol)
if filename == '5_noise.png' : cv2.imwrite('newnoise4.png',imgcol)
if filename == '2_original.png' : cv2.imwrite('neworiginal1.png',imgcol)
if filename == '3_original.png' : cv2.imwrite('neworiginal2.png',imgcol)
if filename == '4_original.png' : cv2.imwrite('neworiginal3.png',imgcol)
if filename == '5_original.png' : cv2.imwrite('neworiginal4.png',imgcol)

scale_percent = 30  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img2 = cv2.resize(imgcol, dim, interpolation=cv2.INTER_AREA)
cv2.namedWindow('bin', )
cv2.imshow('bin', img2)
cv2.waitKey(0)
