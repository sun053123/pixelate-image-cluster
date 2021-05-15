import cv2
import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from collections import Counter
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance

print("Program Function 2 Form original image to Pixelate With Color pantone")
print('ตอ้งการรูปกี่สี:')
noofcol = int(input())

print('ต้องการ pantone รูปคนอื่น หรือ รูปเดิมให้เป็น pixelate : 1 = รูปตัวเอง / 2 = รูปคนอื่น')
pan = int(input())

print('ต้องการรูปกี่ block')
blockimg = int(input())

#import img
img1 = cv2.imread(r"/Users/chomusuke/Desktop/RubikProjectPic/audrey.jpg")
print('dimension ของรูป original = ', img1.shape[0], img1.shape[1])
#enc img for pan1
imageENC = Image.open('/Users/chomusuke/Desktop/RubikProjectPic/A.png')
contrast = ImageEnhance.Contrast(imageENC)
contrast.enhance(1.5).save('/Users/chomusuke/Desktop/RubikProjectPic/Aenc.png')

imageENCed = cv2.imread(r"/Users/chomusuke/Desktop/RubikProjectPic/Aenc.png")
# KERNEL IMG 5*5

kernel = np.ones((5,5),np.float32)/25
img1kerneled = cv2.filter2D(imageENCed,-1,kernel)

plt.subplot(121),plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(img1kerneled, cv2.COLOR_BGR2RGB)),plt.title('Averaging kernel 1/25 , 5x5 reducepixel in same resolution')
plt.xticks([]), plt.yticks([])
plt.show()

#สีที่จะเอาเข้ามาใส่ copy pantone
percent_sclae = 10  # อยากได้กี่เปอเซ็นใส่ตรงนี้
height_resize = int(img1.shape[0] * percent_sclae / 100)
width_resize = int(img1.shape[1] * percent_sclae / 100)
dsize = (width_resize, height_resize)
if (pan == 1):
    img1kerneled = cv2.resize(img1kerneled, dsize)
    img1kerneled = cv2.cvtColor(img1kerneled, cv2.COLOR_BGR2RGB)
if (pan == 2):
    img1pan = cv2.imread(r"/Users/chomusuke/Desktop/RubikProjectPic/A.png")
    kernel = np.ones((5, 5), np.float32) / 25
    img1kerneled = cv2.filter2D(img1pan, -1, kernel)
    img1kerneled = cv2.resize(img1kerneled, dsize)
    img1kerneled = cv2.cvtColor(img1kerneled, cv2.COLOR_BGR2RGB)

modified_image = cv2.resize(img1kerneled, (img1kerneled.shape[0], img1kerneled.shape[1]), interpolation = cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

clf = KMeans(n_clusters = noofcol) #no. of col
labels = clf.fit_predict(modified_image , cv2.COLOR_BGR2RGB)
counts = Counter(labels)
center_colors = clf.cluster_centers_
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

plt.figure(figsize = (10, 10))
plt.pie(counts.values())
plt.show()
print(rgb_colors)

img = img1
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = noofcol
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width = res2.shape[:2]

# Resize input to "pixelated" size
h = blockimg
w = blockimg
temp = cv2.resize(res2, (h, w), interpolation=cv2.INTER_LINEAR)

# Initialize output image


plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
plt.show()

print('สีที่ใช้' , rgb_colors)
h, w, bpp = np.shape(temp)

for py in range(0, h):
    for px in range(0, w):
        ########################
        # หาค่าสีใกล้เคียงกับที่ bing ค่าที่สุด
        # reference : https://stackoverflow.com/a/22478139/9799700
        input_color = (temp[py][px][0], temp[py][px][1], temp[py][px][2])
        tree = sp.KDTree(rgb_colors)
        ditsance, result = tree.query(input_color)
        nearest_color = rgb_colors[result]
        ###################

        temp[py][px][0] = nearest_color[0]
        temp[py][px][1] = nearest_color[1]
        temp[py][px][2] = nearest_color[2]

# show image
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
plt.show()
