import cv2
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Line3D

fig = pyplot.figure()
ax = Axes3D(fig)
for file in os.listdir("calibration pics"):
    img = cv2.imread("calibration pics/"+file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = []
    s = []
    v = []
    for row in hsv:
        for pix in row:
            if not (pix[0] == 0 and pix[1] == 0 and pix[2] == 0):
                h.append(pix[0])
                s.append(pix[1])
                v.append(pix[2])
ax.set_xlabel('h')
ax.set_ylabel('s')
ax.set_zlabel('v')
ax.scatter(h, s, v)
pyplot.show()

# grid calculations
def distance(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)

def combine(i,j):
    return np.average(a[i:j+1],axis=0)

p = np.array(pts)
a = []
for line in p:
    av=np.average(line, axis=0)
    a.append(av)

c = []
start = 0
stop = 0
for i in range(len(a)-1):
    d = distance(a[i],a[i+1])
    if d>1:
        print(i,"->",i+1,": ",d)
        c.append(combine(start,i))
        start = i+1
c.append(a[55])
c = np.delete(c,5,0)
c = np.array(c)

fig = pyplot.figure()
ax = Axes3D(fig, azim=-90, elev=90)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(c[:,0], c[:,1], c[:,2], marker=".")
pyplot.show()

m = np.zeros((3,3,3,3))
for pt in c:
    if pt[2] > 25.5 :
        if pt[1] > -0.5:
            print(pt)

# m[0,0,0]=[-0.76909437, -1.79501393, 19.67269933]
# m[1,0,0]=[ 1.09933398, -1.73251016, 19.23297579]
# m[2,0,0]=[ 2.68673256, -1.75178156, 18.91545972]
# m[0,1,0]=[-0.78600492,  0.03084422, 20.03940513]
# m[1,1,0]=[ 1.03743141, -0.02887931, 19.53614374]
# m[2,1,0]=[ 2.7746796,   0.03875156, 19.21323482]
# m[0,2,0]=[-0.70037812,  2.00504473, 19.31686334]
# m[1,2,0]=[ 0.99864526,  1.83225414, 18.99255824]
# m[2,2,0]=[ 2.79293226,  1.87100416, 19.34957933]
#
# m[0,0,1]=[-0.63923095, -1.55336167, 23.10948798]
# m[1,0,1]=[ 1.22827151, -1.53121634, 22.89666243]
# m[2,0,1]=[ 2.82571112, -1.53911854, 22.92110235]
# m[0,1,1]=[-0.70601444,  0.19690447, 23.44368506]
# m[1,1,1]=[ 1.14607697,  0.22892739, 23.68813074]
# m[2,1,1]=[ 2.87257968,  0.22554941, 22.84709657]
# m[0,2,1]=[-0.74863339,  2.06777042, 22.74769003]
# m[1,2,1]=[ 1.06873505 , 2.05723517, 23.15952859]
# m[2,2,1]=[ 2.85903196 , 2.01478492, 22.91497078]
#
# m[0,0,2]=[-0.50473982, -1.38055799, 27.25350272]
# m[1,0,2]=[ 1.32016166, -1.39651242, 27.29071088]
# m[2,0,2]=[ 3.05693574, -1.39085416, 27.19041852]
# m[0,1,2]=[-0.53092475,  0.37746988, 27.23409331]
# m[1,1,2]=[ 1.17287132,  0.37618205, 26.87633566]
# m[2,1,2]=[ 3.14717526,  0.42218   , 27.20433104]
# m[0,2,2]=[-0.76897878,  2.25547972, 26.44721101]
# m[1,2,2]=[ 1.28294373,  2.24568539, 27.24741827]
# m[2,2,2]=[ 2.98307018 , 2.18773684, 26.57319454]

m[0,0,0]=c[0]
m[1,0,0]=c[1]
m[2,0,0]=c[2]
m[0,1,0]=c[3]
m[1,1,0]=c[4]
m[2,1,0]=np.array([2.54, -1.21, 10.8])
m[0,2,0]=c[5]
m[1,2,0]=c[6]
m[2,2,0]=c[7]

m[0,0,1]=c[8]
m[1,0,1]=c[9]
m[2,0,1]=c[10]
m[0,1,1]=c[11]
m[1,1,1]=c[12]
m[2,1,1]=c[13]
m[0,2,1]=c[14]
m[1,2,1]=c[15]
m[2,2,1]=c[16]

m[0,0,2]=c[17]
m[1,0,2]=c[18]
m[2,0,2]=c[19]
m[0,1,2]=c[20]
m[1,1,2]=c[21]
m[2,1,2]=c[22]
m[0,2,2]=c[23]
m[1,2,2]=c[24]
m[2,2,2]=c[25]

xdist=[]
ydist=[]
zdist=[]
for i in range(3):
    for j in range(3):
        for k in range(3):
            if i+1 < 2:
                print("(",i,",",j,",",k,")->(",i+1,",",j,",",k,") = ",distance(m[i,j,k], m[i+1,j,k]))
                xdist.append(distance(m[i,j,k], m[i+1,j,k]))
            if j+1 < 2:
                print("(",i,",",j,",",k,")->(",i,",",j+1,",",k,") = ",distance(m[i,j,k], m[i,j+1,k]))
                ydist.append(distance(m[i, j, k], m[i, j+1, k]))
            if k+1 < 2:
                print("(",i,",",j,",",k,")->(",i,",",j,",",k+1,") = ",distance(m[i,j,k], m[i,j,k+1]))
                zdist.append(distance(m[i, j, k], m[i, j, k+1]))

print(np.average(xdist))
print(np.average(ydist))
print(np.average(zdist))

truth = np.zeros((3,3,3,3))
center = m [1,1,1]
for i in range(3):
    for j in range(3):
        for k in range(3):
            truth[i,j,k] = center + [(i-1)*2,(j-1)*2,(k-1)*2]

nx=[]
ny=[]
nz=[]
for i in range(3):
    for j in range(3):
        for k in range(3):
            nx.append(m[i,j,k,0])
            ny.append(m[i,j,k,1])
            nz.append(m[i,j,k,2])

fig = pyplot.figure()
ax = Axes3D(fig, azim=-90, elev=90)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(nx, ny, nz, marker="o", c="red")

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i+1 < 3:
                p1 = truth[i,j,k]
                p2 = truth[i+1,j,k]
                line = Line3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]))
                ax.add_line(line)
            if j+1 < 3:
                p1 = truth[i, j, k]
                p2 = truth[i , j+1, k]
                line = Line3D((p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2]))
                ax.add_line(line)
            if k+1 < 3:
                p1 = truth[i, j, k]
                p2 = truth[i , j, k+1]
                line = Line3D((p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2]))
                ax.add_line(line)

pyplot.show()

xdist=[]
ydist=[]
zdist=[]
threedist=[]
for i in range(3):
    for j in range(3):
        for k in range(3):
            xdist.append(math.fabs(m[i,j,k,0]-truth[i,j,k,0]))
            ydist.append(math.fabs(m[i, j, k, 1] - truth[i, j, k, 1]))
            zdist.append(math.fabs(m[i, j, k, 2] - truth[i, j, k, 2]))
            threedist.append((distance(m[i, j, k], truth[i,j,k])))
print(np.average(xdist))
print(np.average(ydist))
print(np.average(zdist))