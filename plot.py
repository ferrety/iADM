import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
results=pickle.load(open("results.dmp","r"))

rnsga=results[(3, 'rNSGAII_solution', 'DTLZ1')]
asf=results[(3, 'ACH_solution', 'DTLZ1')]

c=['b','g','r','c','m','y','k']


for i,r in enumerate(asf[:len(c)]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p=np.array(r[2][1])
    ax.scatter(p[0,0],p[0,1],p[0,2], c=c[i], marker='D')
    ax.scatter(p[-1,0],p[-1,1],p[-1,2], c=c[i], marker='x')
    #ax.scatter(p[1:-1,0],p[1:-1,1],p[1:-1,2], c=c[i], marker='x')

    ax.plot(p[:,0],p[:,1],p[:,2],c=c[i])

    ax.scatter(r[0][0],r[0][1],r[0][2], c=c[i], marker='o')

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    fig.draw()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,r in enumerate(rnsga[:len(c)]):
    p=np.array(r[2][0])
    ax.scatter(p[:,0],p[:,1],p[:,2], c=c[i], marker='x')

    ax.scatter(r[0][0],r[0][1],r[0][2], c=c[i], marker='o')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,r in enumerate(rnsga[:len(c)]):
    p=np.array(r[2][1])
    ax.scatter(p[:,0],p[:,1],p[:,2], c=c[i], marker='x')

    ax.scatter(r[0][0],r[0][1],r[0][2], c=c[i], marker='o')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
plt.show()
