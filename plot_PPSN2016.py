# coding: utf-8
""" Export png graphics for PPSN2016 article from the results_2016-04-16.dmp file  
"""

import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
if not len(sys.argv) >1:
    print('Using default dmp file')
    dmpf='results_2016-04-16.dmp'
else:
    dmpf=sys.argv[1]
    
results=pickle.load(open(dmpf,"r"))

rnsga=results[(3, 'rNSGAII_solution', 'DTLZ1')]
asf=results[(3, 'ACH_solution', 'DTLZ1')]



c=['b','g','r','c','m','y','k']

size=40
name=["ASF","RNSGAII"]
view=[(-108,-50),(-110,-45)]
for i,r in enumerate([asf[3],rnsga[0]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #    ax.set_zlim(0,1)
    #    ax.set_ylim(0,1)
    #ax.set_xlim(0,1)
    ax.view_init(30,-110)
    p0=np.array(r[2][0])
    p=np.array(r[2][1])
    ax.scatter(p[0,0],p[0,1],p[0,2], c=c[i], marker='D',s=size)
    ax.scatter(p[-1,0],p[-1,1],p[-1,2] , marker='s',s=size,c=c[i])
    ax.scatter(p[1:-1,0],p[1:-1,1],p[1:-1,2], c=c[i], marker='x',s=size)#,facecolors='none')
    ax.scatter(r[0][0],r[0][1],r[0][2], c=c[i], marker='o',s=size)
    #ax.scatter(p0[:,0],p0[:,1],p0[:,2],c='y')

    ax.plot(p[:,0],p[:,1],p[:,2],c=c[i])



    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')

    xl=ax.get_xlim()
    ax.set_xlim(xl[0],1)

    zl=ax.get_zlim()
    ax.set_zlim(0,zl[1])
    ax.view_init(*view[i])
    #   fig.show()
    fig.savefig(name[i]+".png",dpi=800,transparent=True,format="png",bbox_inches="tight")
