# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial import Rectangle
import pyMOEA
def rec_dim(r):
    xy=r.mins
    w=r.maxes[0]-r.mins[0]
    h=r.maxes[1]-r.mins[1]
    return xy,w,h

def plot_res(res):

    c=plt.get_cmap("Accent")(np.linspace(0, 1,  len(res[1][-1])+1))
    #c=plt.get_cmap("Accent")(np.linspace(0, 1, len(res[(2, 'ACH_solution', 'DTLZ2')][0][1])))
    #c=plt.get_cmap("Accent")(np.linspace(0, 1, len(res[1])))
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, aspect='equal')
        
    #ax2.scatter(*np.rot90(res[3][1]))
    #ax2.scatter(*np.rot90(res[3][0]),marker="x")
    
    problem="DTLZ2"
    po=np.rot90(np.loadtxt("%s.2D.pf"%problem))
    #ax2.plot(*list(po),alpha = 1, color = 'grey',linewidth=0)
    #ax2.scatter(*list(po),alpha = 1, color = 'grey',linewidth=0,zorder=-1)
    
    objs=np.swapaxes(np.array(res[3][0]),0,1)
    for j,rectangles in enumerate(res[1][:4]):
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        ax2.scatter(*res[3][1][j],c="r") # reference points
        ax2.scatter(*res[3][0][j],marker="d",c="r",zorder=5)
        ax2.scatter(objs[0][:j+1],objs[1][:j+1],marker="d",zorder=2)
        
        ax2.scatter(*list(po),alpha = 1, color = 'grey',linewidth=0,zorder=-1)
        for i,r in enumerate(rectangles):
            #ax2.add_patch(patches.Rectangle(*rec_dim(r),color=c[i],alpha=.75,zorder=-2))
            ax2.add_patch(patches.Rectangle(*rec_dim(r),fill=False,zorder=-2))
        plt.show()

def get_res(nf=2):
    ideal=[0.0]*nf
    nadir=[1.0]*nf
    init_pref=[[Rectangle(ideal,nadir)]]
    return pyMOEA.ADM2_solve(pyMOEA.ACH_solution,'DTLZ2',init_pref)

if __name__=='__main__':
    import shelve
    #db=shelve.open("res.db")
    #res=db["res"]
    #db.close()
    res=get_res(nf=2)
    plot_res(res)
    
