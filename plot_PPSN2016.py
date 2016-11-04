# coding: utf-8
""" Export png graphics for PPSN2016 article from the results_2016-04-16.dmp file  
"""

import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

def plot(results,filename=None,problem=None,view=(30,-110),alpha=1.0,nzl=None,cmap="gnuplot",legend=False):

    
    c=plt.get_cmap(cmap)(np.linspace(0, 1, len(results)))
    
    size=40

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i,r in enumerate(results):
        asp=r[1][0]
        ref=np.array(r[2][1])
        obj=np.array(r[2][0])
        
        # Initial aspiration level
        aspir_mark=ax.scatter(*tuple(asp), c=c[i], marker='D',s=size,label="Aspiration level")
        
        # Reference points as path and x
        ax.plot(ref[1:,0],ref[1:,1],ref[1:,2], c=c[i])
        ref_mark=ax.scatter(ref[1:,0],ref[1:,1],ref[1:,2], c=c[i],marker='x',label="Reference point")

        # Last solution as square box
        solution_mark=ax.scatter(obj[:,0][-1] ,obj[:,1][-1],obj[:,2][-1], c=c[i], marker='s',s=size,label="Final solution")
        
        # Path from last reference points to final solution
        ax.plot(*tuple(np.vstack((ref[-1],obj[-1])).T),c=c[i])

    for set_ticks in [ax.set_xticks,ax.set_yticks,ax.set_zticks]:
        set_ticks([0.0,0.5,1.0])

    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.set_zlabel('$f_3$')

    xl=ax.get_xlim()
    #ax.set_xlim(xl[0],1)

    yl=ax.get_zlim()
    ax.set_ylim(0,yl[1])

    zl=ax.get_zlim()
    #ax.set_zlim(0,zl[1])

    if problem:
        po=np.rot90(np.loadtxt("%s.3D.pf"%problem))
        ax.plot_trisurf(*list(po),alpha = alpha, color = 'grey',linewidth=0)
        #ax.scatter(*list(po), c=c[-1],s=size)
    ax.set_zlim(zl)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    
    if view:
        ax.view_init(*view)

    if legend:
        plt.legend(loc=0, scatterpoints = 1,handles=[aspir_mark,solution_mark,ref_mark],bbox_to_anchor=(.90, .92))
        leg = ax.get_legend()
        for h in leg.legendHandles:
            h.set_color('grey')

        if filename:
            filename="%s-legend"%filename

    if filename:
        fig.savefig(filename+".png",dpi=600,transparent=True,format="png",bbox_inches="tight")
    else:
        fig.show()
    return fig

if __name__=='__main__':
    if not len(sys.argv) >1:
        print('Using default dmp file')
        dmpf='results_2016-04-16.dmp'
    else:
        dmpf=sys.argv[1]
        
    results=pickle.load(open(dmpf,"r"))
    #rnsga=results[(3, 'rNSGAII_solution', 'DTLZ2')][0]
    #asf=results[(3, 'ACH_solution', 'DTLZ2')][3]
    #plot([asf],filename="ASF",view=(-108,-50))
    #plot([rnsga],filename="RNSGAII",view=(-110,-45))
    plot(results[(3, 'ACH_solution', 'DTLZ4')],problem="DTLZ4",alpha=0.5,filename="ASF",legend=True)
    plot(results[(3, 'rNSGAII_solution', 'DTLZ4')],problem="DTLZ4",alpha=0.5,filename="rNSGA2",legend=True)
    #plot([rnsga,asf],view=(-110,-45),problem="DTLZ2")
    #raw_input(">")
    