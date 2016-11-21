# coding: utf-8
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial import Rectangle
from sklearn.manifold import MDS

import sys,argparse

MOEA_PROBLEMS=["Belegundu","Binh","Binh2","Binh3","Binh4","CF1","CF10","CF2","CF3","CF4","CF5","CF6","CF7","CF8","CF9","DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ7","Fonseca","Fonseca2","Jimenez","Kita","Kursawe","LZ09_F1","LZ09_F2","LZ09_F3","LZ09_F4","LZ09_F5","LZ09_F6","LZ09_F7","LZ09_F8","LZ09_F9","Laumanns","Lis","Murata","OKA1","OKA2","Obayashi","Osyczka","Osyczka2","Poloni","Quagliarella","R2_DTLZ2_M5","R3_DTLZ3_M5","Rendon","Rendon2","Schaffer","Schaffer2","Srinivas","Tamaki","Tanaka","UF1","UF10","UF2","UF3","UF4","UF5","UF6","UF7","UF8","UF9","Viennet","Viennet2","Viennet3","Viennet4","WFG1","WFG1_M5","WFG2","WFG3","WFG4","WFG5","WFG6","WFG7","WFG8","WFG9","ZDT1","ZDT2","ZDT3","ZDT4","ZDT6",]


import pyMOEA
def rec_dim(r):
    xy=r.mins
    w=r.maxes[0]-r.mins[0]
    h=r.maxes[1]-r.mins[1]
    return xy,w,h

def plot_res(res,problem='DTLZ2'):
    po=np.rot90(np.loadtxt("data/%s.2D.pf"%problem))

    objs=np.swapaxes(np.array(res[3][0]),0,1)
    for j,rectangles in enumerate(res[1][:-1]):
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        ax2.scatter(*res[3][1][j],c="r") # reference points
        ax2.scatter(*res[3][0][j],marker="d",c="r",zorder=5)
        ax2.scatter(objs[0][:j+1],objs[1][:j+1],marker="d",zorder=2)
        
        ax2.scatter(*list(po),alpha = 1, color = 'grey',linewidth=0,zorder=-1)
        for i,r in enumerate(rectangles):
            ax2.add_patch(patches.Rectangle(*rec_dim(r),fill=False,zorder=-2))
    plt.show()

def plot_mds(ax, values,x_limit=None,y_limit=None,**kwargs):
    """ Create MDS projection subplot of the result generated by the ADM2 

    :param matplotlib.plot.axes ax: axes to plotted on
    :param list(float) values: Values to be plotted (either referece or PO point)
    :param tuple(float,float) x_limit: Limits for x-axis (Use data if None)    
    :param tuple(float,float) y_limit: Limits for y-axis (Use data if None)    
    :param **kwargs: Arguments passed to matplotlib
    :return: Modifed ax
    :rtype matplotlib.pyplot.axes
    """
    def my_mds(m):
        return MDS(n_components=2).fit(m.astype(np.float64)).embedding_
    mds=my_mds(np.array(values))
    data=np.swapaxes(mds,0,1)
    if x_limit:
        ax.set_xlim(x_limit)
    if y_limit:
        ax.set_ylim(y_limit)
        
    # Path
    ax.plot(*data,**kwargs)
    
    # First item
    ax.scatter(data[0][0],data[1][0],zorder=4,marker='x',color="red",s=5)
    ax.scatter(*data,**kwargs)

    return ax

def show_mds(res,problem,save_file=False):
    """ Create MDS projection subplots and show them in a 2x2 grid, i.e, with maximum of 4 plots

    :param list(results) res: List of results generate by get_res
    :param str problem: Name of the problem,needed for outputs
    :param bool save_file: If true, create png of the visualization (default: False)
        
    """    
    title="%s %ik"%(problem,len(res[0][3][0][0]))
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    fig.suptitle(title)
    
    k=lambda i,j:plot_mds(axes[i][j],res[i*(j+1)+(j)][3][0]);
    try:
        [[k(i,j) for i in range(2)]for j in range(2)]
    except IndexError:
        pass
    if save_file:
        fig.savefig("%s.png"%title,dpi=200)
    plt.show()


def plot_contour(k,c,r):
    """  Does not work
    """
    xlist = np.linspace(0,1,10)
    ylist = np.linspace(0,10,20)
    
    X,Y = np.meshgrid(xlist,ylist)
    zlist = []
    
    for x in xlist:
    
        for y in ylist:
            
            z = pyMOEA.utility_function(0,k)
            zlist.append(z)  
    plt.figure()        
    plt.contour(X,Y,zlist)

    plt.show()  
    

def get_res(nf=2,c=None,r=.5,problem='DTLZ2',uf_n=None,**kwargs):
    problem_type=pyMOEA.problem(problem, nf)
    try:
        ideal=problem_type.ideal
        nadir=problem_type.nadir
    except AttributeError:
        ideal=[0.0]*nf
        nadir=[1.0]*nf

    init_pref=[[Rectangle(ideal,nadir)]]
    
    return pyMOEA.ADM2_solve(pyMOEA.ACH_solution,problem,init_pref,**kwargs)#,c=-np.inf)

if __name__=='__main__':
    parser= argparse.ArgumentParser(description="Test ADM2 Agent",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p','--problem', type=str,
                    help='Problem to be solved',default='External')

    parser.add_argument('-k', '--objectives', type=int,
                        help='Number of objectives',default=3)

    parser.add_argument('-n', '--runs', type=int,
                        help='Number of runs',default=1)

    parser.add_argument('-l','--list-problems',action='store_true',
                        help="List available problems")
    
    parser.add_argument('-v', '--verbose', type=int,
                        help='Logging level',default=logging.INFO)

    args=parser.parse_args()    
    verbose= args.verbose
    
    if args.list_problems:
        print "Available problems"
        print "External"
        for p in MOEA_PROBLEMS:
            print p
        sys.exit(0)
    logging.basicConfig(level=verbose)

    res=[]
    for i in range(args.runs):
        res.append(get_res(nf=args.objectives,problem=args.problem))
    if args.objectives==2:
        # Show only last run here
        plot_res(res[-1])
    else:
        show_mds(res,problem=args.problem)    
