##Implimenting k-means with functional approach
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm

def make_empty_df(Num_obs,Num_centroids):
    cnames = []
    for i in range(Num_centroids):

        name = 'C'+ str(i)
        cnames.append(name)
    d = pd.DataFrame(np.zeros((Num_obs, Num_centroids)))
    d.columns = cnames
    return d,cnames

def plot_Scatter(x,y,c1,c2,ax):
    '''
    INPUTS:
    x and y are two col vectors that have all the data Points
    c1,c2 are the col vectors of the centers (same col as x and y)
    ax is the ax that is set up for the plot
    ***Pass something like ax for ax
    example:
    _,ax = plt.subplots(1,1,figsize = (10,10))
    '''
    ax.scatter(x,y,s = 10,label='All Points')
    ax.scatter(c1,c2,s = 400,label = 'Initial Centroids')
    ax.legend()
    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.grid(alpha = .4,linestyle = '--',c='r')
    return ax


def find_closest(data,centroids):
    num_centroids = centroids.shape[0]
    num_obs = data.shape[0]
    closest_matrix,cnames = make_empty_df(num_obs,num_centroids)

    for j,name in enumerate(cnames):
        center_data = centroids[j,:]
        for i in range(data.shape[0]):
            dist = (np.sqrt(sum(data[i,:]-center_data)**2))
            closest_matrix.loc[i,name] =  dist

    closest = []
    for i in range(closest_matrix.shape[0]):
        distances = closest_matrix.loc[i,:].values

        closest.append(np.argmin(distances))
    return closest


def update_centroids(data, closest, centroids):
    num_centroids = centroids.shape[0]
    for i in range(num_centroids):
        cond = np.array(closest) == i

        avg = (np.mean(data[cond,:],axis = 0))
        centroids[i,:] = avg
    return centroids

def plot_updates(data,centroids,closest,color,ax,count):
    num_centroids = centroids.shape[0]
    for i, c in zip(range(num_centroids),color):
        cond = np.array(closest) == i
        d1 = data[cond,1]
        d2 = data[cond,2]
        ax.scatter(d1,d2,color = c,alpha = 1,label = 'center'+str(i))
        ax.scatter(centroids[i,1],centroids[i,2],s = 300,c = c)
        ax.set_title('iteration #'+str(3*count))

    ax.legend()
    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.grid(alpha = .4,linestyle = '--',c='r')
    return ax
