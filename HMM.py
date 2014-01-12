#By Mohit Minhas

import math
import numpy
#from sklearn.hmm import MultinomialHMM
#from hmmn import *
from hmmpy import *
from sklearn.cluster import k_means
#from scipy.cluster.vq import kmeans2


def get_xyz_data(path,name):
    xfl=path+'\\'+name+'_x.csv'
    xx = numpy.genfromtxt(xfl, delimiter=',')
    yfl=path+'\\'+name+'_y.csv'
    xy = numpy.genfromtxt(yfl, delimiter=',')
    zfl=path+'\\'+name+'_z.csv'
    xz = numpy.genfromtxt(zfl, delimiter=',')
    x=[]
    x.append(xx)
    x.append(xy)
    x.append(xz)
    x=numpy.array(x)
    return x

"""
def emprob(M,N):
    a=1/float(N)
    E=numpy.zeros((M,N))
    for i in xrange(M):
        for j in xrange(N):
            E[i][j]=a
    return E
"""

def prior_transition_matrix(K,LR):
    P = numpy.multiply(1/float(LR),numpy.identity(K+1))
    w=1/float(LR)
    for i in xrange(1,K-(LR-1)+1): 
        for j in xrange(1,LR-1+1):
            P[i][i+j]=w
    for i in xrange(K-(LR-2),K+1):
        for j in xrange(1,K-i+1+1):
            P[i][i+(j-1)] = 1/float(K-i+1)
    P=P[1:,1:]
    return P

def get_point_centroids(indata,K,D):
    mean = numpy.zeros((indata.shape[1],D))
    for n in xrange(0,(indata.shape[1])):
        for i in xrange(0,(indata.shape[2])):
            for j in xrange(0,D):
                mean[n][j] = mean[n][j] + indata[j][n][i]
        mean[n] = mean[n]/(indata.shape[2])
    (centroids,x,y)=k_means(mean,K) #random order. change n_jobs to speed up
    return centroids

def get_point_clusters(data,centroids,D):
    XClustered = [[] for x in xrange(data.shape[2])]
    K = centroids.shape[0]
    for n in xrange(0,(data.shape[1])):
        for i in xrange(0,(data.shape[2])):
            temp = numpy.zeros((K,1))
            for j in xrange(0,K):
                #if (D==3)
                temp[j] = math.sqrt(math.pow((centroids[j][0] - data[0][n][i]),2)+math.pow((centroids[j][1] - data[1][n][i]),2)+math.pow((centroids[j][2] - data[2][n][i]),2));
            I = numpy.argmin(temp)
            XClustered[i].append(I)
    XClustered=numpy.array(XClustered)
    return XClustered


def pr_hmm(o,a,b,pi):
    n=len(a[0])
    T=len(o)
    for i in xrange(1,n+1):
        m[1][i]=b[i][o[1]]*pi[i];
    for t in xrange(1,(T-1)+1):
        for j in xrange(1,n+1):
            z=0
            for i in xrange(1,n+1):
                z=z+a[i][j]*m[t][i]
            m[t+1][j]=z*b[j][o[t+1]]
    p=0
    for i in xrange(1,n+1):
        p=p+m[T][i]        
    p=math.log(p)
    return p


D=3
M=12
N=8
LR=2

train_gesture='x'
test_gesture='x'

gestureRecThreshold = 0

training = get_xyz_data('data/train',train_gesture)
testing = get_xyz_data('data/test',test_gesture)

centroids = get_point_centroids(training,N,D)
ATrainBinned = get_point_clusters(training,centroids,D)
ATestBinned = get_point_clusters(testing,centroids,D)

pP = prior_transition_matrix(M,LR)


#W=emprob(M,N)
#print ATrainBinned

#model=MultinomialHMM(n_components=M,startprob_prior=pP,n_iter=50)
#model.n_symbols=N
#print model.n_symbols
#model.fit(ATrainBinned)
#model=MultinomialHMM(ATrainBinned,pP,[1:N]',M,cyc,.00001) #ENTER 
#logprob=model.score(ATestBinned)
#print logprob

hmm=HMM(n_states=M,V=[0,1,2,3,4,5,6,7],A=pP)
print 'TRAINING'
print
baum_welch(hmm,ATrainBinned,graph=False,verbose=True)
print 
print 'TESTING'
print
b=forward(hmm,ATestBinned[0])
print b

#model=DiscreteHmm(numstates=M,numclasses=N)
#model.learn(ATrainBinned,numsteps=ATrainBinned.shape[0])

"""
sumLik = 0.0
minLik = float('inf')
for j in xrange(0,len(ATrainBinned)):
	lik = pr_hmm(ATrainBinned[j],P,E.T,Pi)
	if (lik < minLik):
		minLik = lik
	sumLik = sumLik + lik
	
gestureRecThreshold = 2.0*sumLik/len(ATrainBinned)

print'\n\n********************************************************************\n'
print'Testing {0) sequences for a log likelihood greater than {1)\n'.format(len(ATestBinned),gestureRecThreshold)
print'********************************************************************\n\n'

recs = 0
tLL = numpy.zeros((len(ATestBinned),1))
for j in xrange(1,len(ATestBinned)):
	tLL[j][1] = pr_hmm(ATestBinned[j],P,E.T,Pi)
	if (tLL[j][1] > gestureRecThreshold):
		recs = recs + 1
		print 'Log likelihood: {0) > {1) (threshold) -- FOUND {2) GESTURE!\n'.format(tLL[j][1],gestureRecThreshold,test_gesture)
	else:
		print 'Log likelihood: {0} < {1} (threshold) -- NO {2} GESTURE.\n'.format(tLL[j][1],gestureRecThreshold,test_gesture)
                          
print'Recognition success rate: {0) percent\n'.format(100*recs/len(ATestBinned))
"""
