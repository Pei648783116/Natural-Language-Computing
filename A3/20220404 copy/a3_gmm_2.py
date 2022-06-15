from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))

    def __str__(self):
        return ("name: " + str(self.name) +
                " omega: " + str(self.omega) +
                " mu: " + str(self.mu) +
                " Sigma: " + str(self.Sigma))

    def PreCompute(self):
        # Returns array of 'lowerLog' values for log_b_m_x for each m.
        M, d = self.mu.shape
        lowerLog = np.zeros(M)
        for m in range(M):
            sigmaProduct = np.prod(self.Sigma[m, :])
            lowerLog[m] = np.log((np.pi * 2) ** (d / 2) * (sigmaProduct ** 0.5))
        return lowerLog
        

def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    # print ( 'TODO' )

    d = len(x)
    upperSum = 0.0
    for n in range(d):
        upperSum += (x[n] - myTheta.mu[m, n]) ** 2 / myTheta.Sigma[m, n]
    if len(preComputedForM) > 0:
        # sigmaProduct = preComputedForM[0]
        lowerLog = preComputedForM[m]
    else:
        sigmaProduct = np.prod(myTheta.Sigma[m, :])
        lowerLog = np.log((np.pi * 2) ** (d / 2) * (sigmaProduct ** 0.5))
    # Avoid divsion by zero using logs
    upperLog = -0.5 * upperSum
    # lowerLog = np.log((np.pi * 2) ** (d / 2) * (sigmaProduct ** 0.5))
    log_b_m_x = upperLog - lowerLog
    #print("log_b_m_x")
    #print(m)
    #print(x)
    #print(myTheta)
    #print(preComputedForM)
    #print(upperSum)
    #print(upperLog)
    #print(sigmaProduct)
    #print(lowerLog)
    #print(log_b_m_x)
    return log_b_m_x

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    # print ( 'TODO' )
    d = len(x)
    M = myTheta.omega.shape[0]
    # preComp = np.prod(myTheta.Sigma[m, :])
    preComp = myTheta.PreCompute()
    upperLog = np.log(myTheta.omega[m, 0]) + log_b_m_x(m, x, myTheta, preComp)
    lower = []
    for k in range(M):
        #print(myTheta.omega[k, 0])
        #print(log_b_m_x(k, x, myTheta, [sigmaProduct]))
        #lower += myTheta.omega[k, 0] * np.exp(log_b_m_x(k, x, myTheta, [sigmaProduct]))
        lower.append(log_b_m_x(k, x, myTheta, preComp) + np.log(myTheta.omega[k, 0]))
    #print("log_p_m_x")
    #print(upperLog)
    #print(logsumexp(lower))
    log_p_m_x = upperLog - logsumexp(lower)
    return log_p_m_x

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    # print( 'TODO' )
    #print("logLik")
    M, T = log_Bs.shape
    loglik = []
    for t in range(T):
        #p = 0.0
        log_p = []
        for m in range(M):
            #p += myTheta.omega[m] * np.exp(log_Bs[m, t])
            log_p.append(log_Bs[m, t] + np.log(myTheta.omega[m, 0]))
        #logP += np.log(p)
        loglik.append(logsumexp(log_p))
    #print("logLik: " + str(logP))
    return sum(loglik)


def UpdateParameters(myTheta, X, log_Ps):
    ''' Updates myTheta for new iteration.
    '''
    M, T = log_Ps.shape
    d = X.shape[1]
    #print(M)
    #print(T)
    #print(d)
    #print(log_Bs)
    #print(log_Ps)
    for m in range(M):
        # sum_p = np.exp(logsumexp(log_Ps[m, :]))
        # sum_p_x = np.exp(logsumexp(np.log(X[T]) + log_Ps[m, :]))
        # sum_p_x2 = np.exp(logsumexp(2 * np.log(X[T]) + log_Ps[m, :]))
        sum_p = 0.0
        sum_p_x = np.zeros((d))
        # The following vector represents a diagonal square matrix.
        sum_p_x2 = np.zeros((d))
        for t in range(T):
            p = np.exp(log_Ps[m, t])
            sum_p += p
            sum_p_x += p * X[t]
            sum_p_x2 += p * np.square(X[t])
        myTheta.omega[m] = sum_p / T
        myTheta.mu[m, :] = sum_p_x / sum_p
        myTheta.Sigma[m, :] = (sum_p_x2 / sum_p) - np.square(myTheta.mu[m, :])

    return myTheta


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    # print ('TODO')

    # Set some reasonable initial values
    T = X.shape[0]
    d = myTheta.omega.shape[1]
    index = np.random.randint(0, high=T, size=M)
    
    # Initialize mu to M random elements of X
    for n, i in enumerate(index):
        myTheta.mu[n, :] = X[i, :]
    # Initialize omega to random values that sum to 1
    myTheta.omega = np.random.rand(M, d)
    myTheta.omega = myTheta.omega / np.sum(myTheta.omega)
    # Initialize sigma to 1/M
    myTheta.Sigma.fill(M ** -1)
    #print(myTheta)
    i = 0
    prev_L = np.NINF
    improvement = np.Inf
    while i <= maxIter and improvement >= epsilon:
        print("Train: iter " + str(i) + " with M=" + str(M) + " T=" + str(T))
        # ComputeIntermediateResults
        log_Bs = np.zeros((M, T))
        log_Ps = np.zeros((M, T))
        for m in range(M):
            print("m=" + str(m))
            # sigmaProduct = np.prod(myTheta.Sigma[m, :])
            preComp = myTheta.PreCompute()
            for t in range(T):
                log_Bs[m, t] = log_b_m_x(m, X[t], myTheta, preComp)
                log_Ps[m, t] = log_p_m_x(m, X[t], myTheta)
        # L = ComputeLikelihood(X, myTheta)
        L = logLik(log_Bs, myTheta)
        # myTheta = UpdateParameters(myTheta, X, L)
        myTheta = UpdateParameters(myTheta, X, log_Ps)
        improvement = L - prev_L
        print("logL=" + str(L) + " (improved by " + str(improvement) + ")")
        prev_L = L
        i += 1
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    # print ('TODO')
    T, D = mfcc.shape

    top_k = []
    for model in models:
        M = model.omega.shape[0]
        log_Bs = np.zeros((M, T))
        preComp = model.PreCompute()
        for m in range(M):
            # sigmaProduct = np.prod(model.Sigma[m, :])
            for t in range(T):
                log_Bs[m, t] = log_b_m_x(m, mfcc[t], model, preComp)
        top_k.append({'logp': logLik(log_Bs, model), 'name': model.name})
    top_k.sort(key=lambda x: x['logp'], reverse=True)
    bestModel = top_k[0]['name']
    # print('\n'+str(bestModel))
    # print(str(correctID))
    correctName = models[correctID].name
    print(models[correctID].name)
    for i in range(k):
        print(top_k[i]['name'] + " " + str(top_k[i]['logp']))
    return 1 if (bestModel == correctName) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8 # 8
    epsilon = 1.0
    maxIter = 20 # 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)
                # divisor = 20
                # X = X[:X.shape[0]//divisor,:]
            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(str(accuracy) + " accuracy")
