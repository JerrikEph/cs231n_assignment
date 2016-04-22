import numpy as np

rate = 1e-5

p = np.random.rand(6)
p = p * (p > 0)
p /= np.sum(p)

def loss(P):

    E = 0
    for i in range(6):
        if(P[i] > 0):
            E += P[i]**2/np.sum(P[i:6])

    return E

def grad(P):

    lossP = loss(P)
    error = np.zeros(6)
    for i in range(6):
        q = P.copy()
        if(q[i] > rate):
            q[i] -= rate
            error[i] = lossP - loss(q)

    maxerrorIdx = np.argmax(error)
    tmp = P.copy()
    tmp[maxerrorIdx] -= rate
    lossQ = loss(tmp)
    for i in range(6):
        q = tmp.copy()
        if(q[i] < 1-rate):
            q[i] += rate
            error[i] = loss(q) - lossQ
    minerrorIdx = np.argmin(error)

    ret = np.zeros(6)
    ret[minerrorIdx] += rate
    ret[maxerrorIdx] -= rate
    # print maxerrorIdx
    return ret

def optimize(P, iter=100):
    for i in range(iter):

        P += grad(P)
        # P += rate
        P = P * (P > 0)
        P /= np.sum(P)
        # if (np.sum(P) > 1):
        #     P[np.argmax(P)] -= (np.sum(P) - 1)
        # else:
        #     P[np.argmin(P)] += (1 - np.sum(P))

        if i%5000 == 0:
            print P, 1./P
            print "loss", loss(P)

optimize(p, 100000)

#[ 0.22491714  0.20018057  0.17516616  0.14990164  0.12491391  0.12492058]
#[ 0.22491525  0.20018424  0.17516806  0.14989671  0.12492081  0.12491493]
#[ 0.22491559  0.20017612  0.175166    0.14990557  0.12491978  0.12491695]
#[ 0.2249183   0.20018087  0.17516517  0.14990106  0.12491738  0.12491721] 0.399248666999
#[ 0.22491901  0.20018004  0.17516525  0.14990056  0.12491746  0.12491768] 0.399248666999
#[ 0.22491292  0.20017973  0.17516101  0.14990546  0.12492074  0.12492014] 0.399248667102