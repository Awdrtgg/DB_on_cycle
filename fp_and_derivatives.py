import numpy as np
from scipy.sparse import dok_matrix, identity
from scipy.sparse.linalg import inv

def state_to_tuple(p, q, pq, i):
    x, a, b = 1, 1, 0

    # get current weight
    w = 0
    while (i >= pq[w] and w < len(pq)):
        i -= pq[w]
        w += 1

    # get current column
    while (i >= q[w]):
        i -= q[w]
        a += 1

    b = w - a
    x = i + 1

    return x, a, b

def tuple_to_state(p, q, pq, x, a, b):
    i = 0
    i += sum(pq[0 : a + b])
    i += (a - 1) * q[a + b]
    i += x - 1
    return i

def tuple_change(n, p, q, pq, x, a, b):
    if (a > b):
        # make sure that a <= b
        t = a
        a = b
        b = t

    if (n - a - b - x < x):
        # make sure that x is the least distance of 2 groups
        x = n - a - b - x

    if (x == 0):
        # if this goes to a final state (a sequence of m), then return -(m+1)
        # the +1 is for the result of -0
        return -(a + b + 1)

    if (a == 0):
        return -(b + 1)

    return tuple_to_state(p, q, pq, x, a, b)


def theo_phase_1(n, r):
    # Algorithm 3 
    # Calculation of Q1 and Q2

    p = np.array([int(i/2) for i in range(0, n+1)])
    q = np.array([int((n-i)/2) for i in range(0, n+1)])
    pq = np.multiply(p, q)
    N = np.sum(pq)
    print(N)

    Q1 = np.zeros((N, N), dtype=float)
    Q1_T = np.zeros((N, N), dtype=float)
    Q1_TT = np.zeros((N, N), dtype=float)
    Q2 = np.zeros((N, n-1), dtype=float)

    for i in range(0, N):
        x, a, b = state_to_tuple(p, q, pq, i)
        prob_self = 1
        prob_self_d = 0 # first order derivative
        prob_self_dd = 0 # second order derivative

        ''' #4 & #1 '''
        t = tuple_change(n, p, q, pq, x, a+1, b)
        if (n - a - b - x == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * r / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] += 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d -= 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            
        ''' #5 & #1 '''
        t = tuple_change(n, p, q, pq, x, a, b+1)
        if (n - a - b - x == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * r / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] += 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d -= 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            
        ''' #6 & #1 '''
        t = tuple_change(n, p, q, pq, x-1, a+1, b)
        if (x == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * r / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] += 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d -= 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            
        ''' #7 & #1 '''
        t = tuple_change(n, p, q, pq, x-1, a, b+1)
        if (x == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * r / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] += 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d -= 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)

        ''' #8 & #2 '''
        t = tuple_change(n, p, q, pq, x, a-1, b)
        if (a == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * 1 / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] -= 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d += 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            
        ''' #9 & #2 '''
        t = tuple_change(n, p, q, pq, x+1, a-1, b)
        if (a == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * 1 / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] -= 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d += 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)

        ''' #10 & #3 '''
        t = tuple_change(n, p, q, pq, x, a, b-1)
        if (b == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * 1 / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] -= 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d += 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)

        ''' #11 & #3 '''
        t = tuple_change(n, p, q, pq, x+1, a, b-1)
        if (b == 1): prob = 1 / float(n) * 0.5
        else:        prob = 1 / float(n) * 1 / (1. + r)
        prob_self -= prob
        if (t < 0): Q2[i, -(t+2)] += prob
        else:
            Q1[i, t] += prob
            Q1_T[i, t] -= 1 / float(n) / (1. + r) / (1. + r)
            prob_self_d += 1 / float(n) / (1. + r) / (1. + r)
            Q1_TT[i, t] += 2 / float(n) / (1. + r) / (1. + r) / (1. + r)
            prob_self_dd -= 2 / float(n) / (1. + r) / (1. + r) / (1. + r)

        Q1[i, i] += prob_self
        Q1_T[i, i] += prob_self_d
        Q1_TT[i, i] += prob_self_dd

    # inv
    E = np.eye(N, dtype=float)
    T = E - Q1
    Qn = np.linalg.inv(T)

    # R01
    R01 = np.dot(Qn, Q2)

    # R1X
    R11 = np.dot(Qn, Q1_T)
    R11 = np.dot(R11, Qn)
    R11 = np.dot(R11, Q2)
    R12 = R01

    # R2X
    R21 = np.dot(Qn, Q1_T)
    R21 = np.dot(R21, Qn)
    R21 = np.dot(R21, Q1_T)
    R21 = np.dot(R21, Qn)
    R21 = np.dot(R21, Q2)

    R22 = np.dot(Qn, Q1_TT)
    R22 = np.dot(R22, Qn)
    R22 = np.dot(R22, Q2)
    R23 = R11
    R24 = R12

    return R01, R11, R12, R21, R22, R23, R24

def fp_and_derivative(n, r):
    # theo_phase_1
    R01, R11, R12, R21, R22, R23, R24 = theo_phase_1(n, r)
    
    # theo_phase_2
    y = [1., (r+1.)/(2*r)]
    for i in range(1, n-2):
        y.append(y[i] * 1/r)
    y.append(y[n-2] * 2/(1+r))

    P = np.matrix([(sum(y[0:i]))/(sum(y)) for i in range(1, n)])
    P = np.transpose(P)

    yy = [0.]
    for i in range(1, n-1):
        yy.append( (- (i-1) / (r**i) - i / r**(i+1)) / 2 )
    yy.append(-(n-2)/(r**(n-1)))
    P1 = np.matrix([( sum(yy[0:i]) * sum(y) - sum(y[0:i]) * sum(yy) ) / ( (sum(y)) * (sum(y)) ) for i in range(1, n)])
    P1 = np.transpose(P1)

    yyy = [0.]
    for i in range(1, n-1):
        yyy.append( ( i*(i-1) / (r**(i+1)) + i*(i+1) / r**(i+2) ) / 2 )
    yyy.append((n-2)*(n-1)/(r**n))
    P2 = np.matrix(
        [
            ( sum(yyy[0:i])*sum(y)*sum(y) - sum(y[0:i])*sum(yyy)*sum(y) - 2*sum(yy[0:i])*sum(y)*sum(yy) + 2*sum(y[0:i])*sum(yy)*sum(yy) )
            / ( sum(y)*sum(y)*sum(y) )
            for i in range(1, n)
        ])
    P2 = np.transpose(P2)

    # output results
    filename = 'result-%d.txt' % n
    w = open(filename, 'w+')

    Result = R01 * P
    print(P[1])
    print(Result[0:int((n-2)/2)])
    w.write('The fixation probabilities: \n')
    w.write(str(P[1]))
    w.write('\n')
    w.write(str(Result[0:int((n-2)/2)]))
    w.write('\n\n')
    print()

    Result1 = R11 * P + R12 * P1
    print(P1[1])
    print(Result1[0:int((n-2)/2)])
    w.write('The first-order derivatives of fixation probabilities: \n')
    w.write(str(P1[1]))
    w.write('\n')
    w.write(str(Result1[0:int((n-2)/2)]))
    w.write('\n\n')
    print()

    Result2 = (2*R21*P + R22*P + 2*R23*P1 + R24*P2)/2
    print(P2[1]/2)
    print(Result2[0:int((n-2)/2)])
    w.write('The second-order derivatives of fixation probabilities: \n')
    w.write(str(P2[1]))
    w.write('\n')
    w.write(str(Result2[0:int((n-2)/2)]))

    return P[1], Result1[0:int((n-2)/2)], Result2[0:int((n-2)/2)]

if __name__ == "__main__":
    for i in [6, 25, 50, 60, 70, 80, 90, 100]:
        fp_and_derivative(i, 1.0)
    