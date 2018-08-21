import numpy as np

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
    #print(Q1)
    #print(Q1_T)
    #Q1_D_inv = np.matrix(Q1_T).I

    # R01
    R0 = np.dot(Qn, Q2)

    return Qn, R0, Q1, Q2

def fixation_time(n, r):
    # theo_phase_1
    Qn, R0, Q1, Q2 = theo_phase_1(n, r)

    # get Q3
    Q3 = np.zeros((n+1, n+1), dtype=float)
    Q3[0, 0] = 1.
    Q3[1, 0] = 1. / float(n)
    Q3[n-1, n] = 1. / float(n)
    Q3[n, n] = 1.
    for i in range(1, n):
        if (i <> 1): Q3[i, i-1] = 2. / float(n) * 1. / (1. + r)
        if (i <> (n-1)): Q3[i, i+1] = 2. / float(n) * r / (1. + r)
        Q3[i, i] = 1. - Q3[i, i-1] - Q3[i, i+1]
    Q3 = Q3[1:n, 1:n]

    # theo_phase_2
    y = [1., (r+1.)/(2.*r)]
    for i in range(1, n-2):
        y.append(y[i] * 1./r)
    y.append(y[n-2] * 2./(1.+r))

    P = np.matrix([(sum(y[0:i]))/(sum(y)) for i in range(1, n)])
    P = np.transpose(P)
    Psi = R0 * P
    #print(P)

    Tplus = []
    for i in range(1, n-1):
        Tplus.append(2./n * r/(1.+r))
    Tplus.append(1./n)

    T = []
    t1 = 0.
    for k in range(1, n):
        for l in range(1, k+1):
            t1 += P[l-1, 0] / Tplus[l-1] * y[k] / y[l]

    T.append(t1)
    for i in range(2, n):
        tt = t1 * P[0, 0] / P[i-1, 0] * sum(y[:i])
        for k in range(1, i):
            tt -= P[k-1, 0] / P[i-1, 0] / Tplus[k-1] * (sum(y[k:i]) / y[k])

        T.append(tt)

    T = np.transpose([T])

    X_F = np.multiply(T, P)
    X_S = np.dot(Q1, Psi) + np.dot(Q2, P) + np.dot(Q2, X_F)
    X_S = np.dot(Qn, X_S)
    X_S = np.divide(X_S, Psi)


    # output results
    filename = 'fixation-time-%d-%.1f.txt' % (n, r)
    w = open(filename, 'w+')

    
    print(T[1])
    print(X_S[0:int((n-2)/2)])
    w.write('The fixation times: \n')
    w.write(str(T[1]))
    w.write('\n')
    w.write(str(X_S[0:int((n-2)/2)]))
    w.write('\n\n')
    print()

    return T[1], X_S[0:int((n-2)/2)]

if __name__ == "__main__":
    for i in [6, 25]:
        fixation_time(i, 1.0)
    
