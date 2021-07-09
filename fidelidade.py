#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
from numba import njit
from scipy.linalg import eigh_tridiagonal

start_time = time.time()

# Definir a desordem correlacionada:
@njit(parallel=True, fastmath=True)
def laco(A,N, z):
    Y = []
    for i in range(1,(N+1)):
        y = 0
        for j in range(1,(N+1)):
            y += z[j-1]/((1+abs(i-j)/A)**(2))
        Y.append(y)
    return Y


def rand(A, N):
    z = np.random.random(N)*2-1
    Y = laco(A, N, z)
    e = [(x - np.average(Y))/np.std(Y) for x in Y] 
    return e


# Definir o hamiltoniano e a diagonalização:   
def eight_sates(A, E, N, g=0.01):

    first_diagonal          = rand(A, N)
    second_diagonal         = np.ones(N-1)
    first_diagonal[0]       = E
    first_diagonal[N-1]     = E
    second_diagonal[0]      = g
    second_diagonal[N-2]    = g
    evals, evecs = eigh_tridiagonal(first_diagonal,second_diagonal)
    return evals, evecs


# Definir o cáculo da fidelidade do sistema:
@njit(parallel=True, fastmath=True)
def laco2(evals,evecs,dt,N):
    f_N = 0
    for c in range(len(evecs)):    
        f_N += (np.e**(-1j*dt*evals[c]))*evecs[0][c]*evecs[N-1][c]
    return abs(f_N)


def fidelidade(evals,evecs,N):
    dt      = (0)
    box     = []
    while dt <= 2*10**6:
        box.append(laco2(evals,evecs,dt,N))
        dt  += 1000
    f_n_tau =  max(box)  
    return (1/2 + (1/3)*f_n_tau + (1/6)*f_n_tau**2)


# Definir o plot fidelidade x energia para um experimento:
def plot(A, N):
    
    fidelity    = []
    energy      = []
    E           = -3
    while E <= 3:    
        evals, evecs = eight_sates(A, E, N)
        f = fidelidade(evals, evecs, N)
        fidelity.append(f)
        energy.append(E)
        E += 0.1
    return energy, fidelity


# Definir o valor médio da fidelidade para x-experimentos:
def n_experiments(A, N, M):
    
    Y = np.zeros((M,60))
    for m in range(M):
        h    = plot(A, N)[1]
        Y[m] = h
        
    x = plot(A, N)[0]
    y = np.mean(Y,axis=0)
    return x, y
        

# Atribuir os valores dos parâmetros:
average = int(input('N# medias: '))
A       = 1
g       = 0.01
N       = int(input('tam cadeia: '))
F       = False

# Salvar os dados energia X fidelidade em um .dat:
    
while A <= 200:
    
    if A == 1:
        x,y = n_experiments(A, N, average)
        df  = pd.DataFrame({"x":x,"y":y})
        df.to_csv("FA%dN%dg%.3fM%d.dat" %(A, N, g, average),
                  index=F,header=F,sep=' ')
        A = 25
    else:
        x,y = n_experiments(A, N, average)
        df  = pd.DataFrame({"x":x,"y":y})
        df.to_csv("FA%dN%dg%.3fM%d.dat" %(A, N, g, average),
                  index=F,header=F,sep=' ')
        A *= 2

a = time.time() - start_time
arq = open('tempof.txt','w')
s = str('o programa da fidelidade foi finalizado em %.2f s' %(a))
arq.write(s+'\n')
arq.close()
