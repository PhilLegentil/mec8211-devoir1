#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:35 2025

@author: malatchoumymarine
"""

# importation des modules

import numpy as np
import matplotlib.pyplot as plt

# definition des focntion pour ressoudre le problème

def resolution_EDP_ordre_1(N,S,Deff,R,Ce):
    dr = R/(N-1)
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    #conditions frontières
    #à r=R
    A[-1,-1] = 1
    B[-1] = Ce
    
    #à r=0
    A[0,0] = -3
    A[0,1] =  4
    A[0,2] = -1
    B[0] = 0
    
    #domaine
    ri = dr
    for i in range(1,N-1):
        
        A[i,i] = (-2/dr**2-1/(dr*ri))
        A[i,i+1] = (1/dr**2+1/(dr*ri))
        A[i,i-1] = 1/dr**2
        
        B[i] = S/Deff
        
        ri+=dr
        
    C = np.linalg.solve(A,B)
    
    return C

def resolution_EDP_ordre_2(N,S,Deff,R,Ce):
    dr = R/(N-1)
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    #conditions frontières
    #à r=R
    A[-1,-1] = 1
    B[-1] = Ce
    
    #à r=0
    A[0,0] = -3
    A[0,1] =  4
    A[0,2] =  -1
    
    #domaine
    ri = dr
    for i in range(1,N-1):
        
        A[i,i] = -2/dr**2
        A[i,i+1] = (1/dr**2+1/(2*dr*ri))
        A[i,i-1] = 1/dr**2 - 1/(2*ri*dr)
        
        B[i] = S/Deff
        
        ri+=dr
        
    C = np.linalg.solve(A,B)
    
    return C
    
#paramètres du problème
N = 5
S = 2*10**-8
Deff = 10**-10
Ce = 20
R = 0.5

#solution numérique
C = resolution_EDP_ordre_1(N,S,Deff,R,Ce)
dr = R/(N-1)
r = np.linspace(0,R,N)

#solution analytique
ra = np.linspace(0,R,200)
Ca= 1/4*S/Deff*R**2*(ra**2/R**2-1) + Ce

#graph sol analytique et numérique
plt.plot(r, C, "ro", label=f"solution numérique avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()


#graph norme erreur et ordre de convergence
Ne = [5, 25, 100, 150, 300]

L1 = np.zeros(len(Ne))
L2 = np.zeros(len(Ne))
Linf = np.zeros(len(Ne))

DR = np.zeros(len(Ne))

for i in range(len(Ne)):
    
    DR[i] = R/(Ne[i]-1) 
    C = resolution_EDP_ordre_1(Ne[i],S,Deff,R,Ce)
    
    r = np.linspace(0,R,Ne[i])
    Ca = 1/4*S/Deff*R**2*(r**2/R**2-1) + Ce
    
    for k in range(Ne[i]):
        L1[i]+=1/Ne[i]*abs(C[k]-Ca[k])
        L2[i]+=1/Ne[i]*abs(C[k]-Ca[k])**2
    L2[i] = np.sqrt(L2[i])
    Linf[i] =  max(abs(C-Ca))


p_L2 = np.log(L2[-1]/L2[-2])/np.log(DR[-1]/DR[-2])
print(f"l'odre de l'erreur avec la norme L2 est p = {p_L2}")  

    
plt.figure(figsize=(8, 6))  
plt.loglog(DR, L1,'bo', label="Norme L1")
plt.loglog(DR, L2, 'ro', label="Norme L2")
plt.loglog(DR, Linf, 'yo', label="Norme Linf")
plt.xlabel('Taille de maille $Δr$ (m)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_1$, $L_2$ et $L_inf$  (mol/m^3)', fontsize=12, fontweight='bold')
plt.title("Norme des erreurs en fonction de $Δr$ ")
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
plt.grid(True)
plt.legend()
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.show()





#------------partie E (même calcul avecun ordre 2)------------#




#solution numérique
C_2 = resolution_EDP_ordre_2(N,S,Deff,R,Ce)
C_1= resolution_EDP_ordre_1(N,S,Deff,R,Ce)
dr = R/(N-1)
r = np.linspace(0,R,N)

#solution analytique
ra = np.linspace(0,R,200)
Ca= 1/4*S/Deff*R**2*(ra**2/R**2-1) + Ce

#graph sol analytique et numérique
plt.plot(r, C_1, "bo", label=f"schéma ordre 1 avec {N} points")
plt.plot(r, C_2, "ro", label=f"schéma ordre 2 avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()


#graph erreur
Ne = [5, 25, 100, 150, 300]

L1 = np.zeros(len(Ne))
L2 = np.zeros(len(Ne))
Linf = np.zeros(len(Ne))

DR = np.zeros(len(Ne))

for i in range(len(Ne)):
    
    DR[i] = R/(Ne[i]-1) 
    C = resolution_EDP_ordre_2(Ne[i],S,Deff,R,Ce)
    
    dr = R/(Ne[i]-1)
    r = np.linspace(0,R,Ne[i])
    Ca = 1/4*S/Deff*R**2*(r**2/R**2-1) + Ce
    
    for k in range(Ne[i]):
        L1[i]+=1/Ne[i]*abs(C[k]-Ca[k])
        L2[i]+=1/Ne[i]*abs(C[k]-Ca[k])**2
    L2[i] = np.sqrt(L2[i])
    Linf[i] =  max(abs(C-Ca))
    
p = np.log(L2[-1]/L2[-2])/np.log(DR[-1]/DR[-2])
print(f"l'odre de l'erreur avec la norme L2 est p = {p}")

plt.figure(figsize=(8, 6))  
plt.loglog(DR, L1,'bo', label="Norme L1")
plt.loglog(DR, L2, 'ro', label="Norme L2")
plt.loglog(DR, Linf, 'yo', label="Norme Linf")
plt.xlabel('Taille de maille $Δr$ (m)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_1$, $L_2$ et $L_inf$  (mol/m^3)', fontsize=12, fontweight='bold')
plt.title("Norme des erreurs en fonction de $Δr$ ")
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
plt.grid(True)
plt.legend()
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.show()