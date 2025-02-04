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

class parametres():
    S = 2*10**-8
    Deff = 10**-10
    Ce = 20
    R = 0.5

def resolution_EDP_ordre_1(N, r, prm):

    dr = prm.R/(N-1)
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    #conditions frontières
    #à r=R
    A[-1,-1] = 1
    B[-1] = prm.Ce
    
    #à r=0
    A[0,0] = -3
    A[0,1] =  4
    A[0,2] = -1
    B[0] = 0
    
    #domaine
    S = prm.S
    Deff = prm.Deff
    for i in range(1,N-1):
        
        A[i,i] = (-2/dr**2-1/(dr*r[i]))
        A[i,i+1] = (1/dr**2+1/(dr*r[i]))
        A[i,i-1] = 1/dr**2
        
        B[i] = S/Deff
        
    C = np.linalg.solve(A,B)
    
    return C

def resolution_EDP_ordre_2(N ,r ,prm):
    dr = prm.R/(N-1)
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    #conditions frontières
    #à r=R
    A[-1,-1] = 1
    B[-1] = prm.Ce
    
    #à r=0
    A[0,0] = -3
    A[0,1] =  4
    A[0,2] =  -1
    
    #domaine

    for i in range(1,N-1):
        
        A[i,i] = -2/dr**2
        A[i,i+1] = (1/dr**2+1/(2*dr*r[i]))
        A[i,i-1] = 1/dr**2 - 1/(2*r[i]*dr)
        
        B[i] = prm.S/prm.Deff
        
        
    C = np.linalg.solve(A,B)
    
    return C

def sol_an(N):
    ra = np.linspace(0,prm.R,N)
    solution = 1/4*prm.S/prm.Deff*prm.R**2*(ra**2/prm.R**2-1) + prm.Ce
    return ra, solution

def ordre_convergence(schema, prm):
    
    Ne = [5, 25, 100, 150, 300]

    L1 = np.zeros(len(Ne))
    L2 = np.zeros(len(Ne))
    Linf = np.zeros(len(Ne))

    DR = np.zeros(len(Ne))

    for i in range(len(Ne)):
        
        dr = prm.R/(Ne[i]-1)
        r = np.arange(0, prm.R+dr, dr)
        DR[i] = prm.R/(Ne[i]-1) 
        C = schema(Ne[i], r, prm)
        
        ra, Ca = sol_an(Ne[i])
        
        for k in range(Ne[i]):
            L1[i]+=1/Ne[i]*abs(C[k]-Ca[k])
            L2[i]+=1/Ne[i]*abs(C[k]-Ca[k])**2
        L2[i] = np.sqrt(L2[i])
        Linf[i] =  max(abs(C-Ca))


    ordre_conv = np.log(L2[-1]/L2[-2])/np.log(DR[-1]/DR[-2])
    
    return ordre_conv, L1, L2, DR, Linf
    
#paramètres du problème
N = 5
prm = parametres()
dr = prm.R/(N-1)
r = np.arange(0, prm.R+dr, dr)

#solution numérique
C_o1 = resolution_EDP_ordre_1(N, r, prm)
p_L2_o1, L1_o1, L2_o1, DR_o1, Linf_o1 = ordre_convergence(resolution_EDP_ordre_1, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 1 est p = {p_L2_o1}") 

C_o2 = resolution_EDP_ordre_2(N, r, prm)
p_L2_o2, L1_o2, L2_o2, DR_o2, Linf_o2 = ordre_convergence(resolution_EDP_ordre_2, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 2 est p = {p_L2_o2}") 

#solution analytique
ra, Ca = sol_an(200)

#graph sol analytique et numérique
plt.plot(r, C_o1, "ro", label=f"solution numérique avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()

#graph des normes des erreurs
plt.figure(figsize=(8, 6))  
plt.loglog(DR_o1, L1_o1,'bo', label="Norme L1")
plt.loglog(DR_o1, L2_o1, 'ro', label="Norme L2")
plt.loglog(DR_o1, Linf_o1, 'yo', label="Norme Linf")
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


#graph sol analytique et numérique
plt.plot(r, C_o1, "bo", label=f"schéma ordre 1 avec {N} points")
plt.plot(r, C_o2, "ro", label=f"schéma ordre 2 avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))  
plt.loglog(DR_o2, L1_o2,'bo', label="Norme L1")
plt.loglog(DR_o2, L2_o2, 'ro', label="Norme L2")
plt.loglog(DR_o2, Linf_o2, 'yo', label="Norme Linf")
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