# Résolution d'un problème de diffusion, dépendant du temps, par la méthode d'Euleur implicite avec 
#vérification de la solution par la MMS

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:29:05 2025

@author: malatchoumymarine
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.ticker as ticker
from fct import *


# %% RÉSOLUTION DU PROBLEME

# Paramètres du problème
N = 30  # Nombre de points dans la discrétisation radiale
Deff = 10**-10  # Coefficient de diffusion effectif
Ce = 20  # Concentration à la frontière extérieure
R = 0.5  # Rayon de la région de diffusion
k = 4*10**-9  # Constante de vitesse de réaction
tf = 4*10**9  # Temps final de simulation
dt = 4*10**9/100  # Pas de temps

# Résolution numérique de l'EDP par la méthode d'Euler implicite
C = resolution_EDP_ordre_2_time(N, Deff, R, Ce, k, tf, dt)

# Discrétisation de l'espace
dr = R/(N-1)  # Taille du maillage radial
ri = np.linspace(0,R,N)  # Valeurs discrétisées pour la solution numérique

# Discrétisation pour la solution analytique
ra = np.linspace(0,R,200)

# Graphique : solution numérique
temps_tra = np.array([0,1,2,5,10,99])  # Instants de temps à afficher

for t in temps_tra:
    Ct = C[t,:]  # Extraire la concentration à l'instant t
    plt.plot(ri, Ct, "-+", label=f"solution numérique à t = {t*dt/(3600*24*365):.1f} ans")

plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
# plt.title("Évolution de la concentration en sel à t = 126 ans")
plt.grid("on")
plt.legend(fontsize=8)
plt.show()


# %% APPLICATION DE LA MMS 

# Définition des paramètres pour la solution analytique (MMS)
class Parametres:
    tf = 1  # Temps final
    Nt = 100  # Nombre de points en temps
    N = 20  # Nombre de points en espace
    Deff = 1  # Diffusion
    R = 0.5  # Rayon
    k = 10  # Vitesse de réaction
    Da = k*R**2/Deff  # Nombre de Damköhler

prm = Parametres()

# Variables symboliques pour Sympy
t, r = sp.symbols('t r')

# Solution analytique choisie (fonction sinus)
C_sy = sp.sin(4*r)*sp.exp(-10**-3*t)

# Définition de l'équation différentielle pour la MMS
S_sy = sp.diff(C_sy, t) - prm.Deff*1/r*sp.diff(r*sp.diff(C_sy, r), r) + prm.k*C_sy

# Conversion des solutions symboliques en fonctions numériques
S = sp.lambdify([t,r], S_sy, "numpy")
C_f = sp.lambdify([t,r], C_sy, "numpy")

# Conditions aux limites
Neu = sp.lambdify(t, sp.diff(C_sy, r).subs(r, 0), "numpy")  # Neumann à r=0
Dir = sp.lambdify(t, C_sy.subs(r, prm.R), "numpy")  # Dirichlet à r=R

# Résolution numérique avec la MMS
C_MMS = resolution_EDP_ordre_2_MMS(prm, S, C_f, Neu, Dir)

# Discrétisation de l'espace pour la solution MMS
r_vals = np.linspace(0, prm.R, prm.N)

# Graphique : solution MMS pour différents instants
t_v = np.array([1,100,200,300, 500, 1000])
for ti in t_v:
    # Calcul des valeurs de la solution analytique pour chaque r
    C_ref = [C_sy.subs({t: ti, r: rv, prm.Deff: prm.Deff}).evalf() for rv in r_vals]
    plt.plot(r_vals, C_ref, "-", label=f"t = {ti} s")

plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
# plt.title("profil de concentration chosit pour la MMS")
plt.grid("on")
plt.legend()
plt.show()

# Graphique : terme source MMS pour différents instants
t_v = np.array([1,100,400,1000])
r_vals = r_vals[1:]  # Enlève le premier point pour éviter la singularité à r=0
for ti in t_v:
    # Calcul des valeurs du terme source S pour chaque r
    S_ref = [S_sy.subs({t: ti, r: rv, prm.Deff: prm.Deff}).evalf() for rv in r_vals]
    plt.plot(r_vals, S_ref, "-", label=f"à t = {ti} s")

plt.xlabel("r [m]")
plt.ylabel("terme source MMS [mol/(s.m^3)]")
plt.grid("on")
plt.legend()
plt.show()



# %% ÉTUDE DE CONVERGENCE EN ESPACE

Ne = np.linspace(5, 1000, 15, dtype=int)

L1 = np.zeros(len(Ne))
L2 = np.zeros(len(Ne))
Linf = np.zeros(len(Ne))
DR = np.zeros(len(Ne))

for k in range(len(Ne)):
    L1[k], L2[k], Linf[k], DR[k] = calcul_erreur_espace(Ne[k], prm, S, 
                                                        C_f, Neu, Dir)
    

p_L2 = np.log(L2[-1]/L2[-2])/np.log(DR[-1]/DR[-2])
print(f"l'odre de l'erreur avec la norme L2 en spatial est p = {p_L2}")

plot_convergence(DR, L1, L2, Linf, p_L2, 
                 xlabel='Taille de maille $Δr$ (m)', 
                 ylabel='Erreur $L_1$, $L_2$ et $L_inf$  (mol/m^3)', 
                 title="Norme des erreurs en fonction de $Δr$")
 
# %% ÉTUDE DE CONVERGENCE EN TEMPS

Nt = np.linspace(5, 800, 15, dtype=int)

L1 = np.zeros(len(Nt))
L2 = np.zeros(len(Nt))
Linf = np.zeros(len(Nt))
Dt = np.zeros(len(Nt))

for k in range(len(Nt)):
    L1[k], L2[k], Linf[k], Dt[k] = calcul_erreur_temps(Nt[k], prm, S, 
                                                       C_f, Neu, Dir)
    

p_L2 = np.log(L2[-1]/L2[0])/np.log(Dt[-1]/Dt[0])
print(f"l'odre de l'erreur avec la norme L2 en temps est p = {p_L2}")

plot_convergence(Dt, L1, L2, Linf, p_L2, 
                 xlabel='Pas de temps $Δt$ (m)', 
                 ylabel='Erreur $L_1$, $L_2$ et $L_inf$  (mol/m^3)', 
                 title="Norme des erreurs en fonction de $Δt$")
