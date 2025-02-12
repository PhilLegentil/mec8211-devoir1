# definition des fonction pour resoudre le problème de diffusion par 
# differences finies

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:35 2025

@author: malatchoumymarine
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def resolution_EDP_ordre_1(N, r, prm):
    # resolution du systeme matriciel pour DF d'ordre 1

    dr = prm.R / (N - 1)
    A = np.zeros((N, N))
    B = np.zeros(N)

    # conditions frontières
    # à r=R
    A[-1, -1] = 1
    B[-1] = prm.Ce

    # à r=0
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    B[0] = 0

    S = prm.S
    Deff = prm.Deff
    for i in range(1, N - 1):
        A[i, i] = (-2 / dr ** 2 - 1 / (dr * r[i]))
        A[i, i + 1] = (1 / dr ** 2 + 1 / (dr * r[i]))
        A[i, i - 1] = 1 / dr ** 2

        B[i] = S / Deff

    C = np.linalg.solve(A, B)

    return C


def resolution_EDP_ordre_2(N, r, prm):
    # resolution du systeme matriciel pour DF d'ordre 2

    dr = prm.R / (N - 1)
    A = np.zeros((N, N))
    B = np.zeros(N)

    # conditions frontières
    # à r=R
    A[-1, -1] = 1
    B[-1] = prm.Ce

    # à r=0
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1

    # domaine

    for i in range(1, N - 1):
        A[i, i] = -2 / dr ** 2
        A[i, i + 1] = (1 / dr ** 2 + 1 / (2 * dr * r[i]))
        A[i, i - 1] = 1 / dr ** 2 - 1 / (2 * r[i] * dr)

        B[i] = prm.S / prm.Deff

    C = np.linalg.solve(A, B)

    return C


def resolution_EDP_ordre_2_source2(N, r, prm):
    # resolution du systeme matriciel pour DF d'ordre 2

    dr = prm.R / (N - 1)
    A = np.zeros((N, N))
    B = np.zeros(N)

    # conditions frontières
    # à r=R
    A[-1, -1] = 1
    B[-1] = prm.Ce

    # à r=0
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1

    # domaine

    for i in range(1, N - 1):
        A[i, i] = -2*prm.Deff/ dr ** 2 - prm.k
        A[i, i + 1] = prm.Deff*(1 / dr ** 2 + 1 / (2 * dr * r[i]))
        A[i, i - 1] = prm.Deff*(1 / dr ** 2 - 1 / (2 * r[i] * dr))

    C = np.linalg.solve(A, B)

    return C

def euler_exp(N, r, Niter_t, dt, prm):
    # schema d'Euler explicite pour le problème spécifique du devoir 2
    C = np.zeros((Niter_t,N))
    
    dr = prm.R / (N - 1)
    
    C[:, -1] = prm.Ce
    
    un_tier = 1/3
    for i in range(1, Niter_t):
        
        a = un_tier*(4*C[i-1,1] - C[i-1,2])*prm.Deff*dt*(1/dr**2-1/(2*dr*r[1]))
        b = C[i-1, 1]*(1-2*prm.Deff*dt/(dr**2)-prm.k*dt)
        c = C[i-1, 2]*prm.Deff*dt*(1/(2*dr*r[1])+1/dr**2)
        
        C[i,1] =  a + b + c
        
        for j in range(2, N-1):
            a = C[i-1, j-1]*prm.Deff*dt*(1/dr**2-1/(2*dr*r[j]))
            b = C[i-1, j]*(1-2*prm.Deff*dt/dr**2-prm.k*dt)
            c = C[i-1, j+1]*prm.Deff*dt*(1/(2*dr*r[j])+1/dr**2)
            C[i,j] = a + b + c
        C[i, 0] = un_tier*(4*C[i,1] - C[i,2])
    return C

def euler_imp(N, r, Niter_t, dt, prm, cond_init, bound_gauche, bound_droite, S):
    """methode d'Euler implicite généraliser pour différentes valeurs de conditions frontière"""
    """
    

    Parameters
    ----------

    cond_init : array(size=N)
        Contient les conditions initiales pour chaque coordonnée en r
    bound_gauche : array(size=Niter_t)
        NEUMANN : valeur de la dérivé première pour r=0 à chaque coordonée en t
    bound_droite : array(size=Niter_t)
        DIRICHLET : valeur de la concentration pour r=R à chaque coordonnée en t
    S : fonction 
        retourne la valeur du terme source en chaque point (t,r)

    Returns
    -------
    C : np.array((N_itert, N))
        Chaque ligne contient le profil de concentration 
        en r à la coordonnée temporel i*dt
        

    """

    
    deux_tier = 2/3
    dr = prm.R / (N - 1)
    C = np.zeros((Niter_t,N))
    
    C[0,:]= cond_init.copy()
    
    A = np.zeros((N,N))
    A[0,0] = -3
    A[0,1] = 4
    A[0,2] = -1
    
    A[1,1] = prm.Deff*dt*(4/(3*r[1]*2*dr)+2/(3*dr**2)) + prm.k*dt + 1
    A[1,2] = prm.Deff*dt*(-4/(6*r[1]*dr)-2/(3*dr**2))
    A[N-1,N-1] = 1
    
    for i in range(2,N-1):
        A[i,i-1] = prm.Deff*dt*(1/(r[i]*2*dr)-1/dr**2)
        A[i, i] = 2*dt*prm.Deff/dr**2 + prm.k*dt + 1
        A[i,i+1] = prm.Deff*dt*(-1/(r[i]*2*dr)-1/dr**2)

    for t in range(1, Niter_t):
        B = C[t-1,:].copy()
        B[0] = bound_gauche[t] * 2 * dr
        B[1] +=  deux_tier*bound_gauche[t]*prm.Deff*dt*(1/(2*r[1]) - 1/dr)
        B[1:N-2] -= np.array([S(t*dt, ri) for ri in r[1:N-2]])*dt
        B[-1] = bound_droite[t]
        C[t,:] = np.linalg.solve(A, B)

    return C
    

def sol_an(N, prm):
    # creation des points pour la solution analytique
    ra = np.linspace(0, prm.R, N)
    solution = 1 / 4 * prm.S / prm.Deff * prm.R ** 2 * (ra ** 2 / prm.R ** 2 - 1) + prm.Ce
    return ra, solution


def ordre_convergence(schema, prm):
    # calcul l'ordre de convergence selon le schema

    Ne = [5, 25, 100, 150, 300]

    L1 = np.zeros(len(Ne))
    L2 = np.zeros(len(Ne))
    Linf = np.zeros(len(Ne))

    DR = np.zeros(len(Ne))

    # calcul de L1 et L2 pour chaque Ne
    for i in range(len(Ne)):

        dr = prm.R / (Ne[i] - 1)
        r = np.arange(0, prm.R + dr, dr)
        DR[i] = prm.R / (Ne[i] - 1)
        C = schema(Ne[i], r, prm)

        ra, Ca = sol_an(Ne[i], prm)

        # calcul des normes L1 et L2 pour Ne specifique
        for k in range(Ne[i]):
            L1[i] += 1 / Ne[i] * abs(C[k] - Ca[k])
            L2[i] += 1 / Ne[i] * abs(C[k] - Ca[k]) ** 2
        L2[i] = np.sqrt(L2[i])
        Linf[i] = max(abs(C - Ca))

    # calcul de la pente a partir des deux derniers 
    # points de chaque vecteurs norme
    ordre_conv = np.log(L2[-1] / L2[-2]) / np.log(DR[-1] / DR[-2])

    return ordre_conv, L1, L2, DR, Linf


# Appliquer une régression linéaire en log-log
def fit_poly(x, y):
    log_x, log_y = np.log10(x), np.log10(y)
    coeffs = np.polyfit(log_x, log_y, 1)  # Régression linéaire
    return coeffs  # coeffs[0] = pente, coeffs[1] = intercept

def graph_convergence_polyfit(DR, L1, L2, Linf, titre_graph):
    
    sorted_indices = np.argsort(DR)
    DR_sorted = np.array(DR)[sorted_indices]
    L1_sorted = np.array(L1)[sorted_indices]
    L2_sorted = np.array(L2)[sorted_indices]
    Linf_sorted = np.array(Linf)[sorted_indices]
    
    # Sélection des 3 plus petits DR
    DR_fit = DR_sorted[:3]  
    L1_fit = L1_sorted[:3]
    L2_fit = L2_sorted[:3]
    Linf_fit = Linf_sorted[:3]
    
    # Calcul des régressions pour chaque norme
    slope_L1, intercept_L1 = fit_poly(DR_fit, L1_fit)
    slope_L2, intercept_L2 = fit_poly(DR_fit, L2_fit)
    slope_Linf, intercept_Linf = fit_poly(DR_fit, Linf_fit)
    
    # Génération des lignes de tendance
    DR_line = np.linspace(min(DR), max(DR), 100)  # Étend la ligne sur tout le graphe
    L1_line = 10**(intercept_L1) * DR_line**slope_L1
    L2_line = 10**(intercept_L2) * DR_line**slope_L2
    Linf_line = 10**(intercept_Linf) * DR_line**slope_Linf
    
    # Création du graphique
    plt.figure(figsize=(8, 6))
    
    # Points
    plt.loglog(DR, L1, 'bo', label="Norme L1")
    plt.loglog(DR, L2, 'ro', label="Norme L2")
    plt.loglog(DR, Linf, 'yo', label="Norme Linf")
    
    # Lignes de tendance issues de la régression linéaire
    plt.loglog(DR_line, L1_line, 'b--', linewidth=2, label="Régression L1")
    plt.loglog(DR_line, L2_line, 'r--', linewidth=2, label="Régression L2")
    plt.loglog(DR_line, Linf_line, 'y--', linewidth=2, label="Régression Linf")
    
    plt.xlabel('Taille de maille $Δr$ (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Erreur $L_1$, $L_2$ et $L_{\infty}$  (mol/m³)', fontsize=12, fontweight='bold')
    plt.title("Norme des erreurs en fonction de $Δr$ pour le schéma d'ordre" + titre_graph)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
    plt.grid(True)
    plt.legend()
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.show()
    
def graph_convergence(DR, L1, L2, Linf, titre_graph):
    plt.figure(figsize=(8, 6))
    plt.loglog(DR, L1, 'bo', label="Norme L1")
    plt.loglog(DR, L2, 'ro', label="Norme L2")
    plt.loglog(DR, Linf, 'yo', label="Norme Linf")
    plt.xlabel('Taille de maille $Δr$ (m)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
    plt.ylabel('Erreur $L_1$, $L_2$ et $L_inf$  (mol/m^3)', fontsize=12, fontweight='bold')
    plt.title("Norme des erreurs en fonction de $Δr$ pour le schéma d'ordre" + titre_graph)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
    plt.grid(True)
    plt.legend()
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.show()
    

def MMS_euler_imp(C_sy, prm, vecteur_t, r, N, Niter_t, dt):
    # application de la MMS pour le schéma d'Euler implicite
    t, rf = sp.symbols('t rf')
    C_f = sp.lambdify([t,rf], C_sy, "numpy")
    
    S_sy = - sp.diff(C_sy, t) + prm.Deff/rf*sp.diff(rf*sp.diff(C_sy, rf),rf) - prm.k*C_sy
    
    S = sp.lambdify([t,rf], S_sy , "numpy")
    
    Neumann_gauche = sp.lambdify(t,sp.diff(C_sy, rf).subs(rf, 0) , "numpy")
    Dirichlet_droite = sp.lambdify(t, C_sy.subs(rf,prm.R), "numpy")
    
    bound_gauche = Neumann_gauche(vecteur_t)
    bound_droite = Dirichlet_droite(vecteur_t)
    cond_init_MMS = [C_f(0, ri) for ri in r]
    
    C_MMS = euler_imp(N, r, Niter_t, dt, prm, cond_init_MMS, bound_gauche, bound_droite, S)
    
    plt.plot(r, C_MMS[-1], label="sol num MMS")
    plt.plot(r, C_f(vecteur_t[-1], r), label="sol imposee")
    plt.legend()
    plt.show()
    
    erreur_MMS = abs(C_MMS[-1] - C_f(vecteur_t[-1], r))
    
    return erreur_MMS