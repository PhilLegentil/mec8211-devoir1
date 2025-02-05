# definition des fonction pour resoudre le problème de diffusion par 
# differences finies

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:35 2025

@author: malatchoumymarine
"""

import numpy as np
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
