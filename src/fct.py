# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 21:48:40 2025

@author: Phil
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def resolution_EDP_ordre_2_time(N, Deff, R, Ce, k, tf, dt):
    """
    Résout numériquement une équation différentielle partielle de diffusion à l'aide de la méthode d'Euler implicite.
    
    Cette fonction utilise un schéma en différences finies d'ordre 2 pour résoudre l'équation de diffusion
    sur un espace discrétisé radial, en tenant compte des conditions aux limites (Dirichlet et Neumann) et
    en intégrant la solution dans le temps avec la méthode d'Euler implicite.
    
    Args:
    - N (int) : Nombre de points de discrétisation en espace.
    - Deff (float) : Coefficient de diffusion effectif.
    - R (float) : Rayon de la colonne.
    - Ce (float) : Concentration en sel au bord (condition de Dirichlet).
    - k (float) : Taux de réaction.
    - tf (float) : Temps final de la simulation.
    - dt (float) : Pas de temps.
    
    Returns:
    - C_t (numpy.ndarray) : Matrice des concentrations à chaque pas de temps pour chaque point en espace.
    """
    # Initialisation de la discrétisation et des matrices
    dr = R/(N-1)
    A = np.zeros((N, N))
    B = np.zeros(N)
    C_t = B

    # Conditions aux limites
    A[-1, -1] = 1
    B[-1] = Ce
    A[0, 0] = -3 
    A[0, 1] = 4 
    A[0, 2] = -1
    B[0] = 0
    
    # Construction de la matrice pour les points internes
    for i in range(1, N-1):
        ri = i*dr
        A[i, i] = 2/dr**2*dt*Deff + 1 + k*dt
        A[i, i+1] = -(1/dr**2 + 1/(2*ri*dr))*dt*Deff
        A[i, i-1] = -(1/dr**2 - 1/(2*ri*dr))*dt*Deff
    
    # Résolution temporelle
    t = dt
    while t < tf:
        B = np.linalg.solve(A, B)
        C_t = np.vstack((C_t, B))
        t += dt
    
    return C_t


def fit_poly(x, y):
    """
    Effectue une régression linéaire en utilisant la méthode des moindres carrés sur les logarithmes des données.
    
    Cette fonction effectue une régression linéaire sur les logarithmes en base 10 des données x et y. 
    Elle retourne les coefficients de la droite de régression (pente et intercept) ajustée aux données.
    
    Args:
    - x (numpy.ndarray) : Données sur l'axe des x.
    - y (numpy.ndarray) : Données sur l'axe des y.
    
    Returns:
    - coeffs (numpy.ndarray) : Coefficients de la régression (pente et intercept).
    """
    log_x, log_y = np.log10(x), np.log10(y)
    coeffs = np.polyfit(log_x, log_y, 1)  # Régression linéaire
    return coeffs  # coeffs[0] = pente, coeffs[1] = intercept


def resolution_EDP_ordre_2_MMS(prm, S, C_MMS, Neu, Dir):
    """
    Résout numériquement une équation différentielle partielle de diffusion avec la méthode des solutions manufacturées (MMS).
    
    Cette fonction résout une équation de diffusion en utilisant la méthode des solutions manufacturées pour vérifier 
    la précision du schéma numérique. Elle inclut l'intégration dans le temps et les conditions aux limites de Dirichlet et Neumann.
    
    Args:
    - prm (Parametres) : Objet contenant les paramètres de la simulation (R, N, Nt, etc.).
    - S (function) : Fonction représentant le terme source de l'équation.
    - C_MMS (function) : Fonction représentant la solution analytique utilisée pour la comparaison.
    - Neu (function) : Fonction représentant la condition aux limites de Neumann à r=0.
    - Dir (function) : Fonction représentant la condition aux limites de Dirichlet à r=R.
    
    Returns:
    - C_t (numpy.ndarray) : Matrice des concentrations à chaque pas de temps pour chaque point en espace.
    """
    # Initialisation et définition des matrices
    R = prm.R
    N = prm.N
    Nt = prm.Nt
    dt = prm.tf / prm.Nt
    D = prm.Deff
    dr = R / (N - 1)
    r_vals = np.linspace(0, R, N)
    A = np.zeros((N, N))
    B = np.array([C_MMS(0, i) for i in r_vals])
    C_t = B.copy()

    # Conditions aux limites
    A[-1, -1] = 1
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    # Construction de la matrice pour les points internes
    for i in range(1, N - 1):
        ri = i * dr
        A[i, i] = 2 * dt * D / dr**2 + 1 + prm.k * dt
        A[i, i + 1] = -(1 / dr**2 + 1 / (2 * ri * dr)) * dt * D
        A[i, i - 1] = -(1 / dr**2 - 1 / (2 * ri * dr)) * dt * D

    # Résolution temporelle
    tp = dt
    for j in range(1, Nt):
        B_new = B.copy()
        for i in range(1, N - 1):
            ri = i * dr
            B_new[i] += S(tp, ri) * dt
        B_new[-1] = Dir(tp)
        B_new[0] = Neu(tp) * 2 * dr
        B = np.linalg.solve(A, B_new)
        C_t = np.vstack((C_t, B))
        tp += dt

    return np.array(C_t)


def mat_C_MMS(C, Nr, Nt, dr, dt):
    """
    Crée une matrice des concentrations pour une solution analytique donnée.
    
    Cette fonction génère une matrice des concentrations à partir d'une fonction C (solution exacte),
    en évaluant la solution à chaque point du maillage radial et pour chaque instant de temps.
    
    Args:
    - C (function) : Fonction représentant la solution exacte de la concentration.
    - Nr (int) : Nombre de points radiaux.
    - Nt (int) : Nombre de points temporels.
    - dr (float) : Pas de discrétisation spatial (radial).
    - dt (float) : Pas de discrétisation temporel.
    
    Returns:
    - C_mat (numpy.ndarray) : Matrice des concentrations à chaque instant de temps pour chaque point radial.
    """
    C_mat = np.zeros((Nt, Nr))
    for k in range(Nr):
        for i in range(Nt):
            C_mat[i, k] = C(i * dt, k * dr)
    return C_mat


def calcul_erreur_espace(Ne, prm, S, C_f, Neu, Dir):
    """
    Calcule les erreurs de la solution numérique par rapport à la solution exacte dans l'espace (erreur L1, L2 et Linf).
    
    Cette fonction calcule les erreurs entre la solution numérique et la solution exacte dans l'espace
    pour différentes tailles de maillage radial (Ne), en utilisant les normes L1, L2 et Linf.
    
    Args:
    - Ne (int) : Nombre de points de discrétisation en espace.
    - prm (Parametres) : Paramètres de la simulation.
    - S (function) : Fonction représentant le terme source.
    - C_f (function) : Fonction représentant la solution exacte.
    - Neu (function) : Condition de Neumann à r=0.
    - Dir (function) : Condition de Dirichlet à r=R.
    
    Returns:
    - L1 (float) : Erreur L1.
    - L2 (float) : Erreur L2.
    - Linf (float) : Erreur Linf.
    - dr (float) : Pas de discrétisation radial.
    """
    L1 = 0
    L2 = 0
    Linf = 0
    prm.N = Ne
    dr = prm.R / (Ne - 1)
    dt = prm.tf / prm.Nt
    C = resolution_EDP_ordre_2_MMS(prm, S, C_f, Neu, Dir)
    C_exact = mat_C_MMS(C_f, Ne, prm.Nt, dr, dt)
    
    for ti in range(prm.Nt):
        for k in range(Ne):
            L1 += dt * dr * abs(C[ti, k] - C_exact[ti, k])
            L2 += dt * dr * abs(C[ti, k] - C_exact[ti, k])**2
    
    L2 = np.sqrt(L2)
    Linf = np.max(np.abs(C - C_exact))
        
    return L1, L2, Linf, dr



def calcul_erreur_temps(Nt, prm, S, C_f, Neu, Dir):
    """
    Calcule les erreurs de la solution numérique par rapport à la solution exacte dans le temps (erreur L1, L2 et Linf).
    
    Cette fonction calcule les erreurs entre la solution numérique et la solution exacte dans le temps
    pour différentes tailles de pas de temps (Nt), en utilisant les normes L1, L2 et Linf.
    
    Args:
    - Nt (int) : Nombre de points de discrétisation en temps.
    - prm (Parametres) : Paramètres de la simulation.
    - S (function) : Fonction représentant le terme source.
    - C_f (function) : Fonction représentant la solution exacte.
    - Neu (function) : Condition de Neumann à r=0.
    - Dir (function) : Condition de Dirichlet à r=R.
    
    Returns:
    - L1 (float) : Erreur L1.
    - L2 (float) : Erreur L2.
    - Linf (float) : Erreur Linf.
    - dt (float) : Pas de discrétisation temporel.
    """
    prm.N = 1200
    prm.Nt = Nt
    L1 = 0
    L2 = 0
    Linf = 0
    dr = prm.R / (prm.N - 1)
    dt = prm.tf / (Nt - 1)
    
    C = resolution_EDP_ordre_2_MMS(prm, S, C_f, Neu, Dir)
    C_exact = mat_C_MMS(C_f, prm.N, Nt, dr, dt)
    
    for ti in range(Nt):
        for k in range(prm.N):
            L1 += dt * dr * abs(C[ti, k] - C_exact[ti, k])
            L2 += dt * dr * abs(C[ti, k] - C_exact[ti, k])**2
    
    L2 = np.sqrt(L2)
    Linf = np.max(np.abs(C - C_exact))
        
    return L1, L2, Linf, dt


# %% ÉTUDE DE CONVERGENCE EN ESPACE ET EN TEMPS

def plot_convergence(D, L1, L2, Linf, p_L2, xlabel, ylabel, title):
    """
    Fonction pour tracer les courbes de convergence des erreurs L1, L2 et Linf en fonction de D (taille de mailles ou pas de temps).
    
    Parameters:
    - D : Liste des tailles de mailles ou des pas de temps.
    - L1, L2, Linf : Erreurs calculées pour chaque taille de mailles ou pas de temps.
    - p_L2 : Ordre d'erreur en norme L2.
    - xlabel : Légende de l'axe X.
    - ylabel : Légende de l'axe Y.
    - title : Titre du graphique.
    """
    plt.figure(figsize=(8, 6))  
    plt.loglog(D, L1, 'bo', label="Norme L1")
    plt.loglog(D, L2, 'ro', label="Norme L2")
    plt.loglog(D, Linf, 'yo', label="Norme Linf")
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
    plt.grid(True)

    # Régression linéaire pour ajuster la tendance des erreurs
    sorted_indices = np.argsort(D)
    D_sorted = np.array(D)[sorted_indices]
    L1_sorted = np.array(L1)[sorted_indices]
    L2_sorted = np.array(L2)[sorted_indices]
    Linf_sorted = np.array(Linf)[sorted_indices]

    D_fit = D_sorted[:6]  
    L1_fit = L1_sorted[:6]
    L2_fit = L2_sorted[:6]
    Linf_fit = Linf_sorted[:6]

    # Calcul des régressions pour chaque norme
    slope_L1, intercept_L1 = fit_poly(D_fit, L1_fit)
    slope_L2, intercept_L2 = fit_poly(D_fit, L2_fit)
    slope_Linf, intercept_Linf = fit_poly(D_fit, Linf_fit)

    # Génération des lignes de tendance
    D_line = np.linspace(min(D), max(D), 100) 

    # Étend la ligne sur tout le graphe
    L1_line = 10**(intercept_L1) * D_line**slope_L1
    L2_line = 10**(intercept_L2) * D_line**slope_L2
    Linf_line = 10**(intercept_Linf) * D_line**slope_Linf

    # Tracé des lignes de régression
    plt.loglog(D_line, L1_line, 'b--', linewidth=2, label="Régression L1")
    plt.loglog(D_line, L2_line, 'r--', linewidth=2, label="Régression L2")
    plt.loglog(D_line, Linf_line, 'y--', linewidth=2, label="Régression Linf")

    # Affichage de l'équation de la courbe L2
    equation = f"y = {10**intercept_L2:.2e} * x^{slope_L2:.2f}"
    plt.text(0.5, 0.1, f"Norme L2 : {equation}", fontsize=12, transform=plt.gca().transAxes, color='r')
    
    # Affichage de l'ordre de l'erreur
    plt.text(0.5, 0.2, f"p = {p_L2:.5g}", fontsize=12, transform=plt.gca().transAxes, color='k')

    plt.legend()
    plt.show()

    