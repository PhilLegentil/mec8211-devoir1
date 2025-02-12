# Résolution d'un probleme de diffusion par différence finies
# génération des vecteurs solutions et création des graphes

import fct
import numpy as np
import sympy as sp
import pytest

import matplotlib.pyplot as plt
# import pytest


# parametres du probleme
class Parametres:
    S = 2 * 10 ** -8
    Deff = 10 ** -10
    Ce = 20
    R = 0.5
    k = 4e-9
    Nm = 100
    #nb d'année à simuler
    annee = 10
    #valeur du dt en jours
    nb_jours_dt = 20
    #conversion du dt en secondes
    dt = nb_jours_dt*24*3600

prm = Parametres()
N = prm.Nm
dr = prm.R / (N - 1)
r = np.arange(0, prm.R + dr/2, dr)
#nombre d'itérations en temps totales pour le probleme transitoire
Niter_t = int(prm.annee*365/prm.nb_jours_dt)

t, rf = sp.symbols('t rf')

vecteur_t = np.linspace(0, Niter_t*prm.dt, Niter_t)

# %%
# solution numérique et calcul de l'ordre de convergence
C_o1 = fct.resolution_EDP_ordre_1(N, r, prm)
p_L2_o1, L1_o1, L2_o1, DR_o1, Linf_o1 = fct.ordre_convergence(fct.resolution_EDP_ordre_1, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 1 est p = {p_L2_o1}")

C_o2 = fct.resolution_EDP_ordre_2(N, r, prm)
p_L2_o2, L1_o2, L2_o2, DR_o2, Linf_o2 = fct.ordre_convergence(fct.resolution_EDP_ordre_2, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 2 est p = {p_L2_o2}")

# solution analytique
ra, Ca = fct.sol_an(200, prm)

# graph sol analytique et numérique pour l'ordre 1
plt.plot(r, C_o1, "ro", label=f"Solution numérique avec {N} points")
plt.plot(ra, Ca, label="Solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre \n schéma d'ordre 1")
plt.grid("on")
plt.legend()
plt.show()

# graph des normes L1 et L2 des erreurs pour l'ordre 1
fct.graph_convergence_polyfit(DR_o1, L1_o1, L2_o1, Linf_o1, " 1")

# ------------partie E (même calcul avec un ordre 2)------------#

# graph sol analytique et numérique ordre 1 et 2
plt.plot(r, C_o1, "bo", label=f"schéma ordre 1 avec {N} points")
plt.plot(r, C_o2, "ro", label=f"schéma ordre 2 avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()

# graph des normes L1 et L2 des erreurs pour l'ordre 2
fct.graph_convergence(DR_o2, L1_o2, L2_o2, Linf_o2, " 2")

C_o2_source = fct.resolution_EDP_ordre_2_source2(N, r, prm)
# %%

cond_init = np.zeros(N)
cond_init[-1] = prm.Ce

source_1 = sp.lambdify([t, rf], 0, "numpy")

C_imp = fct.euler_imp(N, r, Niter_t, prm.dt, prm,cond_init, np.zeros(Niter_t) , np.repeat(prm.Ce, Niter_t), source_1)

plt.plot(r, C_imp[-1,:], label='imp')
plt.plot(r,C_o2_source, label='stat')
plt.legend()
plt.show()

# %%


C_sy = sp.exp(7*rf)*sp.exp(-10**-10*t)

fct.MMS_euler_imp(C_sy, prm, vecteur_t, r, N, Niter_t, prm.dt)
# %%

pytest.main(['-q', '--tb=long', 'corr.py'])
