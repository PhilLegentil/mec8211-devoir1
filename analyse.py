# Résolution d'un probleme de diffusion par différence finies
# génération des vecteurs solutions et création des graphes

import fct
import numpy as np
import matplotlib.pyplot as plt

# parametres du probleme
class parametres():
    S = 2*10**-8
    Deff = 10**-10
    Ce = 20
    R = 0.5

N = 5
prm = parametres()
dr = prm.R/(N-1)
r = np.arange(0, prm.R+dr, dr)

#solution numérique et calcul de l'ordre de convergence
C_o1 = fct.resolution_EDP_ordre_1(N, r, prm)
p_L2_o1, L1_o1, L2_o1, DR_o1, Linf_o1 = fct.ordre_convergence(fct.resolution_EDP_ordre_1, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 1 est p = {p_L2_o1}") 

C_o2 = fct.resolution_EDP_ordre_2(N, r, prm)
p_L2_o2, L1_o2, L2_o2, DR_o2, Linf_o2 = fct.ordre_convergence(fct.resolution_EDP_ordre_2, prm)
print(f"l'odre de l'erreur avec la norme L2 pour le schema d'ordre 2 est p = {p_L2_o2}") 

#solution analytique
ra, Ca = fct.sol_an(200, prm)

#graph sol analytique et numérique pour l'ordre 1
plt.plot(r, C_o1, "ro", label=f"solution numérique avec {N} points")
plt.plot(ra, Ca, label="solution analytique")
plt.xlabel("r [m]")
plt.ylabel("concentration en sel [mol/m^3]")
plt.title("Évolution de la concentration en sel selon la position radiale du cylindre")
plt.grid("on")
plt.legend()
plt.show()

#graph des normes L1 et L2 des erreurs pour l'ordre 1
fct.graph_convergence(DR_o1, L1_o1, L2_o1, Linf_o1, " 1")

#------------partie E (même calcul avecun ordre 2)------------#

#graph sol analytique et numérique ordre 1 et 2
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
fct.graph_convergence(DR_o2, L1_o2, L2_o2, Linf_o2," 2")
