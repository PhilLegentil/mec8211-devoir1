# tests unitaires pour les fonctions qui resolve le probleme de diffusion en differences finis
from fct import *


class Parametres:
    S = 2 * 10 ** -8
    Deff = 10 ** -10
    Ce = 20
    R = 0.5


prm = Parametres()


class Test:

    def test_dfo1(self):
        C = np.array([9.32291667, 9.94791667, 11.82291667, 15.15625, 20], dtype=float)
        r = np.array([0, .125, .25, .375, .5], dtype=float)
        resultat_FDo1 = resolution_EDP_ordre_1(5, r, prm)
        err = abs(resultat_FDo1 - C)
        assert (all(err < 1e-06))

    def test_dfo2(self):
        C = np.array([7.5, 8.28125, 10.625, 14.53125, 20], dtype=float)
        r = np.array([0, .125, .25, .375, .5], dtype=float)
        resultat_FDo2 = resolution_EDP_ordre_2(5, r, prm)
        err = abs(resultat_FDo2 - C)
        assert (all(err < 1e-06))

    def test_ordre(self):
        k = .99254062
        p = ordre_convergence(resolution_EDP_ordre_1, prm)[0]
        err = abs(p - k)
        assert (err < 1e-06)

    def test_L1(self):
        L1 = np.array([1.0625, 0.24375, 0.0621212, 0.0414989, 0.0207915], dtype=float)
        L1_fct = ordre_convergence(resolution_EDP_ordre_1, prm)[1]
        err = abs(L1 - L1_fct)
        assert (all(err < 1e-06))

    def test_L2(self):
        L2 = np.array([1.25908159, 0.282202, 0.0717703, 0.0479357, 0.0240121], dtype=float)
        L2_fct = ordre_convergence(resolution_EDP_ordre_1, prm)[2]
        err = abs(L2 - L2_fct)
        assert (all(err < 1e-06))

    def test_DR(self):
        DR = np.array([0.125, 0.0208333, 0.00505051, 0.0033557, 0.00167224], dtype=float)
        DR_fct = ordre_convergence(resolution_EDP_ordre_1, prm)[3]
        err = abs(DR - DR_fct)
        assert (all(err < 1e-06))

    def test_Linf(self):
        Linf = np.array([1.822916, 0.455279, 0.12098, 0.0813772, 0.0411036], dtype=float)
        Linf_fct = ordre_convergence(resolution_EDP_ordre_1, prm)[4]
        err = abs(Linf - Linf_fct)
        assert (all(err < 1e-06))
