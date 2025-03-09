# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:30:40 2025

@author: Phil
"""

import pytest
import numpy as np
from fct import (
    resolution_EDP_ordre_2_time,
    fit_poly,
    resolution_EDP_ordre_2_MMS,
    mat_C_MMS,
    calcul_erreur_espace,
    calcul_erreur_temps,
)

# Test pour la fonction resolution_EDP_ordre_2_time
def test_resolution_EDP_ordre_2_time():
    N = 10
    Deff = 1.0
    R = 1.0
    Ce = 0.0
    k = 0.1
    tf = 1.0
    dt = 0.1
    result = resolution_EDP_ordre_2_time(N, Deff, R, Ce, k, tf, dt)
    assert result.shape == (11, 10), "La forme de la matrice C_t est incorrecte"
    assert np.isclose(result[-1, -1], Ce), "La concentration à la frontière ne correspond pas à Ce"

# Test pour la fonction fit_poly
def test_fit_poly():
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 9])
    coeffs = fit_poly(x, y)
    assert np.isclose(coeffs[0], 2), "La pente de la régression n'est pas correcte"
    assert np.isclose(coeffs[1], 0), "L'ordonnée à l'origine n'est pas correcte"


def test_resolution_EDP_ordre_2_MMS():
    class Parametres:
        def __init__(self):
            self.R = 1.0
            self.N = 10
            self.Nt = 10
            self.tf = 1.0
            self.Deff = 1.0
            self.k = 0.1
    
    prm = Parametres()
    
    def S(t, r):
        return 0.0
    
    def C_MMS(t, r):
        return np.exp(-t) * np.sin(r)
    
    def Neu(t):
        return 0.0
    
    def Dir(t):
        return 1.0
    
    result = resolution_EDP_ordre_2_MMS(prm, S, C_MMS, Neu, Dir)
    assert result.shape == (10, 10), "La forme de la matrice C_t est incorrecte"
    assert np.isclose(result[-1, -1], 1.0), "La concentration à la frontière ne correspond pas à Dir"


def test_mat_C_MMS():
    def C_exact(t, r):
        return np.exp(-t) * np.sin(r)
    
    Nr = 10
    Nt = 10
    dr = 0.1
    dt = 0.1
    
    result = mat_C_MMS(C_exact, Nr, Nt, dr, dt)
    assert result.shape == (Nt, Nr), "La forme de la matrice C_mat est incorrecte"
    assert np.isclose(result[0, 0], np.sin(0)), "La valeur au premier point est incorrecte"


def test_calcul_erreur_espace():
    class Parametres:
        def __init__(self):
            self.R = 1.0
            self.N = 10
            self.Nt = 10
            self.tf = 1.0
            self.Deff = 1.0
            self.k = 0.1
    
    prm = Parametres()
    
    def S(t, r):
        return 0.0
    
    def C_f(t, r):
        return np.exp(-t) * np.sin(r)
    
    def Neu(t):
        return 0.0
    
    def Dir(t):
        return 1.0
    
    L1, L2, Linf, dr = calcul_erreur_espace(10, prm, S, C_f, Neu, Dir)
    
    assert L1 >= 0, "L'erreur L1 ne peut pas être négative"
    assert L2 >= 0, "L'erreur L2 ne peut pas être négative"
    assert Linf >= 0, "L'erreur Linf ne peut pas être négative"


def test_calcul_erreur_temps():
    class Parametres:
        def __init__(self):
            self.R = 1.0
            self.N = 10
            self.Nt = 10
            self.tf = 1.0
            self.Deff = 1.0
            self.k = 0.1
    
    prm = Parametres()
    
    def S(t, r):
        return 0.0
    
    def C_f(t, r):
        return np.exp(-t) * np.sin(r)
    
    def Neu(t):
        return 0.0
    
    def Dir(t):
        return 1.0
    
    L1, L2, Linf, dt = calcul_erreur_temps(10, prm, S, C_f, Neu, Dir)
    
    assert L1 >= 0, "L'erreur L1 ne peut pas être négative"
    assert L2 >= 0, "L'erreur L2 ne peut pas être négative"
    assert Linf >= 0, "L'erreur Linf ne peut pas être négative"

