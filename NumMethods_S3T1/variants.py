import numpy as np


def params(v):
    """ Chooses parameters according to the variant number"""
    KC = [2.0, 3.0, 2.5, 3.0, 1.0, 4.0, 4.5, 3.7, 3.0, 1.3, 0.5, 4.0, 2.0, 3.0, 3.5, 2.7, 6.0, 4.0, 0.5, 1.5, 3.0, 5.0,
          2.5, 5.7]
    FC = [2.5, 0.5, 2.0, 3.5, 1.5, 0.5, 7.0, 1.5, 1.5, 3.5, 2.0, 2.5, 3.5, 2.5, 0.7, 3.5, 1.5, 2.5, 3.0, 3.7, 2.5, 0.3,
          5.7, 2.5]
    EC = [1 / 3, 1 / 4, 2 / 3, 4 / 3, 2 / 3, -5 / 4, -2 / 3, -4 / 3, 1 / 4, 2 / 3, 2 / 5, 4 / 7, 5 / 3, 7 / 4, -5 / 3,
          -7 / 3, 5 / 3, 5 / 4, 2 / 5, 4 / 7, 4 / 3, -7 / 4, -4 / 3, -4 / 7]
    KS = [4.0, 5.0, 4.0, 2.0, 3.0, 2.0, 1.4, 2.4, 4.0, 6.0, 2.4, 2.5, 3.0, 5.0, 2.4, 4.4, 2.0, 2.5, 4.0, 3.0, 4.0, 7.0,
          2.4, 4.4]
    FS = [3.5, 2.5, 3.5, 3.5, 5.5, 4.5, 1.5, 4.5, 3.5, 4.5, 1.5, 5.5, 1.5, 0.5, 5.5, 2.5, 0.5, 1.5, 3.5, 2.5, 5.5, 0.5,
          2.5, 4.3]
    ES = [-3, -1 / 3, -3.0, -2 / 3, -2.0, 1 / 8, -1 / 3, 2 / 3, -3.0, -1 / 8, -6.0, -0.6, -4.0, 3 / 8, -3 / 4, 5 / 3,
          -1.3, -2 / 7, -3.0, 3 / 4, -3.5, 2 / 3, -3 / 3, 2 / 7]
    KL = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 4.3, 0.0, 0.0, 0.0, 0.0, 5.4, 5.0, 3.0, 3.0, 0.0, 0.0,
          0.0, 0.0]
    K0 = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0,
          7.0, 5.0]
    A = [1.5, 1.7, 0.1, 1.0, 2.5, 1.3, 2.1, 1.8, 2.5, 0.7, 1.1, 1.8, 1.5, 2.3, 1.1, 2.8, 3.5, 2.7, 1.1, 1.5, 2.5, 0.5,
         0.2, 0.8]
    B = [3.3, 3.2, 2.3, 3.0, 4.3, 2.2, 3.3, 2.3, 3.3, 3.2, 2.5, 2.9, 2.3, 2.9, 2.3, 4.3, 3.7, 3.2, 2.3, 3.0, 4.3, 2.2,
         3.1, 1.3]
    Al = [1 / 3, 0, 1 / 5, 0, 2 / 7, 0, 2 / 5, 0, 2 / 3, 0, 2 / 5, 0, 1 / 5, 0, 4 / 5, 0, 2 / 3, 0, 2 / 5, 0, 2 / 7, 0,
          3 / 5, 0]
    Be = [0, 1 / 4, 0, 1 / 6, 0, 5 / 6, 0, 3 / 5, 0, 1 / 4, 0, 4 / 7, 0, 2 / 5, 0, 3 / 7, 0, 3 / 4, 0, 5 / 6, 0, 3 / 5,
          0, 4 / 7]
    if v < 1 or v > 24:
        return 0
    v = v - 1

    def f(x):
        return KC[v] * np.cos(FC[v] * x) * np.exp(EC[v] * x) + KS[v] * np.sin(FS[v] * x) * np.exp(ES[v] * x) \
               + KL[v] * x + K0[v]

    return A[v], B[v], Al[v], Be[v], f


def exactint(v):
    """ Gives an "exact" value for the variant number"""
    exactvals = [7.077031437995793610263911711602477164432,
                 11.83933565874812191864851199716726555747,
                 3.578861536040539915439859609644293194417,
                 -41.88816344003630606891235682900290027460,
                 10.65722906811476196545133157861241468330,
                 10.83954510946909397740794566485262705081,
                 4.461512705331194112840828080521604042844,
                 1.185141974956241824914878594317090726677,
                 20.73027110955223102601793414048307154080,
                 24.14209267859915860831257727834195698139,
                 18.60294785731848208626949366919856494853,
                 57.48462064655285571820619434508191055583,
                 32.21951452884234295708696008290380201405,
                 348.8181344253911360363589124705960119778,
                 27.56649553650691538577624747358600185818,
                 -3246.875926327328894367882485073567528036,
                 2308.287524452809436132810373422088766896,
                 78.38144689028315358349839381435476300192,
                 8.565534222407634006755741863827588778916,
                 161.7842904748235945321114040034846373768,
                 -262.7627605704703725392313581988618564726,
                 69.34894027882668315183332391303280381054,
                 28.98579534502018413362688379858804448110,
                 -4.249393101145035941757249850984813018073]
    if v < 1 or v > 24:
        return 0
    return exactvals[v-1]