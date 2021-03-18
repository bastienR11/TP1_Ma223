import numpy as np
import time as t
import matplotlib.pyplot as plt
from math import log

TempsGauss=[]
TempsGauss2=[]
ValueGauss=[]
TempsLU=[]
TempsLU2=[]
ValueLU=[]
Tempspivotpartiel=[]
Tempspivotpartiel2=[]
Valuepivotpartiel=[]
Tempspivottotal=[]
Tempspivottotal2=[]
Valuepivottotal=[]

def ReducDeGauss (Aaug): 
    """Fonction qui permet la réduction de Gauss sans changement d'indice, avec comme argument une matrice augmentée et retourne une matrice triangulaire supérieure augmentée
    """
    m,n = Aaug.shape                                  # on relève le nombre de lignes et de colonnes
#    if m != n-1:
#        print("On ne peux pas effectuer la fonction. La matrice doit être augmentée pour obtenir un résultat")
 #       return
    A = np.copy(Aaug)                                 #copie de la matrice sur laquelle le programme sera effectuée
    for i in range(m):
        l = A[i,i]                                    #on prend le coefficient de la diagonale de la ligne i
        if (l == 0):                                  #test de la valeur du coefficient
            print("La réduction de A n'existe pas")
            return
        else:
            for j in range(i+1,m):
                g = A[j,i]/l                          #calcul du coefficient de ligne
                A[j,:] -= g*A[i,:]                    #transformation de la matrice pour avoir des zéros sous la diagonale
    return A

def ResolutionSystTriSup (Taug):
    """Fonction qui permet la résolution d'un système sous la forme d'une matrice triangulaire supérieure. On a comme argument une matrice  triangulaire supérieure augmentée et retourne une matrice colonne solution du système
    """
    m,n = Taug.shape                                  #enregistrement du nombre de lignes et de colonnes de la matrice
    if m != n-1:
        print("On ne peux pas effectuer la fonction. La matrice doit être augmentée pour obtenir un résultat")
        return
    T = np.copy(Taug)                                 #copie de la matrice sur laquelle le programme sera effectuée
    X = np.zeros((m,1))                               #création de la matrice colonne résultat que l'on remplira au fur et à mesure
    for k in range(m-1,-1,-1):
        bk = T[k,n-1]                                 #enregistrement de la valeur de la colonne augmentée
        tk = T[k,k]                                   #enregistrement de la valeur du coeffiecient de la variable que l'on calcule
        s = 0                                         #ajout du premier terme nul d'une somme
        for i in range(k+1,m):
            s+= (T[k,i])*(X[i,0])                     #somme de toutes les variables déjà calculée dans la matrice colonne multipliée par leurs coefficients de ligne
        X[k,0] = (bk-s)/tk                            #calcul de la variable de la ligne
    return X

def Gauss(A,B):
    """Fonction qui permet la résolution AX = B via l'utilisation de la méthode de Gauss. Elle a comme argument une matrice carrée et une matrice colonne du même nombre de lignes et retourne une matrice colonne résultat
    """
    B=B.reshape((B.size,1))
    m1,n1 = A.shape
    m2,n2 = B.shape
    if m1 != n1:
        print("La matrice A n'est pas carrée, le calcul n'est donc pas possible")
        return
    if n2 != 1 or m2 != m1:
        print("La matrice B n'est pas aux bonnes dimensions, le calcul n'est pas possible")
        return
    C = np.concatenate((A,B),axis=1)
    T = ReducDeGauss(C)
    X = ResolutionSystTriSup(T)
    return X

def ReducDeGaussPivotPartiel (Aaug):
    """Fonction qui permet la réduction de Gauss en prennant le coefficient de la colonne le plus grand en valeur absolue. On a comme argument une matrice augmentée et retourne une matrice triangulaire supérieure augmentée
    """
    m,n = Aaug.shape                                  #enregistrement du nombre de lignes et de colonnes de la matrice
    if m != n-1:
        print("On ne peux pas effectuer la fonction. La matrice doit être augmentée pour obtenir un résultat")
        return
    A = np.copy(Aaug)                                 #copie de la matrice sur laquelle le programme sera effectuée
    for i in range(m):
        l = A[i,i]                                    #on prend le coefficient de la diagonale de la ligne i
        l1 = abs(l)                                   #enregistrement de la valeur absolue de ce coefficient
        v = i                                         #enregistrement du numéro de la ligne
        for j in range(i+1,m):
            if l1 <abs(A[j,i]):                       #comparaison de la valeur absolue enregistrée avec les valeurs absolues des éléments inférieurs de la colonne
                l1 = abs(A[j,i])                      #échange de la valeur si la valeur absolue est plus grande ainsi que du numéro de ligne
                v = j
        if (l1 == 0):
            print("La réduction de A n'existe pas")
            return
        else:
            if v != i:                                #comparaison du numéro de ligne
                L = A[i,:].copy()
                A[i,:] = A[v,:]                       #échange des deux lignes dans le cas ou cette valeur est différente
                A[v,:] = L
            l = A[i,i]                                #on choisit l'élément de la diagonale de la ligne (cas ou le coefficient est négatif)
            for k in range(i+1,m):
                g = A[k,i]/l                          #calcul du coefficient de ligne
                A[k,:] -= g*A[i,:]                    #transformation de la matrice pour avoir des zéros sous la diagonale
    return A

def GaussChoixPivotPartiel(A,B):
    """Fonction qui permet la résolution AX = B via l'utilisation de la méthode de Gauss en choissisant un pivot dans la colonne si possible. Elle a comme argument une matrice carrée et une matrice colonne du même nombre de lignes et retourne une matrice colonne résultat
    """
    B=B.reshape((B.size,1))
    m1,n1 = A.shape
    m2,n2 = B.shape
    if m1 != n1:
        print("La matrice A n'est pas carrée, le calcul n'est donc pas possible")
        return
    if n2 != 1 or m2 != m1:
        print("La matrice B n'est pas aux bonnes dimensions, le calcul n'est pas possible")
        return
    C = np.concatenate((A,B),axis = 1)
    T = ReducDeGaussPivotPartiel(C)
    X = ResolutionSystTriSup(T)
    return X

def DecompositionLU (A):
    """Fonction qui permet la décomposition LU d'une matrice. On a comme argument une matrice carrée et retourne une matrice L une matrice traingulaire inférieure avec des 1 sur sa diagonale et une matrice U triangulaire supérieure
    """
    m,n = A.shape                                     #enregistrement du nombre de lignes et de colonnes de la matrice
    if m != n:
        print("On ne peux pas effectuer la fonction. La matrice doit être carrée pour obtenir un résultat")
        return
    U = np.copy(A)                                    #copie de la matrice sur laquelle le programme sera effectuée
    L = np.eye(m)                                     #création d'une matrice identité que l'on remplira au fur et à mesure
    for i in range(m):
        l = U[i,i]                                    #on prend le coefficient de la diagonale de la ligne i
        if (l == 0):
            return("La décomposition de A n'existe pas")
        else:
            for j in range(i+1,m):
                g = U[j,i]/l                          #calcul du coefficient de ligne
                U[j,:] -= g*U[i,:]                    #transformation de la matrice pour avoir des zéros sous la diagonale
                L[j,i] = g                            #ajout du coefficient de ligne dans la matrice identité
    return L,U
    
def ResolutionSystTriInf (Laug):
    """Fonction qui permet la résolution d'un système sous la forme d'une matrice triangulaire inférieure.
       On a comme argument une matrice triangulaire inférieure augmentée et retourne une matrice colonne solution du système
    """
    m,n = Laug.shape                                  #enregistrement du nombre de lignes et de colonnes de la matrice
    if m != n-1:
        print("On ne peux pas effectuer la fonction. La matrice doit être augmentée pour obtenir un résultat")
        return
    L = np.copy(Laug)                                 #copie de la matrice sur laquelle le programme sera effectuée
    Y = np.zeros((m,1))                               #création de la matrice colonne résultat que l'on remplira au fur et à mesure
    for i in range(0,m):
        b = L[i,n-1]                                  #enregistrement de la valeur de la colonne augmentée
        t = L[i,i]                                    #enregistrement de la valeur du coeffiecient de la variable que l'on calcule
        s = 0                                         #ajout du premier terme nul d'une somme
        for k in range(0,i):
            s += (L[i,k])*(Y[k,0])                    #somme de toutes les variables déjà calculée dans la matrice colonne multipliée par leurs coefficients de ligne
        Y[i,0] = (b-s)/t                              #calcul de la variable de la ligne
    return Y

def ResolutionLU (L,U,B):
    """Fonction qui permet la résolution AX = B via l'utilisation de la décomposition LU de la matrice A.
       Elle a comme argument les matrices L et U de la décomposition de A et une matrice colonne du
       même nombre de lignes et retourne une matrice colonne résultat
    """
    B=B.reshape((B.size,1))
    m1,n1 = L.shape
    m2,n2 = U.shape
    m3,n3 = B.shape
    if m1 != n1 or m2 != n2 or m1 != m2 or n1 != n2:
        print("La décomposition n'est pas correcte, la fonction ne pourra pas donner de résultat")
        return
    if n3 != 1 or m3 != m1:
        print("La matrice B n'est pas aux bonnes dimensions, le calcul n'est pas possible")
        return
    Laug = np.concatenate((L,B),axis = 1)
    Y = ResolutionSystTriInf(Laug)
    T = np.concatenate((U,Y),axis=1)
    X = ResolutionSystTriSup(T)
    return X

def GaussChoixPivotTotal(A,B):
    """Fonction qui permet la résolution AX = B via l'utilisation de la méthode de Gauss en choissisant un pivot dans la matrice si possible. Elle a comme          argument une matrice carrée et une matrice colonne du même nombre de lignes et retourne une matrice colonne résultat
    """
    X= np.hstack([A,B])  
    n,m = X.shape
    if m != n+1:
        print("Il ne s'agit pas d'une matrice augmentée")
        return
    
    for k in range(0,m-1):
        for i in range(k+1,m-1):
            for var in range(i,m-1):
                if abs(X[var,k]) > abs(X[k,k]):
                    L0 = np.copy(X[k,:])
                    X[k,:] = X[var,:]
                    X[var,:] = L0
            if X[k,k] == 0:
                print("Il y a un pivot nul")
                return
            g = X[i,k]/X[k,k]
            X[i,:] = X[i,:] - g*X[k,:]
    
    X= ResolutionSystTriSup(X)

    return X 

A = np.array ([[1., 2, 4], [2, 8, 4], [4, 4, 24]])
B = np.array ([[2],[6],[-6]])

sol=Gauss(A,B)
print(sol)
sol2=GaussChoixPivotPartiel(A,B)
print(sol2)
L,U=DecompositionLU(A)
sol3=ResolutionLU(L,U,B)
print(sol3)
sol4=GaussChoixPivotTotal(A,B)
print(sol4)
for n in range(0,1000,20):

    try:
        A = np.random.randint(low = 1, high = n, size = (n,n))
        B = np.random.randint(low = 1, high = n, size = (n,1))
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t1 = t.time()
        Gauss(A,B)
        t2 = t.time()
        t3 = t2-t1
        TempsGauss.append(t3)
        TempsGauss2.append(log(t3))
        ValueGauss.append(n)
    except:
        pass

x1 = ValueGauss
y1 = TempsGauss

plt.plot(x1, y1, color = "red", label="gauss" )
plt.xlabel("dimension de la matrice")
plt.ylabel("temps de calcul")
plt.title("temps résolution des matrices en fonction de leur taille")
plt.savefig("Gauss",format ="png")
plt.show()

for n in range(0,1000,20):
    try:
        A = np.random.randint(low = 1, high = n, size = (n,n))
        B = np.random.randint(low = 1, high = n, size = (n,1))
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t4 = t.time()
        L,U=DecompositionLU(A)
        ResolutionLU (L,U,B)
        t5 = t.time()
        t6 = t5-t4
        TempsLU.append(t6)
        TempsLU2.append(log(t6))
        ValueLU.append(n)
    except:
        pass

x2 = ValueLU
y2 = TempsLU

plt.plot(x2, y2, color = "blue",label="LU")
plt.xlabel("dimension de la matrice")
plt.ylabel("temps de calcul")
plt.title("temps résolution des matrices en fonction de leur taille")
plt.savefig("LU",format ="png")
plt.show()


for n in range(0, 1000,20):
    try:
        A = np.random.randint(low = 1, high = n, size = (n,n))
        B = np.random.randint(low = 1, high = n, size = (n,1))
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t7 = t.time()
        GaussChoixPivotPartiel(A,B)
        t8 = t.time()
        t9 = t7-t6
        Tempspivotpartiel.append(t8)
        Tempspivotpartiel2.append(log(t8))
        Valuepivotpartiel.append(n)
    except:
        pass

x3 = Valuepivotpartiel
y3 = Tempspivotpartiel

plt.plot(x3, y3, color = "green", label="PivotPartiel")
plt.xlabel("dimension de la matrice")
plt.ylabel("temps de calcul")
plt.title("temps résolution des matrices en fonction de leur taille")
plt.savefig("pivot_partiel",format ="png")
plt.show()

for n in range(0,1000,20):
    try:
        A = np.random.randint(low = 1, high = n, size = (n,n))
        B = np.random.randint(low = 1, high = n, size = (n,1))
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t10 = t.time()
        GaussChoixPivotTotal(A,B)
        t11 = t.time()
        t12 = t11-t10
        Tempspivottotal.append(t12)
        Tempspivottotal2.append(log(t12))
        Valuepivottotal.append(n)
    except:
        pass

y4 = Tempspivottotal
x4 = Valuepivottotal

plt.plot(x4, y4, color = "black", label="PivotTotak")
plt.xlabel("dimension de la matrice")
plt.ylabel("temps de calcul")
plt.title("temps résolution des matrices en fonction de leur taille")
plt.savefig("pivot_total",format ="png")
plt.show()

y10 = TempsGauss2
y20 = TempsLU2
y30 = Tempspivotpartiel2
y40 = Tempspivottotal2

plt.plot(x1, y10, color = "red", label="gauss" )
plt.plot(x2, y20, color = "blue",label="LU")
plt.plot(x3, y30, color = "green", label="PivotPartiel")
plt.plot(x4, y40, color = "black", label="PivotTotak")
plt.xlabel("dimension de la matrice")
plt.ylabel("temps de calcul")
plt.title("comparaison des temps de résolution en utilisant les différentes méthodes")
plt.savefig("comparaison",format ="png")
plt.show()