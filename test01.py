import numpy as np

def escalonar_filas(M):
    """ 
        Retorna la Matriz Escalonada por Filas 
    """
    A = np.copy(M)
    if (issubclass(A.dtype.type, np.integer)):
        A = A.astype(float)

    # Si A no tiene filas o columnas, ya esta escalonada
    f, c = A.shape
    if f == 0 or c == 0:
        return A

    # buscamos primer elemento no nulo de la primera columna
    i = 0
    
    while i < f and A[i,0] == 0:
        i += 1

    if i == f:
        # si todos los elementos de la primera columna son ceros
        # escalonamos filas desde la segunda columna
        B = escalonar_filas(A[:,1:])
        
        # y volvemos a agregar la primera columna de zeros
        return np.block([A[:,:1], B])


    # intercambiamos filas i <-> 0, pues el primer cero aparece en la fila i
    if i > 0:
        A[[0,i],:] = A[[i,0],:]

    # PASO DE TRIANGULACION GAUSSIANA:
    # a las filas subsiguientes les restamos un multiplo de la primera
    A[1:,:] -= (A[0,:] / A[0,0]) * A[1:,0:1]

    # escalonamos desde la segunda fila y segunda columna en adelante
    B = escalonar_filas(A[1:,1:])

    # reconstruimos la matriz por bloques adosando a B la primera fila 
    # y la primera columna (de ceros)
    return np.block([ [A[:1,:]], [ A[1:,:1], B] ])


def back_substitution(A_aug):
    """
    Realiza la sustitución hacia atrás para obtener la identidad en el lado izquierdo
    de la matriz aumentada y la inversa en el lado derecho.
    """
    n = A_aug.shape[0]
    
    # Desde la última fila hacia la primera
    for i in range(n-1, -1, -1):
        # Normalizar la fila de pivote
        A_aug[i] = A_aug[i] / A_aug[i, i]
        
        # Hacer ceros en las filas superiores
        for j in range(i):
            A_aug[j] -= A_aug[i] * A_aug[j, i]
    
    return A_aug[:, n:]


def invertir_matriz(A):
    """
    Calcula la inversa de la matriz A utilizando eliminación gaussiana.
    """
    # Crear una matriz aumentada con A y la identidad, aplicamos eliminacion y la sustitución hacia atrás.
    Id = np.identity(A.shape[0])
    matriz_aumentada = np.hstack((A,Id))
    x = escalonar_filas(matriz_aumentada)
    A_inv = back_sustitution(x)
    
    return A_inv


def resolver_sistema(A, b):
    """
    Resuelve el sistema Ax = b utilizando la inversa de A obtenida mediante eliminación gaussiana.
    """

    A_inv = invertir_matriz(A)
    x = A_inv @ b
    return x


# Modificación de la función de cambio de base
def cambio_base_matriz(B_prime, B):
    """
    Calcula la matriz de cambio de base de B a B' usando `resolver_sistema`.
    """
    n = len(B)
    C = np.zeros((n, n))
    
    # Resolver los sistemas B_prime_i = C * B_i
    for i in range(n):
        C[:, i] = resolver_sistema(B,B_prime[i])
    
    return C
