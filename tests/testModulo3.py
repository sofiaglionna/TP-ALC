# Tests L03-Normas
from Modulos.modulo3 import norma, normaliza, normaExacta, normaMatMC, condMC, condExacta
import numpy as np

# Tests norma
print("TESTS NORMA")
assert(np.allclose(norma(np.array([0,0,0,0]),1), 0))
assert(np.allclose(norma(np.array([4,3,-100,-41,0]),"inf"), 100))
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

print("------ÉXITO!!!!\n")

# Tests normaliza
print("TEST NORMALIZA")

# caso borde
# print("---TEST NORMALIZA NULO")
# test_borde = normaliza([np.array([0,0,0,0])],2)
# assert(len(test_borde) == 1)
# assert(np.allclose(test_borde[0],np.array([0,0,0,0])))
# print("------ÉXITO!!!!")

# normaliza norma 2
print("---TEST NORMALIZA 2")
test_n2 = normaliza([np.array([1]*k) for k in range(1,11)],2)
assert(len(test_n2) != 0)
for x in test_n2:
    assert(np.allclose(norma(x,2),1))
print("------ÉXITO!!!!")

# normaliza norma 1
print("---TEST NORMALIZA 1")
test_n1 = normaliza([np.array([1]*k) for k in range(2,11)],1)
assert(len(test_n1) != 0)
for x in test_n1:
    assert(np.allclose(norma(x,1),1))
print("------ÉXITO!!!!")

# normaliza norma inf
print("---TEST NORMALIZA INF")
test_nInf = normaliza([np.random.rand(k) for k in range(1,11)],'inf')
assert(len(test_nInf) != 0)
for x in test_nInf:
    assert(np.allclose(norma(x,'inf'),1))

print("------ÉXITO!!!!\n")

# Tests normaExacta
print("TEST normaExacta")

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[0],2))
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[1],2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[0] ,6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[1],7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)))[0] <=10)
assert(normaExacta(np.random.random((4,4)))[1] <=4)

print("------ÉXITO!!!!\n")

# Test normaMatMC
print("TEST normaMatMC")

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A)[1],rtol=1e-1)) 

print("------ÉXITO!!!!\n")

# Test condMC
print("TEST condMC")

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

print("------ÉXITO!!!!\n")

# Test condExacta
print("TEST condExacta")

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A)[0]
normaA_ = normaExacta(A_)[0]
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A)[1]
normaA_ = normaExacta(A_)[1]
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))

print("------ÉXITO!!!!\n")

print("---FINALIZADO LABO 3!---")