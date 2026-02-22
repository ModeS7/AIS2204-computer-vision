import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

corners = [ [-1,0.5,0.5], [+1,0.5,0.5], [0,-0.5,0.5], [0,0.5,-0.5] ]

face1 = [ corners[0], corners[1], corners[2] ]
face2 = [ corners[0], corners[1], corners[3] ]
face3 = [ corners[0], corners[2], corners[3] ]
face4 = [ corners[1], corners[2], corners[3] ]

vertices = np.array([face1,face2,face3,face4],dtype=float)

hvertices1 = np.reshape(vertices, (12,3))
hvertices2 = np.hstack( [hvertices1, np.ones([12,1])] )

hvertices = np.reshape(hvertices2, (4,3,4))


ob = Poly3DCollection(vertices, linewidths=1, alpha=0.2)
ob.set_facecolor( [0.5, 0.5, 1] )
ob.set_edgecolor([0,0,0])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(ob)
plt.show()



R1 = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
R2 = np.array([[2**0.5/2, -2**0.5/2, 0], [2**0.5/2, 2**0.5, 0],[0, 0, 1]])
R3 = np.array([[2**0.5/2, 2**0.5/2, 0], [2**0.5/2, -2**0.5/2, 0],[0, 0, 1]])
R4 = np.array([[2**0.5/2, 0, -2**0.5/2], [0, 1, 0],[2**0.5/2, 0, 2**0.5]])
R5 = np.array([[1, 0, 0], [0, 0, 1],[0, 1, 0]])
R6 = np.array([[0, 1, 0], [1, 0, 0],[0, 0, 1]])
R7 = np.array([[2, 0, 0], [0, 2, 0],[0, 0, 2]])
R8 = np.matmul(R6, R7)
R9 = np.array([[1/3, 0, 0], [0, 1/3, 0],[0, 0, 1/3]])
R10 = np.matmul(R2, R4)

vertices0 = np.matmul((vertices + np.array([2,0,0])), R4)

print(vertices0)

#ob1 = Poly3DCollection(vertices0, linewidths=1, alpha=0.2)
#ob1.set_facecolor( [0.5, 0.5, 1] )
#ob1.set_edgecolor([0,0,0])
#fig = plt.figure()
#ax1 = fig.add_subplot(111, projection='3d')
#ax1.add_collection3d(ob1)
#plt.show()

#H1 = np.array([[2**0.5/2, -2**0.5/2, 0, 1], [2**0.5/2, 2**0.5, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

H11 = np.array([[3**0.5/2, -0.5, 0, 2], [3**0.5/2, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

H2 = np.array([[2**0.5/2, 0, -2**0.5/2, 0], [0, 1, 0, 1],[2**0.5/2, 0, 2**0.5, 1], [0, 0, 0, 1]])

H22 = np.array([[0, 0, 1, -2], [0.5, -3**0.5/2, 0, 0], [0.5, 3**0.5/2, 0, 0], [0, 0, 0, 1]])

#newhvertices1 = np.matmul(hvertices, np.matmul(H22, H11))

newhvertices1 = np.matmul(hvertices, H22)
newvertices1 = newhvertices1[:,:,:3]

ob2 = Poly3DCollection(newvertices1, linewidths=1, alpha=0.2)
ob2.set_facecolor([0.5, 0.5, 1])
ob2.set_edgecolor([0, 0, 0])
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.add_collection3d(ob2)
plt.show()

