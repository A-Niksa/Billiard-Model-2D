# APPROACH: Timestepping
# LIBRARIES
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import cv2
import glob
import os
import gc

# FUNCTIONS
def distanceCalc(x1,y1,x2,y2):
    return math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)

def wallCollision(x,y):
    if x >= x_len-r:
        return 3
    elif x <= r:
        return 1
    elif y >= y_len-r:
        return 2
    elif y <= r:
        return 4
    else:
        return False

def similarRemover(A1,A2): #should find a better way for removing similar tuplets in disk collision
    length = len(A1) # len(A1) = len(A2)
    list_indices = []
    for i in range(length):
        for j in range(length):
            if i != j and A1[i] == A2[j] and A1[j] == A2[i]:
                list_indices.append(j)
    A1 = np.delete(A1,list_indices)
    A2 = np.delete(A2,list_indices)
    return A1,A2

def diskCollision(X_slice,Y_slice): # X[:,i]
    D1 = np.array([]); D2 = D1.copy() # would D2 = np.array([]) have been faster?
    for i in range(n):
        for j in range(n):
            if i!=j and distanceCalc(X_slice[i],Y_slice[i],X_slice[j],Y_slice[j]) <= 2.25*r: # < 2.25*r bc of numerical errors I suppose
                D1 = np.append(D1,i)
                D2 = np.append(D2,j)
    return similarRemover(D1,D2)

def initConditions():
    X_0 = np.random.uniform(r,x_len-r,n)
    Y_0 = np.random.uniform(r,y_len-r,n)
    Ux = np.random.uniform(u_min,u_max,n) # did not put 0 because Ux is going to be updated
    Uy = np.random.uniform(u_min,u_max,n)
    return X_0,Y_0,Ux,Uy

def initCheck(X_0,Y_0): # checking to see if the disks don't collide at t = 0 (othewise: generate new initial conditions)
    D1,D2 = diskCollision(X_0,Y_0)
    length = len(D1)
    while length != 0:
        X_0,Y_0,Ux,Uy = initConditions()
        D1,D2 = diskCollision(X_0,Y_0)
        length = len(D1)
    return X_0,Y_0

def velocityTransformer(x1,y1,ux1,uy1,x2,y2,ux2,uy2): # converting ux and uy to u_parallel and u_perpendicular
    slope = (y2-y1)/(x2-x1)
    slope_normal = -1/slope
    theta = math.atan(slope_normal)
    beta1 = theta-math.atan(uy1/ux1) # theta-alpha1
    beta2 = theta-math.atan(uy2/ux2) # theta-alpha2
    u1 = math.sqrt(ux1**2 + uy1**2)
    u2 = math.sqrt(ux2**2 + uy2**2)
    u1_parallel = u1 * math.cos(beta1)
    u1_perpendicular = u1 * math.sin(beta1)
    u2_parallel = u2 * math.cos(beta2)
    u2_perpendicular = u2 * math.sin(beta2)
    return theta, u1_parallel, u1_perpendicular, u2_parallel, u2_perpendicular


def velocityInverter_wall(ux,uy,m = 0): # m: wall number
    if m == 1 or m == 3:
        return -ux,uy
    else:
        return ux,-uy

def velocityInverter_disks(x1,y1,ux1,uy1,x2,y2,ux2,uy2):
    theta, u1_parallel, u1_perpendicular, u2_parallel, u2_perpendicular = \
        velocityTransformer(x1,y1,ux1,uy1,x2,y2,ux2,uy2)
    u1_perpendicular, u2_perpendicular = -u1_perpendicular, -u2_perpendicular # using - only once?
    gamma = 180-theta; delta = theta-90
    ux1 = u1_parallel*math.cos(gamma) + u1_perpendicular*math.cos(delta)
    uy1 = u1_parallel*math.sin(gamma) + u1_perpendicular*math.sin(delta)
    ux2 = u2_parallel*math.cos(gamma) + u2_perpendicular*math.cos(delta)
    uy2 = u2_parallel*math.sin(gamma) + u2_perpendicular*math.sin(delta)
    return ux1, uy1, ux2, uy2

def simulationProgress(t_i): # t_i: time index
    t_i += 1 # for convenience
    percentage = int(t_i/n_ts * 100)
    if percentage != 100:
        os.system('cls')
        n_bars = percentage//5
        if percentage < 10:
            print("%s%% " % percentage, end = ' ')
        else:
            print("%s%%" % percentage, end = ' ')
        print("[",end = '')
        for i in range(1,21):
            if i <= n_bars:
                print("|",end = '')
            else:
                print(" ",end = '')
        print("]")

# VARIABLES
x_len = 150 # length along the x axis
y_len = 50 # length along the y axis
n_ts = 200 # number of timesteps (we have a memory leak problem)
t_ts = 0.1 # time increments (total time = t_ts * n_ts)
u_min = -5 # minimum initial velocity
u_max = 5 # maximum initial velocity
n = 20 # number of disks
r = 2 # disk radius
fc = '#0AC4A8' # face color of the disks (turquoise)

# INITIAL CONDITIONS
X_0,Y_0,Ux,Uy = initConditions()
X_0,Y_0 = initCheck(X_0,Y_0)

# DEFINING THE GEOMETRY AND ARRAYS
X = np.zeros((n,n_ts)); X[:,0] = X_0.copy()
Y = np.zeros((n,n_ts)); Y[:,0] = Y_0.copy()
circles = [plt.Circle((X[j,0],Y[j,0]),radius = r,linewidth = 0) for j in range(n)]

# DRIVING LOOP
for i in range(n_ts):
    plt.clf()
    if i != 0:
        for j in range(n):
            X[j,i] = X[j,i-1] + Ux[j] * t_ts
            Y[j,i] = Y[j,i-1] + Uy[j] * t_ts
            wC = wallCollision(X[j,i],Y[j,i])
            if wC == False:
                pass
            else:
                Ux[j],Uy[j] = velocityInverter_wall(Ux[j],Uy[j],wC)
        dC1,dC2 = diskCollision(X[:,i],Y[:,i])
        dC1 = dC1.tolist() # should do it in a cleaner way
        dC2 = dC2.tolist()
        #dC = np.concatenate((dC1,dC2)) # would be partially better if included in the function
        length = len(dC1)
        if length != 0:
            for d in range(length): # d:disk index
                dC1[d] = int(dC1[d])
                dC2[d] = int(dC2[d])
                Ux[dC1[d]],Uy[dC1[d]],Ux[dC2[d]],Uy[dC2[d]]\
                     = velocityInverter_disks(X[dC1[d],i],Y[dC1[d],i],Ux[dC1[d]],Uy[dC1[d]], \
                         X[dC2[d],i],Y[dC2[d],i],Ux[dC2[d]],Uy[dC2[d]])

    # Visualisation
    circles = [plt.Circle((X[j,i],Y[j,i]),radius = r,linewidth = 0) for j in range(n)] # can be faster if we add it in the for loop above
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,aspect = 'equal')
    c = mc.PatchCollection(circles)
    c.set_facecolor = fc  # will have to do it in savefig too (weird default)
    ax.add_collection(c)
    i_modified = i + 1
    plt.title("%s" % i_modified)
    axes = plt.gca()
    axes.set_xlim([0,x_len])
    axes.set_ylim([0,y_len])
    plt.savefig("2D_rectangular_TS_%s" % i_modified, facecolor = fig.get_facecolor(), edgecolor='none', dpi = 300) # saves as png by default
    # default: https://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
    plt.close('all')
    gc.collect()

    # Progress check
    simulationProgress(i)

# EXPORTING VIDEO
# VIDEO EXPORTATION
test_img = cv2.imread("2D_rectangular_TS_1.png")
height, width, layers = test_img.shape
framesize = (width, height)
output = cv2.VideoWriter("2D_rectangular_TS_video.avi",cv2.VideoWriter_fourcc(*'DIVX'),18,framesize)
for fname in sorted(glob.glob("*.png"), key = os.path.getmtime):
    img = cv2.imread(fname)
    output.write(img)
    os.remove(fname)

output.release()

# NOTIFYING THE USER THAT THE SIMULATION IS DONE (is not clean tbh)
os.system('cls')
print("100% [||||||||||||||||||||]")