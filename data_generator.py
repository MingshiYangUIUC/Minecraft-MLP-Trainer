# Generate a random dataset of optimal aiming using bow and arrow.

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from multiprocessing import Pool
import os
wd = os.path.dirname(__file__)
def equations2(p,R,y):
    a,t = p
    return (100*3*np.cos(a)*(1-0.99**t)-R, -5*t+100*(3*np.sin(a)+5)*(1-0.99**t)-y)

def ATprecise(R,y):
    if R == 0:
        ag = 0
    else:
        ag = np.arctan(y/R)

    sol,id,ier,mesg =  fsolve(equations2, (ag,np.sqrt(R**2+y**2)/3),args=(R,y),full_output=True)
    #print(sol,ier,mesg)
    if ier == 1:
        a,t = sol
        a = np.round((a * 180 / np.pi),4)
        t = np.round(t,4)
        return np.array((a,t))
    else:
        return ['o','o']
    

def zxtoRr(z,x):# angle is relative to z axis
    R = np.sqrt(x**2+z**2)
    if z == 0:
        r = np.pi/2*np.sign(x)
    else: r = np.arctan(x/z)
    r = r/np.pi*180
    return R,r


def aim3D(PosT3,VT3,g=0):
    #target velocity vector VT3 (dx,dy,dz)
    #target position (x,y,z)
    #R,r = zxtoRr(PosT3[2],PosT3[0])
    #PosT0 = np.array(R,Post3[1])

    tol = 0.01 # tolerance of time difference error
    lr = 0.9 # learning rate
    tries = 0
    trimax = 15
    tbase = 1
    grad = 0
    err = 1000
    errl = []
    while np.abs(err) > tol and tries < trimax:
        tbase += grad*lr
        PosTp = PosT3+VT3*tbase # estimated final point 3D
        R,r = zxtoRr(PosTp[2],PosTp[0])
        PosT = np.array((R,PosTp[1])) # estimated R and y
        #tn = getT(PosT[0],PosT[1])
        tn = ATprecise(PosT[0],PosT[1])[1]
        #print(PosT,grad,tbase,tn)
        if tn == 'o':
            err = tbase
            grad = err / -2
        else: 
            err = tn-tbase
            grad = err
        #grad += 0.01 fwd
        errl.append(err)
        tries += 1
    if tn != 'o' and tries < trimax:
        #at = (getA(PosT[0],PosT[1]),tn)
        at = ATprecise(PosT[0],PosT[1])
        #print(f'Angle:{at[0]}, Time:{at[1]}, Tries:{tries}, Error:{np.round(err,4)}')
        if g > 0:
            print(f'Angle:{at[0]}, Time:{at[1]}, Tries:{tries}, Error:{np.round(err,4)}')
        return 'Y',tries,at,(R,r),PosT3,VT3,PosTp
    else: return 'N', errl


def create_dataset_chunk(chunk_size, maxv):
    chunk_data = np.zeros((chunk_size, 7),dtype=np.float16)
    i = 0
    while i < chunk_size:
        R, y = np.random.uniform(0, 150), np.random.uniform(-100, 70)
        VT3 = np.random.uniform(-maxv, maxv, 3)  # vfront, vtop, vright
        VT3[-1] = np.abs(VT3[-1])
        output = aim3D(np.array((R, y, 0)), VT3)
        if output[0] == 'Y':
            chunk_data[i] = [R, y, VT3[0], VT3[1], VT3[2], output[2][0], output[2][1]]
            i += 1
        if i % 100 == 0:
            print(str(i).zfill(6),chunk_size,end='\r')
    return chunk_data

def concatenate_datasets(datasets):
    return np.concatenate(datasets, axis=0)

if __name__ == '__main__':
    maxv = 0 # max speed of target in meters/tick.
    N = 2400000
    num_processes = 24
    chunk_size = N // num_processes
    
    # Create a pool of worker processes
    pool = Pool(num_processes)

    # Use each process to create a portion of the dataset
    dataset_chunks = pool.starmap(create_dataset_chunk, [(chunk_size, maxv)] * num_processes)
    print('\n')
    # Concatenate the dataset chunks
    dataset = concatenate_datasets(dataset_chunks)
    
    # Print the shape of the final dataset
    print(dataset.shape,dataset.nbytes/1024**2)

    # define output path
    np.save(os.path.join(wd,'data',f'ATdata_{maxv}_{N}_0.npy'),dataset)