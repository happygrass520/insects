import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class VelocityField:
    # Constants for calculating potential
    # DELTA = 0.0001
    DELTA = 1

    def __init__(self, p1, p2, p3):
        self.p_x = p1
        self.p_y = p2
        self.p_z = p3

    def get_velocity(self, coordinates):
        bounds = self.p_x.shape[0]
        x,y,z = coordinates
        if x < 0 or x >= bounds - 1:
            return (0,0,0)
        if y < 0 or y >= bounds - 1:
            return (0,0,0)
        if z < 0 or z >= bounds - 1:
            return (0,0,0)
        v_x = (self.p_z[x,y+self.DELTA,z] - self.p_z[x,y-self.DELTA,z])
        v_x = v_x - (self.p_y[x,y,z+self.DELTA] - self.p_y[x,y,z-self.DELTA])
        v_x = v_x / ( 2.0 * float(self.DELTA) )

        v_y = (self.p_x[x,y,z+self.DELTA] - self.p_x[x,y,z-self.DELTA])
        v_y = v_y - (self.p_z[x+self.DELTA,y,z] - self.p_z[x-self.DELTA,y,z])
        v_y = v_y / ( 2.0 * float(self.DELTA) )

        v_z = (self.p_y[x+self.DELTA,y,z] - self.p_y[x-self.DELTA,y,z])
        v_z = v_z - (self.p_x[x,y+self.DELTA,z] - self.p_x[x,y-self.DELTA,z])
        v_z = v_z / ( 2.0 * float(self.DELTA) )

        return (v_x, v_y, v_z)

    def round_velocity_vector(self, vel_vec):
        x,y,z = vel_vec
        return (int(x), int(y), int(z))


def precalculate_values(shape):
    # Arbtritrary value, look up why 4 feels good
    res1 = (4,4,4)
    res2 = (1,1,1)
    res3 = (8,4,8)
    # Generate p1,p2,p3 since they are generated after one another
    # they should be suffuciently different randoms
    print("Generating perlin noise...")
    p1 = generate_perlin_noise_3d(shape, res1)
    print("Generated P1...")
    p2 = generate_perlin_noise_3d(shape, res2)
    print("Generated P2...")
    p3 = generate_perlin_noise_3d(shape, res3)
    print("Generated P3!")
    print(f"P1: mean:{np.mean(p1)} min:{np.min(p1)} max: {np.max(p1)}")
    print(f"P2: mean:{np.mean(p2)} min:{np.min(p2)} max: {np.max(p2)}")
    print(f"P3: mean:{np.mean(p3)} min:{np.min(p3)} max: {np.max(p3)}")
    return p1, p2, p3

def generate_perlin_noise_3d(shape, res):
    """
    Generate perlin noise in 3D
    Taken from Github pvigier/perlin-numpy - perlin3d.py
    """
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def main():
    shape = (128, 128, 128)
    precalculate_values(shape)
    quit()
    res = (4,4,4)
    noise = generate_perlin_noise_3d(shape, res)
    print(noise.shape)

    fig = plt.figure()
    images = [[plt.imshow(layer, cmap='gray', interpolation='lanczos', animated=True)] for layer in noise]
    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True)
    plt.show()

    shape = (128, 128, 128)
    res = (1,4,4)
    noise = generate_perlin_noise_3d(shape, res)

    fig = plt.figure()
    images = [[plt.imshow(layer, cmap='gray', interpolation='lanczos', animated=True)] for layer in noise]
    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True)
    plt.show()

if __name__ == '__main__':
    main()
