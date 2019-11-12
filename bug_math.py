import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import time

from pprint import pprint

class VelocityField:
    # Constants for calculating potential
    # DELTA = 0.0001
    DELTA = 1
    # In the ramp function, this is used to 
    # amplify the approach to a boundary
    D_0 = 3.0
    P_GAIN = 400.0
    # DEBUG = True
    DEBUG = False

    def __init__(self, p1, p2, p3, bound_x, bound_y, bound_z):
        self.p_x = p1
        self.p_y = p2
        self.p_z = p3

        self.bound_x = bound_x
        self.bound_y = bound_y
        self.bound_z = bound_z

    def plot_alpha_ramp(self):
        # Max size that we send in is (bound / 2)
        # if a bug is in the exact middle
        x = np.arange(0, self.p_x.shape[0] / 2, 1)
        x = np.arange(0, 10, 1)
        different_ds = [1.0, 2.0, 4.0, 8.0]
        ys = []
        for ds in different_ds:
            y = [ self.ramp_function(x_n / ds) for x_n in x ]
            plt.plot(x,y)
        plt.show()
        quit()

    def plot_vec_field(self, step_size = 1):
        """
        Create a 3D plot of the velocity field
        EXPENSIVE as f*** to run
        """
        # Create new grid that can hold the vectors
        # We can get the shape from any one of the p's
        # shape_len = self.p_x.shape[0] // step_size
        shape_len = self.bound_x // step_size

        # Grid        
        X, Y, Z = np.meshgrid(
                np.arange(0, shape_len, 1),
                np.arange(0, shape_len, 1),
                np.arange(0, shape_len, 1),
                )
        # Values
        # U, V, W = np.meshgrid(
                # np.arange(0, shape_len, 1, dtype=np.float),
                # np.arange(0, shape_len, 1, dtype=np.float),
                # np.arange(0, shape_len, 1, dtype=np.float),
                # )
        U = np.zeros((shape_len, shape_len, shape_len), dtype=np.float)
        V = np.zeros((shape_len, shape_len, shape_len), dtype=np.float)
        W = np.zeros((shape_len, shape_len, shape_len), dtype=np.float)

        # A bit strange, but lower-case letters are actual
        # values (index or scalar from vec field) and big
        # letter are arrays
        print("Started calculating vector field...")
        total_values_to_calc = (shape_len**3)
        total_values_calculated = 0
        average_time = 0.0
        total_time = 0.0
        for (x,y,z) in itertools.product(range(shape_len),repeat=3):
            if total_values_calculated % 10 == 0:
                percent_done = float(total_values_calculated) / float(total_values_to_calc) * 100.0
                percent_to_int = int(percent_done)
                prog_string = percent_to_int * '#'
                prog_rev_string = (100 - percent_to_int) * ' '
                time_left = average_time * (total_values_to_calc - total_values_calculated)
                if time_left > 60.0:
                    time_left = time_left / 60.0
                    time_string = 'minutes'
                else:
                    time_string = 'seconds'
                print(f"[{prog_string}{prog_rev_string}] ({percent_to_int}%) {time_left:.0f} {time_string} left...  \r", end='')
                # print(f"Currently at:({x},{y},{z}) [{total_values_calculated}/{total_values_to_calc} = {percent_done}%]         \r", end='')
            time_start = time.time()
            # middle_point = np.array([4.0,4.0,4.0]) 
            middle_point = np.array([4.0,4.0,4.0]) 
            current_point = np.array([float(x),float(y),float(z)])
            vec = middle_point - current_point
            # vec = current_point - middle_point
            print("----------------")
            print(middle_point)
            print(current_point)
            print(vec)
            if np.linalg.norm(vec) != 0.0:
                vec /= np.linalg.norm(vec)
            print(vec)
            print(np.linalg.norm(vec))
            # u, v, w = self.get_velocity((x*step_size,y*step_size,z*step_size))
            # U[x,y,z] = u * 200.0
            # V[x,y,z] = v * 200.0
            # W[x,y,z] = w * 200.0
            # U[x,y,z] = u
            # V[x,y,z] = v
            # W[x,y,z] = w
            U[x,y,z] = vec[0]
            V[x,y,z] = vec[1]
            W[x,y,z] = vec[2]
            time_end = time.time()
            total_time += time_end - time_start
            if total_values_calculated != 0:
                average_time = total_time / float(total_values_calculated)
            total_values_calculated += 1
        print("\nDone")
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.quiver3D(X,Y,Z,U,V,W, length=0.3, normalize=True)
        # ax.quiver3D(X,Y,Z,U,V,W)
        plt.show()
        quit()

    def get_velocity(self, coordinates):
        x,y,z = coordinates

        # Old style, where P = N
        """
        bounds = self.p_x.shape[0]
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
        v_z = v_z / ( 2.0 * float(self.DELTA)  )
        """

        # normal and alpha from boundary and ramp function
        n, a = self.get_closest_boundary_normal(coordinates)
        v_x = (self.get_N((x,y+self.DELTA,z), a, n)[2] - self.get_N((x,y-self.DELTA,z), a, n)[2])
        v_x = v_x - (self.get_N((x,y,z+self.DELTA), a, n)[1] - self.get_N((x,y,z-self.DELTA), a, n)[1])
        v_x = v_x / ( 2.0 * float(self.DELTA) )

        v_y = (self.get_N((x,y+self.DELTA,z), a, n)[0] - self.get_N((x,y-self.DELTA,z), a, n)[0])
        v_y = v_y - (self.get_N((x+self.DELTA,y,z), a, n)[2] - self.get_N((x-self.DELTA,y,z), a, n)[2])
        v_y = v_y / ( 2.0 * float(self.DELTA) )

        v_z = (self.get_N((x+self.DELTA,y,z), a, n)[1] - self.get_N((x-self.DELTA,y,z), a, n)[1])
        v_z = v_z - (self.get_N((x,y+self.DELTA,z), a, n)[0] - self.get_N((x,y-self.DELTA,z), a, n)[0])
        v_z = v_z / ( 2.0 * float(self.DELTA) )

        if self.DEBUG: print(f"velocity vector:({v_x},{v_y},{v_z})")
        return (v_x, v_y, v_z)

    def get_N(self, coordinates, alpha, normal):
        """
        Helper function
        N is defined as
        N = alpha * P + (1 - alpha)(normal [dot] P) * normal
        """
        x,y,z = coordinates
        # p are always boxes of same length
        # We need to adjust the coordinates if we step outside the grid
        # paper does not explain how to handle this
        limit_x = self.p_x.shape[0]
        limit_y = self.p_y.shape[0]
        limit_z = self.p_z.shape[0]
        if x < 0:
            x = 0
        elif x >= limit_x:
            x = limit_x - 1
        if y < 0:
            y = 0
        elif y >= limit_y:
            y = limit_y - 1
        if z < 0:
            z = 0
        elif z >= limit_z:
            z = limit_z - 1
        # Lets get it on!
        p_x = self.p_x[x,y,z]
        p_y = self.p_y[x,y,z]
        p_z = self.p_z[x,y,z]

        # Convert to numpy arrays
        P = np.array([p_x * self.P_GAIN , p_y * self.P_GAIN, p_z * self.P_GAIN])
        P_abs = np.array([np.abs(p_x * self.P_GAIN) , np.abs(p_y * self.P_GAIN), np.abs(p_z * self.P_GAIN)])
        normal = np.array(normal)
        if self.DEBUG: print("---------------")
        if self.DEBUG: print(f"P Before:{P}")
        if self.DEBUG: print(f"normal Before:{normal}")
        if self.DEBUG: print(f"alpha before: {alpha}")
        N = alpha * P + ((1.0 - alpha) * np.dot(normal, P_abs) * normal)
        if self.DEBUG: print(f"N: {N}")
        return N

    def round_velocity_vector(self, vel_vec):
        x,y,z = vel_vec
        return (int(x), int(y), int(z))

    def get_closest_boundary_normal(self, coordinates):
        """
        Returns normal of the closest boundary surface
        as well as the result from ramp_function
        To be multiplied/dotted with P to create N

        has to be calculated in each step for each coordinate
        """
        # Prepare normals for each boundary surface
        # TODO not sure about these
        normal_x_bottom = (0, 0, 1)
        normal_x_top = (0, 0, -1)
        normal_y_bottom = (1, 0, 0)
        normal_y_top = (-1, 0, 0)
        normal_z_bottom = (0, 1, 0)
        normal_z_top = (0, -1, 0)
        x,y,z = coordinates
        distance_x_top = np.sqrt((self.bound_x - x) ** 2)
        distance_x_bottom = np.sqrt((0 - x) ** 2)
        distance_y_top = np.sqrt((self.bound_y - y) ** 2)
        distance_y_bottom = np.sqrt((0 - y) ** 2)
        distance_z_top = np.sqrt((self.bound_z - z) ** 2)
        distance_z_bottom = np.sqrt((0 - z) ** 2)
        # Create two lists, so that we can use min() to find the normal we want
        dist_pairs = [
                ('x_top', distance_x_top, normal_x_top),
                ('x_bot', distance_x_bottom, normal_x_bottom),
                ('y_top', distance_y_top, normal_y_top),
                ('y_bot', distance_y_bottom, normal_y_bottom),
                ('z_top', distance_z_top, normal_z_top),
                ('z_bot', distance_z_bottom, normal_z_bottom),
                ]
        min_dist = min([
                distance_x_top,
                distance_x_bottom,
                distance_y_top,
                distance_y_bottom,
                distance_z_top,
                distance_z_bottom,
            ])
        normals_to_add = []
        for (name, dist, normal) in dist_pairs:
            if dist == min_dist:
                normals_to_add.append((name, normal))
        final_n_x = 0
        final_n_y = 0
        final_n_z = 0
        for (_,(x,y,z)) in normals_to_add:
            final_n_x += x
            final_n_y += y
            final_n_z += z
        final_normal = (final_n_x, final_n_y, final_n_z)
        #Trying something
        # final_normal = (final_n_x * -1, final_n_y * -1, final_n_z * -1)
        if self.DEBUG: print('----------------')
        if self.DEBUG: print(f"coordinates:{coordinates}")
        if self.DEBUG: print(f"normals_to_add:{normals_to_add}")
        if self.DEBUG: print(f"final_normal:{final_normal}")
        if self.DEBUG: print(f"min_dist:{min_dist}")
        # d_0 is 4 to test (Basically how fast to deteriorate speed when approaching boundary)
        ramp_fn_arg = min_dist / self.D_0
        return final_normal, self.ramp_function(ramp_fn_arg)

    def ramp_function(self, r):
        """
        ramp function from the insect flight paper
        """
        if r > 1:
            return 1
        return ((3.0/8.0) * (r**5)) - ((10.0/8.0) * (r ** 3)) + ((15.0/8.0) * r)


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
