
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation
import scipy.ndimage.measurements as spmeas
import scipy.interpolate as spint
import numba
import time
import tqdm

matplotlib.rcParams['figure.dpi'] = 200

file_location = 'trap 20 mW_0.tif'
im = Image.open(file_location, 'r')

def get_frame(i):
    try:
        im.seek(i)
        image = np.array(im)[1:-1]
        return image
    except:
        raise AssertionError ('Fault in image retrieval')


def noise_calculator(image):
    """
    function returns the threshold noise_value from an image in the tif image stack

    image is an image directly from the tif image stack

    """
    median_intensity    = np.mean(image)
    distance_to_median  = np.abs(image - median_intensity)
    half_max_distance   = (np.max(image) - median_intensity)/2
    return half_max_distance, median_intensity


def remove_noise(image, noise, median_intensity):
    denoised_image = image - (median_intensity + noise)
    denoised_image[denoised_image < 0] = 0
    return denoised_image


def center_of_mass(image):
    total_mass = np.sum(image)
    if total_mass == 0:
        return 0, 0

    else:
        return spmeas.center_of_mass(image) #[YCOM,XCOM]


class QI_Tracker:
    # -------------------------------------------------------------------------------------------
    # QI Tracker Class Object.
    # For more information j.h.h.vandergronde@student.tudelft.nl
    # -------------------------------------------------------------------------------------------
    # CLASS ATTRIBUTES WHICH WILL BE STORED IN THE OBJECT.
    # IT CAN SIMPLY BE CALLED WITH self.attr INSIDE THE CLASS FUNCTIONS
    # OUTSIDE IT CAN ALSO BE CALLED, BUT NOW FROM THE OBJECT NAME (SEE EXAMPLE).
    # WHEN INITIALIZING THE OBJECT THEY CAN BE OVERWRITTEN BY SETTING THEM AS KEYWORD ARGUMENTS.
    # -------------------------------------------------------------------------------------------
    # numpy ndarray - contains the (first) image data
    image = 0
    # float - over-sampling of radial bins
    radialoversampling = 2.0
    # float - over-sampling of angular spokes
    angularoversampling = 0.7
    # float - minimum radius of polar grid
    minradius = 0.0
    # float - maxiumum radius of polar grid
    maxradius = 0.0
    # By default 2.5, but should be able to be changed in the init function as kwarg.
    max_radius_denom = 2.5
    # integer - no. of iterations
    iterations = 10
    # integer - no. of spokes per quadrant
    spokesnoperquad = 0
    # integer - no. of radial bins
    radbinsno = 0
    # numpy ndarray - linear space containg radii
    radbins = 0
    # numpy ndarray - linear space containing angles
    angles = 0
    # float - angilar step size
    angularstep = 0
    # numpy ndarray - grid containing angles
    argsgrid = 0
    # numpy ndarray - grid containing radii
    radiigrid = 0
    # numpy ndarray - x coordinates from the sampled polar grid
    X0samplinggrid = 0
    # numpy ndarray - y coordinates from the sampled polar grid
    Y0samplinggrid = 0

    # This function initialize the QI_Tracker object which is called with:
    # var = QI_Tracker(image)
    def __init__(self, im, **kwargs):
        """
        Initializes the QI tracker class and returns the object.

        Arguments:
        im -- Numpy 2D array of the (first) image

        Keyword arguments:
        radialoversampling
        angularoversampling
        minradius
        maxradius
        iterations
        ... and many other other attributes that are defined in the class.

        """
        assert 'np' in globals(), "numpy must be imported at the beginning of the file as np."
        assert 'plt' in globals(), "matplotlib.pyplot must be imported at the beginning of the file as plt."
        assert type(im) is list or type(im) is np.ndarray, "Image must be of type list or ndarray"
        assert len(im) > 0, "Image cannot be empty"

        # convert to numpy array if list is given.
        if type(im) is not np.ndarray:
            im = np.array(im)

        self.image = im

        # Override the default class attributes from the keyword arguments
        # Condition: Only if they exist in the class.
        for arg, val in kwargs.items():
            if arg in dir(self): setattr(self, arg, val)

        # -------------------------------------------------------------------------------------------
        # Creating the polar grid.
        # -------------------------------------------------------------------------------------------
        # Define the max-radius of the polar grid.
        self.maxradius = np.min(self.image.shape)/self.max_radius_denom
        # The no. of radial bins is defined as (r_max - r_min) x over-sampling.
        # int(...) <--> Solves warning cannot safely be interpretated as integer.
        self.radbinsno = int((self.maxradius - self.minradius) *self.radialoversampling)
        # Generate a linear space of radii, with the sampling given by radbinsno.
        self.radbins = np.linspace(self.minradius, self.maxradius, self.radbinsno)
        # The no. of spokes per quadrant is defined as .5πr_max x over-sampling (in this case under-sampling)
        # int(...) <--> Solves warning cannot safely be interpretated as integer.
        self.spokesnoperquad = int(np.ceil( (1/2) *np.pi *self.maxradius *self.angularoversampling))
        # From the no. of spokes per quadrant compute the angles in an array with a linear space.
        # Start at -π/4 and end at the same location 7π/4.
        # The total number of points then becomes 4 times the no. of spokes per quadrant +1 (including zero)
        self.angles = np.linspace(-(1/4)*np.pi,(7/4)*np.pi, 4*self.spokesnoperquad +1)
        # Define the angular step size, can also with self.angles[1] - self.angles[0]
        self.angularstep = np.pi/(2*self.spokesnoperquad)
        # Center the angles.
        self.angles = self.angles[0:-1] + self.angularstep/2
        # Generate a 2D grid containing the angles (args) and radii.
        self.argsgrid, self.radiigrid = np.meshgrid(self.angles, self.radbins)
        # Create X,Y coords from the polar grid.
        self.X0samplinggrid = self.radiigrid*np.cos(self.argsgrid)
        self.Y0samplinggrid = self.radiigrid*np.sin(self.argsgrid)
        return None

    def show_grid(self):
        """
        Plots the polar grid
        """
        plt.clf()
        fig = plt.figure(figsize=(8,8), dpi=100)
        ext = [
            np.amin(self.angles), # x-min
            np.amax(self.angles), # x-max
            np.amin(self.radbins), # y-min
            np.amax(self.radbins) # y-max
        ]
        plt.subplot(121)
        plt.imshow(self.X0samplinggrid, extent=ext, aspect='auto', origin='center')
        plt.xlabel('angular bins')
        plt.ylabel('radial bins')
        plt.subplot(122)
        plt.imshow(self.Y0samplinggrid, extent=ext, aspect='auto', origin='center')
        plt.xlabel('angular bins')
        plt.ylabel('radial bins')
        plt.tight_layout()
        plt.show()

    # This function corresponds to the main function TrackXY_by_QI in the mathlab file
    def track_xy(im, QI, sho, xm, ym):
        """
        Tracks the image

        Arguments:


        Keyword arguments:
        """
        return None


def subpix_step(d):
    #this function performs a subpixel step by parabolic fitting
    hf = 3
    ld = len(d)
    xs = np.array(range(ld))
    x = np.argmax(d)
    lo = int(np.amax([x-hf,0])) #Cropping (must be atleast 0 to slice an array)
    hi = int(np.amin([x+hf, ld-1])) #Cropping (must be max ld-1 to slice an array
    ys = d[lo:hi]
    xs = xs[lo:hi]
    prms = np.polyfit(xs, ys, 2)
    temp1 = (prms[1],prms[0], prms[1]/prms[0])
    return -prms[1]/(2*prms[0])


def SymCenter(prf):
    #this function find the symmetry center of an array.
    #print(prf)
    mp = np.nanmean( prf )
    #mp = np.where(np.isnan(prf), mp, prf)   #If a NaN value is found, replace it with mp
    fw = prf - mp  #forward
    rv = np.flip(prf)- mp #reverse
    d = np.real(np.fft.ifft(np.fft.fft(fw)*np.conjugate(np.fft.fft(rv))))
    d = np.fft.fftshift(d).T  #Swap first and second half
    #[val, x] = np.max(d)
    return (subpix_step(d)+len(prf)/2)/2


qi_tracker = QI_Tracker(get_frame(0))
#qi_tracker.show_grid()


YCOM , XCOM = center_of_mass(get_frame(0))

xnw = xol = XCOM
ynw = yol = YCOM

error_x = np.zeros(qi_tracker.iterations+1)

def track_xy(im, QI, sho, xnw, ynw):
    prequit = False
    xol, yol = xnw, ynw
    for ii in range(0, qi_tracker.iterations+1):
        if not prequit:

            x_samplinggrid = np.sort((qi_tracker.X0samplinggrid + xnw)[:,0])
            y_samplinggrid = np.sort((qi_tracker.Y0samplinggrid + ynw)[:,0])

            x_arrays = np.arange( im.shape[0] )
            y_arrays = np.arange( im.shape[1] )

            interp_func = spint.RectBivariateSpline(x_arrays, y_arrays, im)

            all_profiles = interp_func(x_samplinggrid, y_samplinggrid)

            (aa,rara) = all_profiles.shape

            number_of_spokes = int(aa/4)

            qi_profiles = np.zeros((4,rara))
            qi_profiles[0,:] = np.nanmean( all_profiles[0:number_of_spokes+1,:], axis=0 ) #EAST
            qi_profiles[1,:] = np.nanmean( all_profiles[number_of_spokes+1:2*number_of_spokes+1,:],axis = 0 ) #NORTH
            qi_profiles[2,:] = np.nanmean( all_profiles[2*number_of_spokes+1:3*number_of_spokes+1,:], axis=0 ) #WEST
            qi_profiles[3,:] = np.nanmean( all_profiles[3*number_of_spokes+1:4*number_of_spokes+1,:], axis=0 ) #SOUTH

            qi_horizontal = np.concatenate(( [np.flip( qi_profiles[2,:] )], [ np.flip( qi_profiles[0,:] ) ]), axis=None )
            qi_vertical   = np.concatenate( ([np.flip( qi_profiles[3,:] )], [ np.flip( qi_profiles[1,:] ) ]), axis=None )

            fudge_factor = np.pi/2

            sym_center_qi_hor = SymCenter( qi_horizontal )
            sym_center_qi_ver = SymCenter( qi_vertical )

            xnw = -(( len(qi_horizontal)/2 - sym_center_qi_hor) + 0.5 )/qi_tracker.radialoversampling/fudge_factor + xol
            ynw = -(( len( qi_vertical )/2 - sym_center_qi_ver) + 0.5 )/qi_tracker.radialoversampling/fudge_factor + yol

            error_x[ii] = (xnw - xol)**2 + (ynw - yol)**2

            if np.isnan(ynw) or np.isnan(xnw):
                prequit = True
        else:
            prequit = True
            ynw, xnw = yol, xol
    if np.isnan(xnw) or np.isnan(ynw):
        xnw, ynw = xol, yol
    XQI, YQI = xnw, ynw

    return XQI, YQI

start = time.time()

frame = get_frame(53)
noise, median_intensity = noise_calculator(frame)
cleaned_image = remove_noise(frame, noise, median_intensity)

COM = center_of_mass(cleaned_image)


end = time.time()
elapsed = end-start

print("Computation took: {:1f} seconds".format(elapsed))

plt.imshow(cleaned_image)
plt.plot(COM[1],COM[0], marker='1', color='red')


num = im.n_frames

show = 0



Xest, Yest = np.asarray([xnw]), np.asarray([ynw])

color = np.asarray([])

pbar = tqdm.tqdm(total=num)

for i in range(1,num):
    current_frame = get_frame(i)

    xnw, ynw = track_xy(current_frame, qi_tracker, show, Xest[i-1], Yest[i-1])

    if np.abs(Xest[i-1]-xnw) > 10 or np.abs(Yest[i-1]-ynw) > 10:
        noise, median_intensity = noise_calculator(current_frame)
        cleaned_image = remove_noise(current_frame, noise, median_intensity)

        color_cur = 'red'

        ynw, xnw = center_of_mass(cleaned_image)

    else:
        color_cur = 'white'

    #print(xnw, ynw)

    Xest = np.append(Xest, xnw)
    Yest = np.append(Yest, ynw)
    color = np.append(color, color_cur)
    pbar.update(1)

pbar.close()
plt.close()

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,8))
line, = ax[0].plot(0, 0, marker="1", color="black")
txt = ax[0].text(0,0,'o', color='white')
ani_im = ax[0].imshow(get_frame(0))

ax[0].set_ylim(0,get_frame(0).shape[0])
ax[0].set_xlim(0,get_frame(0).shape[1])

linex, = ax[1].plot(0,0, label='Xest')
liney, = ax[1].plot(0,0, label='Yest')


ax[1].set_ylim(0,get_frame(0).shape[0])
ax[1].set_xlim(0,num)

ax[1].legend()



def animation_frame(iterant):
    ani_im.set_array(get_frame(iterant))

    #fig.set_title(str(iterant))

    line.set_xdata(Xest[iterant])
    line.set_ydata(Yest[iterant])
    txt.set_position((Xest[iterant],Yest[iterant]+10))
    txt.set_text( ( str(Xest[iterant])[0:3],str(Yest[iterant])[0:3] ) )
    txt.set_color(color[iterant])

    linex.set_xdata(np.arange(iterant))
    liney.set_xdata(np.arange(iterant))
    linex.set_ydata(Xest[0:iterant])
    liney.set_ydata(Yest[0:iterant])
    return line,

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(num), interval=1)
plt.show()
