
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation
import scipy.ndimage.measurements as spmeas
import scipy.interpolate as spint
from numba import jit
import time
import tqdm
from scipy.interpolate.ndgriddata import griddata
from scipy.interpolate.interpolate import interp2d
from scipy import ndimage

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

"""
def noise_calculator(image):

    function returns the threshold noise_value from an image in the tif image stack

    image is an image directly from the tif image stack


    median_intensity    = np.mean(image)
    distance_to_median  = np.abs(image - median_intensity)
    half_max_distance   = (np.max(image) - median_intensity)/2
    return half_max_distance, median_intensity


def remove_noise(image, noise, median_intensity):
    denoised_image = image - (median_intensity + noise)
    denoised_image[denoised_image < 0] = 0
    return denoised_image

"""
def center_of_mass(image):
    total_mass = np.sum(image)
    if total_mass == 0:
        return 0, 0

    else:
        return spmeas.center_of_mass(image) #[YCOM,XCOM]









def calculateNoise(I):
    show = 1
    if 1:
        # Delete background intensity
        In = I.flatten()           # Most Python functions don't work for a 2D array, so In has to be flattened.
        Im = np.sort(In)            # A sorted list of I, with background intensity.
        plt.figure()
        plt.title('Sorted intensity values')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.plot(Im)                # In this plot it can be seen that Im has been correctly sorted.
        med = np.median(Im)         # The median of IM can be calculated from the flattened sorted array Im.
        Imed = abs(I - med)


        # Calculate the noise intensity
        ilst = np.sort(Imed)                    # To create a sorted list of all present intensities.
        ilstflat = np.sort(ilst.flatten())      # ilst has to be flattened to find the max value with Python.
        ilst_t = ilst.transpose()
        ilstflat_t = np.sort(ilst_t.flatten())  # With the normal flatten function the rows are put bewelow eacht other, but we want to put the columns next to each other. Thus the transposed version of ilst is needed.
        b = len(Im)
        x = np.arange(1,b+1)
        Ihalf = (max(In)-med)/2
        # Ihalf=max(ilstflat)/2

        # Make sure that the flattened ilst is also sorted, otherwise the wrong index is found.
        xIhalf_value = list(filter(lambda i: i > Ihalf, ilstflat_t))[0]         # This function gives the first value where ilst > Ihalf. With this value it can be checked if the position calculated in the next line is correct.
        xIhalf = next(x for x, val in enumerate(ilstflat_t)
                                      if val > Ihalf)                           # Here the position of the first index with a value bigger than Ihalf is found.

        range_up=int(np.floor(b/2))
        coeff1= np.polyfit(x[0:range_up], ilstflat_t[0:range_up],1)
        coeff2 = np.polyfit(x[int(xIhalf):b], ilstflat_t[int(xIhalf):b], 1)     # Makes a linear fit to the highest half of all intensities in 'ilist'.
        if show:
            plt.figure()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Noise intensity')
            plt.grid()
            plt.plot(x,np.sort(ilstflat_t),'b-')

        # The rightmost intersection is the chosen noise intensity which will be removed from all images.
        fit1 = np.polyval(coeff1,x)                                     # Returns an array of all values of the fitted line for every x coordinate.
        fit2 = np.polyval(coeff2,x)
        xcross=round((coeff1[1]-coeff2[1])/(coeff2[0]-coeff1[0]))
        ycross= np.polyval(coeff1,xcross)
        fit=np.concatenate([fit1[0:int(xcross-1)],fit2[int(xcross-1):]])
        if show:
            plt.plot(x,np.sort(ilstflat_t),'b-', x,fit,'r'), plt.ylim([0,max(ilstflat)])
        scale=max(x)/max(ilstflat)
        dist=np.sqrt((x-xcross)**2+(ycross-ilstflat_t*scale)**2)        # Calculates the difference between the true intensities and the values of crossing fitted lines.
        xmindist = next(x for x, val in enumerate(dist)
                                      if val == min(dist))              # Calculates the x value of the rightmost intersection of the fitted line and the curve of all true intensities.
        threshold=ilstflat_t[(xmindist-1)]+np.std(ilst)                 # Calculates the intensity, which belongs to the above calculated x value. This equals the intensity of the noise which should be delete.
        if show:
            plt.plot(x,np.sort(ilstflat_t),'b-', x,fit,'k', xcross,ycross,'kx', xmindist, ilstflat_t[xmindist],'mo', [0,max(x)],[threshold,threshold],'r')
        return threshold

#step5: subtract the noise from the image array
def removeNoise(I, Inoise):
    #This is done in two steps: first the background intensity is removed:
    med = np.median(I) #the median of all values in I is taken as the background intensity
    Ib = I - med
    #Then the calculated noise is removed.
    Ic = Ib - Inoise
    #Now make sure all values are positive.
    Ic[Ic<0] = 0
    if show:
        plt.figure()
        plt.subplot(311)
        plt.title("frame with noise and background")
        plt.imshow(I)
        plt.subplot(312)
        plt.title("frame with noise without background")
        plt.imshow(Ib)
        plt.subplot(313)
        plt.title("frame without noise")
        plt.imshow(Ic)
        plt.tight_layout()
        plt.show()
    return Ic






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
        self.X0samplinggrid = ( self.radiigrid*np.cos(self.argsgrid) ).T
        self.Y0samplinggrid = ( self.radiigrid*np.sin(self.argsgrid) ).T
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
    prf_ = np.where(np.isnan(prf), mp, prf)   #If a NaN value is found, replace it with mp
    fw = prf_ - np.nanmean(prf_)  #forward

    rv = np.flip(prf_) - np.nanmean(prf_) #reverse
    d = np.real(np.fft.ifft(np.fft.fft(fw)*np.conjugate(np.fft.fft(rv))))
    d = np.fft.fftshift(d).T  #Swap first and second half
    #[val, x] = np.max(d)
    return (subpix_step(d)+len(prf_)/2)/2


qi_tracker = QI_Tracker(get_frame(0))
#qi_tracker.show_grid()


YCOM , XCOM = center_of_mass(get_frame(0))

xnw = xol = XCOM
ynw = yol = YCOM

error_x = np.zeros(qi_tracker.iterations+1)

@jit
def get_eval_coords(xsampling,ysampling):
    x_points = xsampling.flatten()
    y_points = ysampling.flatten()
    return x_points, y_points

@jit
def rebuild_grid(values, shape):
    return values.reshape(shape)



def track_xy(im, QI, sho, xnw, ynw):
    prequit = False
    xol, yol = xnw, ynw
    #xnw = 63.1812
    #ynw = 59.2335

    for ii in range(0, qi_tracker.iterations+1):
        if not prequit:
            x_samplinggrid = ( qi_tracker.X0samplinggrid + xnw )
            y_samplinggrid = ( qi_tracker.Y0samplinggrid + ynw )
            """
            ----------------------------------------------------------------------------------
            This function doesn't yield usable results since our samplinggrid is not an evenly spaced rectangular grid
            ----------------------------------------------------------------------------------
            """

            x_samplinggrid = (qi_tracker.X0samplinggrid + xnw)
            y_samplinggrid = (qi_tracker.Y0samplinggrid + ynw)

            x_arrays = np.arange( im.shape[0] )
            y_arrays = np.arange( im.shape[1] )

            eval_x, eval_y = get_eval_coords(x_samplinggrid, y_samplinggrid)

            interp_func = spint.RectBivariateSpline(x_arrays, y_arrays, im)

            evald_values = interp_func.ev(eval_x, eval_y)

            all_profiles = rebuild_grid(evald_values, x_samplinggrid.shape)

            """
            ---------------------------------------------------------------------------------------
            This function cant work since it needs an array of points of all (x,y) coordinates we want to evaluate at in ascending order which is something we cant do since we cant sort our points in such a way that the (x,y) coordinates stay linked.
            ---------------------------------------------------------------------------------------
            """

            """


            points_mesh = np.meshgrid(x_points, y_points)
            points = (points_mesh[0].flatten(), points_mesh[1].flatten())
            grid = (x_samplinggrid, y_samplinggrid)

            all_profiles = griddata(points, im.flatten(),grid, method="linear")
            """

            """
            This function wont work since it only works with 1-D arrays for x and y which would be easy to make if the grid was rectangular..., but it isnt so it would take multiple nested for loops which will be slow and error prone. And even then we get an array of outputs that assumes a rectangular grid and would therefore be distorted
            """
            """

            #x, y = get_eval_coords(x_samplinggrid, y_samplinggrid)

            interp_func = interp2d(x_points,y_points,im)

            #evald_values = interp_func(x,y)

            all_profiles = np.zeros(x_samplinggrid.shape)

            for i in range(0,x_samplinggrid.shape[1]): #looping over x
                for j in range(0,x_samplinggrid.shape[0]): #looping over y
                    all_profiles[j][i] = interp_func(x_samplinggrid[j][i],y_samplinggrid[j][i])

            #all_profiles = evald_values.reshape(x_samplinggrid.shape)
            """

            """
            This function is promising but currently yields some strange results which we do not yet fully understand and can't really be tested until te rest works.
            """

            """
            all_profiles = np.asarray(x_samplinggrid.shape)

            all_profiles = ndimage.map_coordinates(im, [x_samplinggrid, y_samplinggrid], order=1, cval=0.0, prefilter=False)
            """

            (aa,rara) = x_samplinggrid.shape

            number_of_spokes = round(aa/4)

            qi_profiles = np.zeros((4,rara))
            qi_profiles[0,:] = np.nanmean( all_profiles[0:number_of_spokes+1,:], axis=0 ) #EAST
            qi_profiles[1,:] = np.nanmean( all_profiles[number_of_spokes+1:2*number_of_spokes+1,:],axis = 0 ) #NORTH
            qi_profiles[2,:] = np.nanmean( all_profiles[2*number_of_spokes+1:3*number_of_spokes+1,:], axis=0 ) #WEST
            qi_profiles[3,:] = np.nanmean( all_profiles[3*number_of_spokes+1:4*number_of_spokes+1,:], axis=0 ) #SOUTH

            qi_vertical = np.concatenate(( [np.flip( qi_profiles[2,:] )], [  qi_profiles[1,:]  ]), axis=None )
            qi_horizontal =  np.concatenate( ([np.flip( qi_profiles[3,:] )], [  qi_profiles[0,:]  ]), axis=None )

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
            xnw, ynw = xol, yol
    if np.isnan(xnw) or np.isnan(ynw):
        xnw, ynw = xol, yol
    XQI, YQI = xnw, ynw

    return XQI, YQI


#COM = center_of_mass(cleaned_image)



num = im.n_frames

show = 0



Xest, Yest = np.asarray([xnw]), np.asarray([ynw])

color = np.asarray([])


noise = calculateNoise(get_frame(0))

pbar = tqdm.tqdm(total=num)
for i in range(0,num):
    current_frame = get_frame(i)
    """
    noise, median_intensity = noise_calculator(current_frame)
    cleaned_image = remove_noise(current_frame, noise, median_intensity)
    current_im = cleaned_image
    """


    cleaned_im = removeNoise(current_frame, noise)


    xnw, ynw = track_xy(cleaned_im, qi_tracker, show, Xest[i-1], Yest[i-1])
    """
    if np.abs(Xest[i-1]-xnw) > 10 or np.abs(Yest[i-1]-ynw) > 10:
        noise, median_intensity = noise_calculator(current_frame)
        cleaned_image = remove_noise(current_frame, noise, median_intensity)

        color_cur = 'red'

        ynw, xnw = center_of_mass(cleaned_image)

    else:
        color_cur = 'white'
    """
    #print(xnw, ynw)
    color_cur = 'white'
    Xest = np.append(Xest, xnw)
    Yest = np.append(Yest, ynw)
    color = np.append(color, color_cur)
    pbar.update(1)

pbar.close()
plt.close()

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(4,5))
line, = ax[0].plot(0, 0, marker="1", color="black")
txt = ax[0].text(0,0,'o', color='white')
ani_im = ax[0].imshow(get_frame(0), origin='lower')

ax[0].set_ylim(0,get_frame(0).shape[0])
ax[0].set_xlim(0,get_frame(0).shape[1])

linex, = ax[1].plot(0,0, label='Xest')
liney, = ax[1].plot(0,0, label='Yest')


ax[1].set_ylim(0,get_frame(0).shape[0])
ax[1].set_xlim(0,num)

ax[1].legend()



def animation_frame(iterant):
    ani_im.set_array(removeNoise(get_frame(iterant),noise))

    #fig.set_title(str(iterant))

    line.set_xdata(Yest[iterant])
    line.set_ydata(Xest[iterant])
    txt.set_position((Yest[iterant],Xest[iterant]+10))
    txt.set_text( ( str(Xest[iterant])[0:3],str(Yest[iterant])[0:3] ) )
    txt.set_color(color[iterant])

    linex.set_xdata(np.arange(iterant))
    liney.set_xdata(np.arange(iterant))
    linex.set_ydata(Yest[0:iterant])
    liney.set_ydata(Xest[0:iterant])
    return line,

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(num), interval=1)
plt.show()

plt.close()
plt.subplot(211)
plt.plot(Xest, label='y')
plt.ylim((np.min(Xest)-1,np.max(Xest)+1))
plt.legend()

plt.subplot(212)
plt.plot(Yest, label='x')
plt.ylim((np.min(Yest)-1,np.max(Yest)+1))
plt.legend()
plt.show()

plt.scatter(Yest,Xest)
plt.ylim((np.min(Xest),np.max(Xest)))
plt.xlim((np.min(Yest),np.max(Yest)))
plt.show()

TEST = 0