import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import j1

import generateProjection as gP
from stig import constants

util_img = 'UTIL'
sim_img = 'SIM'

def magnitude_to_brightness(magnitude, saturation_magnitude=2.5):
    """
    Convert stellar magnitude to relative brightness with saturation
    
    Parameters:
    magnitude (float): Star magnitude
    saturation_magnitude (float): Magnitude at which the star reaches full saturation
    
    Returns:
    float: Normalized brightness in [0,1] range
    """
    # If magnitude is less than or equal to saturation magnitude, return full brightness
    if magnitude <= saturation_magnitude:
        return 1.0
    
    # Otherwise, calculate relative brightness using magnitude formula
    # Normalized relative to the saturation magnitude
    # The factor 0.4 is the standard factor for magnitude calculation
    relative_brightness = 10**(-0.4 * (magnitude - saturation_magnitude))
    
    return relative_brightness

def add_noise(image, 
              full_well=70000,
              dark_current_mean=100, 
              dark_current_std=5,
              readout_std=50,
              num_hot_pixels=0,
              hot_pixel_value=35000,
              stray_light=6000):
    """
    Add realistic noise to a simulated astronomical image.
    """
    # Convert image from [0,1] range to electrons
    electrons = image * full_well
    
    # 1. Add dark current noise (Gaussian)
    dark_noise = np.random.normal(dark_current_mean, dark_current_std, size=electrons.shape)
    electrons += dark_noise
    
    # 2. Add shot noise (Poisson)
    # Clip to zero first to avoid issues with negative values
    electrons = np.clip(electrons, 0, None)
    # Generate Poisson samples for each pixel
    poisson_samples = np.random.poisson(electrons)
    
    # Convert back to float64 before adding read noise
    electrons = poisson_samples.astype(np.float64)
    
    # 3. Add read noise (Gaussian)
    read_noise = np.random.normal(0, readout_std, size=electrons.shape)
    electrons += read_noise
    
    # 4. Add stray light
    electrons += stray_light
    
    # 5. Add hot/dead pixels - making sure they're truly single pixels
    if num_hot_pixels > 0:
        # Get the image dimensions
        height, width = electrons.shape
        
        # Generate random coordinates for hot/dead pixels
        y_coords = np.random.randint(0, height, num_hot_pixels)
        x_coords = np.random.randint(0, width, num_hot_pixels)
        
        # Set each pixel to either hot or dead
        for i in range(num_hot_pixels):
            # 50% chance of hot pixel, 50% chance of dead pixel
            electrons[y_coords[i], x_coords[i]] = hot_pixel_value if np.random.random() < 0.5 else 0
    
    # Clip values to valid electron range and convert back to [0,1]
    electrons = np.clip(electrons, 0, full_well)
    noisy_image = electrons / full_well
    
    return noisy_image

def generate_airy_disk(x, y, center_x, center_y, brightness=1.0, overexpose=5, rad=10):
    """
    Generate an Airy disk at the specified position with specified brightness
    
    Parameters:
    x, y (numpy arrays): Coordinate grids
    center_x, center_y (float): Center position of the star
    brightness (float): Star brightness factor [0,1]
    overexpose (int): Overexposure factor
    rad (float): Radius of the Airy disk
    
    Returns:
    numpy array: Airy disk pattern
    """
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    z_raw = (overexpose+1) * 4 * (j1(r)/r)**2
    # Replace NaN at r=0 with the limit value
    z_raw[np.isnan(z_raw)] = (overexpose+1) * 4
    
    # Apply brightness scaling
    z_scaled = z_raw * brightness
    
    return np.clip(z_scaled, 0, 1)

def gen_airy(overExpose:int=5, rad:float=10):
    """Original gen_airy function, kept for compatibility"""
    # create meshgrid of spaces 
    r = lambda x,y: np.sqrt(x**2 + y**2)
    x = np.linspace(-1*rad, rad, 1000)
    y = np.linspace(-1*rad, rad, 1000)

    xM, yM = np.meshgrid(x, y)
    d = r(xM, yM)

    # calculate brightness based on Airy Disk
    z_raw = (overExpose+1) * 4 * (j1(d)/d)**2
    zM = np.clip(z_raw, 0, 1)

    return xM, yM, zM

def plot_star(star:pd.Series, axes, xM, yM, zM):
    """
    Original plot_star function, modified to use magnitude
    """
    # Get magnitude if available
    brightness = 1.0
    if 'v_magnitude' in star:
        brightness = magnitude_to_brightness(star['v_magnitude'])
    else:
        brightness = magnitude_to_brightness(5.0)
    
    # Scale the zM values by the brightness factor
    scaled_zM = zM * brightness
    
    axes.pcolormesh(xM + star['IMG_X'], yM + star['IMG_Y'], scaled_zM, cmap='gray')
    return

def gen_bkgd(axes, img_wd:int, img_ht:int):
    """Original gen_bkgd function, kept for compatibility"""
    # create meshgrid
    xB = np.linspace(0, img_wd-1, 10)
    yB = np.linspace(0, img_ht-1, 10)

    xB, yB = np.meshgrid(xB, yB)

    # set brightness to 0 (black)
    zB = np.zeros((np.size(xB[0]), np.size(yB[0])))

    # plot on axes
    axes.pcolormesh(xB, yB, zB, cmap='gray')
    return

def setup_util_image(img_wd:int, img_ht:int):
    """Original setup_util_image function, kept for compatibility"""
    fig = plt.figure()
    ax = plt.axes()
    
    fig.add_axes(ax)

    ax.set_aspect('equal')
    ax.set_xlim(0, img_wd)
    ax.set_ylim(0, img_ht)
    
    return fig, ax

def setup_sim_image(img_wd:int, img_ht:int):
    """Original setup_sim_image function, kept for compatibility"""
    fig = plt.figure(frameon=False)
    iw = img_wd/100
    ih = img_ht/100

    fig.set_size_inches(iw, ih, forward=False)
    ax = plt.Axes(fig, [0.,0.,1.,1.])

    ax.set_aspect('equal')
    ax.set_xlim(0, img_wd)
    ax.set_ylim(0, img_ht)
    
    ax.set_axis_off()
    fig.add_axes(ax)

    gen_bkgd(ax, img_wd, img_ht)

    return fig, ax

def create_image_with_noise(starlist:pd.DataFrame, img_wd:int, img_ht:int, ra:float, dec:float, roll:float, 
                           dark_current_mean=100, dark_current_std=5, readout_std=50, 
                           num_hot_pixels=0, hot_pixel_value=35000, full_well=70000, stray_light=6000,
                           outfp:str="./", showPlot:bool=False):
    """
    Create a simulated star image with realistic noise
    """
    # Create a grid for the full image
    # Make sure we're using integer dimensions
    img_wd, img_ht = int(img_wd), int(img_ht)
    
    # Start with a black image
    image = np.zeros((img_ht, img_wd))
    
    # Track star info for debugging
    star_info = []
    
    # Add each star to the image
    for _, star in starlist.iterrows():
        # Get star position
        star_x, star_y = star['IMG_X'], star['IMG_Y']
        
        # Get star magnitude and convert to brightness
        magnitude = None
        if 'v_magnitude' in star:
            magnitude = star['v_magnitude']
        else:
            magnitude = 5.0  # Default magnitude if none provided
            
        brightness = magnitude_to_brightness(magnitude)
        
        # Save star info for logging
        star_info.append(f"{star_x}\t{star_y}\tMag:{magnitude:.2f}\tBrightness:{brightness:.3f}")
        print(f"{star_x}\t{star_y}\tMag:{magnitude:.2f}\tBrightness:{brightness:.3f}")
        
        # Define the region around the star to compute the Airy disk
        # Use a smaller radius for efficiency
        rad = 20  # pixels
        x_min = max(0, int(star_x - rad))
        x_max = min(img_wd, int(star_x + rad + 1))
        y_min = max(0, int(star_y - rad))
        y_max = min(img_ht, int(star_y + rad + 1))
        
        # Create localized grid for this star
        x_local = np.arange(x_min, x_max)
        y_local = np.arange(y_min, y_max)
        x_grid, y_grid = np.meshgrid(x_local, y_local)
        
        # Generate Airy disk for this star with appropriate brightness
        airy = generate_airy_disk(
            x_grid, y_grid, star_x, star_y, 
            brightness=brightness,
            overexpose=5, rad=10
        )
        
        # Add the star to the main image
        image[y_min:y_max, x_min:x_max] += airy
    
    # Clip the image values to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Create a copy for the utility image without noise
    util_image = image.copy()
    
    # Add noise to the simulation image
    noisy_image = add_noise(
        image,
        full_well=full_well,
        dark_current_mean=dark_current_mean,
        dark_current_std=dark_current_std,
        readout_std=readout_std,
        num_hot_pixels=num_hot_pixels,
        hot_pixel_value=hot_pixel_value,
        stray_light=stray_light
    )
    
    # Generate filenames
    fname = '_{:.3f}_{:.3f}_{:.3f}.png'.format(ra, dec, roll)
    util_name = outfp + util_img + fname
    sim_name = outfp + sim_img + fname
    
    # Create and save the utility image (without noise)
    fig_util = plt.figure(figsize=(10, 10*img_ht/img_wd))
    ax_util = fig_util.add_subplot(111)
    ax_util.imshow(util_image, cmap='gray', origin='lower', interpolation='none')
    ax_util.set_title('({:.3f}, {:.2f}, {:.2f})'.format(ra, dec, roll))
    ax_util.set_aspect('equal')
    fig_util.savefig(util_name, dpi=img_wd/10)
    
    # Create and save the simulation image (with noise)
    fig_sim = plt.figure(figsize=(10, 10*img_ht/img_wd), frameon=False)
    ax_sim = plt.Axes(fig_sim, [0., 0., 1., 1.])
    ax_sim.set_axis_off()
    fig_sim.add_axes(ax_sim)
    
    # Use interpolation='none' to ensure hot pixels remain as single pixels
    ax_sim.imshow(noisy_image, cmap='gray', origin='lower', interpolation='none')
    
    fig_sim.savefig(sim_name, dpi=img_wd/10)
    
    # Show plot if requested
    if showPlot:
        plt.figure(figsize=(18, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(util_image, cmap='gray', origin='lower', interpolation='none')
        plt.title('Original Image')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(noisy_image, cmap='gray', origin='lower', interpolation='none')
        plt.title('Noisy Image')
        plt.colorbar()
        
        # Print star info in the plot
        plt.figure(figsize=(10, 5))
        plt.axis('off')
        plt.text(0.1, 0.9, f"Star information (RA={ra:.2f}, DEC={dec:.2f}, ROLL={roll:.2f})", 
                 fontsize=12, fontweight='bold')
        
        for i, info in enumerate(star_info):
            plt.text(0.1, 0.8 - 0.05*i, info, fontsize=10)
        
        plt.show()
    
    plt.close(fig_util)
    plt.close(fig_sim)
    
    return noisy_image

def create_image(starlist:pd.DataFrame, img_wd:int, img_ht:int, ra:float, dec:float, roll:float,
                dark_current_mean=100, dark_current_std=5, readout_std=50, 
                num_hot_pixels=0, hot_pixel_value=35000, full_well=70000, stray_light=6000,
                outfp:str="./", showPlot:bool=False):
    """
    Original create_image function, updated to use the noise model
    """
    # If noise parameters are provided or we want to use magnitude scaling, use the new implementation
    if dark_current_mean > 0 or readout_std > 0 or num_hot_pixels > 0:
        return create_image_with_noise(
            starlist, img_wd, img_ht, ra, dec, roll,
            dark_current_mean, dark_current_std, readout_std,
            num_hot_pixels, hot_pixel_value, full_well, stray_light,
            outfp, showPlot
        )
    
    # Otherwise, use the original implementation for backward compatibility
    framed_fig, framed_ax = setup_util_image(img_wd=img_wd, img_ht=img_ht)
    sim_fig, sim_ax = setup_sim_image(img_wd=img_wd, img_ht=img_ht)

    ttl = '({}, {}, {})'.format(ra, dec, roll)
    framed_ax.set_title(ttl)

    # set figure name
    fname = '_{}_{}_{}.png'.format(ra, dec, roll)

    # store airy distribution
    xM, yM, zM = gen_airy()

    # iterate over stars
    for index, row in starlist.iterrows():
        # Get magnitude if available
        magnitude = None
        if 'MAGNITUDE' in row:
            magnitude = row['MAGNITUDE']
        elif 'MAG_V' in row:
            magnitude = row['MAG_V']
        else:
            magnitude = 5.0  # Default magnitude
            
        brightness = magnitude_to_brightness(magnitude)
        print(f"{row['IMG_X']}\t{row['IMG_Y']}\tMag:{magnitude:.2f}\tBrightness:{brightness:.3f}")

        # Use the modified plot_star function that considers magnitude
        plot_star(row, framed_ax, xM, yM, zM)
        plot_star(row, sim_ax, xM, yM, zM)
    
    mname = outfp + util_img + fname
    framed_fig.savefig(mname)
    plt.close(framed_fig)
    
    pname = outfp + sim_img + fname
    sim_fig.savefig(pname)
    plt.close(sim_fig)
    
    if showPlot:
        plt.close(sim_fig)
        plt.show()        

    return

def parse_arguments():
    parser = argparse.ArgumentParser(description='generates images of stars given coordinates and image dimensions',prefix_chars='@')
    
    parser.add_argument('@n', type=int, help='Number of pictures to create; Default: 1', default=1)
    parser.add_argument('@fp', type=str, help='dataframe pickle filepath (Single Image); Default: Random', default=None)
    parser.add_argument('@cam', type=str, help='Set camera config filepath; Default: Alvium', default=constants.DEFAULT_ALVIUM)
    parser.add_argument('@dname', type=str, help='folder where images are going; Default: _StarTrackerTestImages/simImages/', default=constants.SIM_IMAGES)
    parser.add_argument('@ra', type=float, help='Custom right ascension, in degrees', default=None)
    parser.add_argument('@dec', type=float, help='Custom declination, in degrees', default=None)
    parser.add_argument('@roll', type=float, help='Custom roll, in degrees', default=None)
    parser.add_argument('@plot', type=bool, help='Whether to plot or not', default=False)
    
    # Add new noise parameters
    parser.add_argument('@dark_mean', type=float, help='Mean dark current (electrons)', default=100)
    parser.add_argument('@dark_std', type=float, help='Dark current std dev (electrons)', default=5)
    parser.add_argument('@read_std', type=float, help='Readout noise std dev (electrons)', default=50)
    parser.add_argument('@hot_pixels', type=int, help='Number of hot/dead pixels', default=0)
    parser.add_argument('@hot_value', type=float, help='Value for hot pixels (electrons)', default=35000)
    parser.add_argument('@full_well', type=float, help='Full well capacity (electrons)', default=70000)
    parser.add_argument('@sat_mag', type=float, help='Saturation magnitude (2.5 or lower = full brightness)', default=2.5)
    parser.add_argument('@stray', type=float, help='Stray light (electrons)', default=6000)
    
    return parser.parse_args()

def generate_image(data=pd.DataFrame, camera:str=constants.DEFAULT_ALVIUM, 
                   direc:str=constants.SIM_IMAGES, plot:bool=False, 
                   ra:float=None, dec:float=None, roll:float=None,
                   dark_current_mean=100, dark_current_std=5,
                   readout_std=50, num_hot_pixels=0, hot_pixel_value=35000,
                   full_well=70000, stray_light=6000):
    """Generate a star image with the given parameters"""
    cfg = json.load(open(camera))

    img_wd = cfg['IMAGE_X']
    img_ht = cfg['IMAGE_Y']

    create_image(
        data, img_wd, img_ht, ra, dec, roll, 
        dark_current_mean=dark_current_mean,
        dark_current_std=dark_current_std,
        readout_std=readout_std,
        num_hot_pixels=num_hot_pixels,
        hot_pixel_value=hot_pixel_value,
        full_well=full_well,
        stray_light=stray_light,
        outfp=direc, 
        showPlot=plot
    )

if __name__ == '__main__':
    args = parse_arguments()

    if args.fp is not None:
        data = pd.read_pickle(args.fp)
        generate_image(
            data=data, camera=args.cam, direc=args.dname, plot=args.plot,
            dark_current_mean=args.dark_mean, dark_current_std=args.dark_std,
            readout_std=args.read_std, num_hot_pixels=args.hot_pixels,
            hot_pixel_value=args.hot_value, full_well=args.full_well, stray_light=args.stray
        )
    else:
        for i in range(args.n):
            ra = args.ra or random.uniform(0, 360)
            dec = args.dec or random.uniform(-90, 90)
            roll = args.roll or random.uniform(0, 360)
            
            random_data = gP.generate_projection(ra=ra, dec=dec, roll=roll, cfg_fp=args.cam, plot=False, save=args.dname)
            generate_image(
                data=random_data, camera=args.cam, direc=args.dname, plot=args.plot, 
                ra=ra, dec=dec, roll=roll,
                dark_current_mean=args.dark_mean, dark_current_std=args.dark_std,
                readout_std=args.read_std, num_hot_pixels=args.hot_pixels,
                hot_pixel_value=args.hot_value, full_well=args.full_well, stray_light=args.stray
            )
