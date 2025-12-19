#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 18:38:31 2025

@author: petershea
"""
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.ndimage import convolve


class persistence_characterization:
    
    def __init__(self):
        '''
        class
        '''  
        
        
        
    def cubefit(self,xarr,yarr):
        popt, pcov = curve_fit(lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, xarr, yarr)
        return popt, pcov
    
    

    def linfit(self,xarr,yarr):
        popt, pcov = curve_fit(lambda x, a, b: a*x + b, xarr, yarr)
        return popt, pcov
    
    
    
    def reduce_persistence_series(self,data_path,plot_save_path=None,data_save_path=None):
        '''
        data_path: absolute path to directory containing persistence characteriaztion data set
        '''  
        
        files = sorted(os.listdir(data_path))
        files = [f for f in files if f.endswith(".fits") or f.endswith(".fz")]
        self.imgs = files # write to global file
        light_frame = files.pop(0)
        dark_frames = files
        
        hdul_light = fits.open(light_frame)
        gain = hdul_light[1].header['EGAIN']
        self.light_exptime = hdul_light[1].header['EXPTIME']
        self.dark_exptime = fits.open(dark_frames[0])[1].header.get('EXPTIME',None)
        self.Obj = hdul_light[1].header.get("OBJECT", None)
        self.detector = hdul_light[1].header.get("INSTRUME", None)
        
        imgdata_light = np.copy(hdul_light[1].data)
        size = len(imgdata_light)
        xmin, xmax = 350, size - 350
        ymin, ymax = 200, size - 200
        imgdata_light = imgdata_light[ymin:ymax, xmin:xmax] * gain
        
        # Log fluence bins from light image
        finite_light_mask = np.isfinite(imgdata_light)
        
        # kernal for pixel convolution
        kernel = np.array([
            [0.0,1.0,0.0],
            [1.0,1.0,1.0],
            [0.0,1.0,0.0]])
        
        # Normalize the kernel to ensure it sums to 1 (or close)
        kernel /= np.sum(kernel)
    
        # Convolve to sum values in neighborhood
        sum_matrix = convolve(imgdata_light, kernel, mode='constant', cval=0.0)
        
        # Convolve a mask to count how many valid values contributed
        valid_counts = convolve(np.ones_like(imgdata_light), kernel, mode='constant', cval=0.0)
        
        # Element-wise division to get average
        avg_fluence_model = sum_matrix / valid_counts 
        avg_fluence_model = avg_fluence_model[finite_light_mask]
        
        self.nbins = 100 # determine number of fluence bins
        fluence_bins = np.logspace(np.log10(np.min(avg_fluence_model)), np.log10(np.max(avg_fluence_model)), self.nbins)
        bin_centers = np.sqrt(fluence_bins[:-1] * fluence_bins[1:])  # geometric mean
        
        # Get light frame timestamp
        light_time = hdul_light[1].header['JD'] # time of light exposure in JD
        
        # Saving Values
        persistence_matrix = [] # list to store persistence data
        dt_arr = np.array([]) # list to store time difference of each dark from light
        
        for dark_file in dark_frames:
            
            # Load and process dark same as light
            hdul_dark = fits.open(dark_file)
            imgdata_dark = np.copy(hdul_dark[1].data)[ymin:ymax, xmin:xmax] * gain
            imgdata_dark = imgdata_dark[finite_light_mask]
            
            # Get dt
            dark_time = hdul_dark[1].header['JD']
            dt = (dark_time - light_time) * 24 * 3600 # get time in seconds
            dt -= self.light_exptime # subtract light exposure so its time from end of ccd exposure
        
            if dt <= 0:
                continue
        
            dt_arr = np.append(dt_arr,dt) # append time
        
            # Bin persistence
            persistence = []
            for i in range(len(fluence_bins)-1):
                bin_mask = (avg_fluence_model >= fluence_bins[i]) & (avg_fluence_model < fluence_bins[i+1]) # determine pixels corresponding to fluence bins
                persistence.append(np.median(imgdata_dark[bin_mask]) if np.any(bin_mask) else np.nan) # append average persistence for binned pixels
        
            persistence_matrix.append(np.array(persistence) / hdul_dark[1].header['EXPTIME']) # divide by exposure time of dark to get e-/s
        
        persistence_matrix = np.array(persistence_matrix).T  # shape: (n_bins, n_times)
        
        if plot_save_path is not None:
        
            # Plot
            fig, ax = plt.subplots(2,2,figsize=(13,12),dpi=300, layout='constrained') #if inline layout='constrained',dpi=1000
            
            pixel_counts, _ = np.histogram(avg_fluence_model, bins=fluence_bins) # count number of pixels in each fluence bin
            
            # --- Plot 1: Persistence vs Fluence (scatter) ---
            norm_dt = colors.Normalize(vmin=dt_arr.min(), vmax=dt_arr.max())
            cmap_dt = plt.colormaps['viridis_r']
            
            for i, dt in enumerate(dt_arr):
                color = cmap_dt(norm_dt(dt))
                ax[0, 0].scatter(bin_centers, persistence_matrix[:, i], color=color, marker='.') 
            
            
            ### need to finish this option
            if self.persistence_model is not None:
            
                cubic = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d
                line = lambda x, a, b: a*x + b
                xarr = np.linspace(np.min(bin_centers),np.max(bin_centers),10000)
                
                
                self.avg_fluence = np.mean(hdul_light[1].data[ymin:ymax, xmin:xmax])
                self.med_fluence = np.median(hdul_light[1].data[ymin:ymax, xmin:xmax])
                self.fluence_std =np.std(hdul_light[1].data[ymin:ymax, xmin:xmax])
                
                self.avg_fps = self.avg_fluence/self.light_exptime
                
                '''
                ax[0, 0].plot(xarr,cubic(xarr,*popt),c='k')
                ax[0, 0].plot(xarr,cubic(xarr,*self.avg_model),c='r')
                ax[0, 0].plot(xarr,cubic(xarr,self.avg_model[0],
                                            self.avg_model[1], ###################################################### COME BACK TO FIX
                                            self.avg_model[2],
                                            line(self.med_fluence,*self.d_equ)),c='blue')
                '''
                
            sm = cm.ScalarMappable(cmap=cmap_dt, norm=norm_dt)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax[0, 0])
            cbar.set_label("Time After Exposure [s]")
            
            ax[0, 0].set_xscale('log')
            ax[0, 0].set_ylim(2.25,5.25)
            ax[0, 0].set_xlabel("Fluence [e⁻]")
            ax[0, 0].set_ylabel("Persistence [e⁻ / s]")
            ax[0, 0].set_title(f"Persistence {self.Obj}")
            ax[0, 0].grid(True, which='both', ls='--', alpha=0.4)
            ax[0, 0].set_box_aspect(1)
            
            # --- Plot 2: Persistence decay over time per fluence bin (lines) ---
            norm_fluence = LogNorm(vmin=np.min(bin_centers), vmax=np.max(bin_centers))
            cmap_fluence = plt.colormaps['viridis']
            
            for i in range(self.nbins - 1):
                color = cmap_fluence(norm_fluence(bin_centers[i]))
                #ax[1, 0].plot(dt_array, persistence_matrix[i, :]-persistence_matrix[i,0], color=color)
                ax[1, 0].plot(dt_arr, persistence_matrix[i, :], color=color)
            
            #ax[1, 0].set_yscale('log')
            ax[1, 0].set_xlabel('Time After Exposure [s]')
            ax[1, 0].set_ylabel('Persistence [e⁻ / s]')
            ax[1, 0].set_title('Persistence Decay per Fluence Bin')
            ax[1, 0].set_box_aspect(1)
            ax[1, 0].axvline(x=60, linestyle='--', c='gray', alpha=0.4)
            
            sm2 = cm.ScalarMappable(cmap=cmap_fluence, norm=norm_fluence)
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax[1, 0])
            cbar2.set_label('Fluence [e⁻]')
            cbar2.locator = LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10)
            cbar2.update_ticks()
            
            # --- Plot 3: Histogram of Fluence ---
            ax[0, 1].scatter(bin_centers, pixel_counts, marker='.')
            ax[0, 1].set_yscale('log')
            ax[0, 1].set_xscale('log')
            ax[0, 1].set_xlabel("Fluence [e⁻]")
            ax[0, 1].set_ylabel("Number of Pixels")
            ax[0, 1].set_title("Histogram of Pixel Fluence")
            ax[0, 1].grid(True, which="both", ls="--", alpha=0.4)
            ax[0, 1].set_box_aspect(1)
            
            # --- Plot 4: CCD Image ---
            ax[1, 1].imshow(hdul_light[1].data[ymin:ymax, xmin:xmax], cmap='gray', norm=LogNorm(), origin='lower')
            ax[1, 1].set_title(f"CCD Image Exposure: {self.light_exptime}s")
            
            plt.show()
            
            fig.savefig(plot_save_path) # save data plots created
            
            
            
        if data_save_path is not None:

            combined = np.vstack([dt_arr, persistence_matrix])
            
            header = 'Time(s)\t' + '\t'.join([
                f"{fluence_bins[i]:.3e}-{fluence_bins[i+1]:.3e}" 
                for i in range(len(fluence_bins)-1)
            ])
            
            np.savetxt(data_save_path, combined.T, fmt="%.6e", delimiter='\t',header=header) # save time and persistence data 
        
        
        
        return dt_arr, bin_centers, persistence_matrix
    
    
    
    def model_persistence_decay(self, dt_arrs, bins, persist_matricies, visualize=False):
        '''
        Takes data from multiple persistence series with DIFFERENT INITIAL EXPOSURE TIMES of the SAME OBJECT
        '''
        
        points = [] # list to store data points
        
        for i in range(persist_matricies):  # loop over each observation set
            for j in range(bins[i]):  # fluence bins
                for k in range(dt_arrs[i]):  # time steps
                    fluence = bins[i, j]
                    time = dt_arrs[i, k]
                    persistence = persist_matricies[i, j, k]
                    if np.isfinite(persistence):
                        points.append((fluence, time, persistence)) # synthesize data points to create 3d meshgrid
        
        points = np.array(points) # convert to ndarray
        
        
        
        if visualize is not False:
            
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # plot points to create a surface
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                            cmap='viridis', linewidth=0.2)
            
            ax.set_xlabel('Fluence [e⁻]')
            ax.set_ylabel('Time [s]')
            ax.set_zlabel('Persistence [e⁻/s]')
            ax.set_title(f'{self.Obj} Persistence vs Fluence and Time')
            
            plt.tight_layout()
            plt.show()
            
            # rotation for 3d figure to save
            def rotate(angle):
                ax.view_init(elev=30, azim=angle)
                return fig,
            
            # save 3d gif
            rot = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 3), interval=100)
            
            rot.save(f'persistence_data_{self.Obj}.gif', dpi=100, writer='pillow')
        
        
        # separate points into fluence, time, and persistence
        fluence_array = points[:, 0]
        time_array = points[:, 1]
        persistence_array = points[:, 2]
        
        # defines model of persistence
        def model(X,A,B,C):
            f, t, exp = X
            return A * (np.exp(-(B*f + C)*(t-exp)) - 1)
        
        # Fit the model
        popt, pcov = curve_fit(
            model,
            (fluence_array, time_array, np.full(len(time_array),self.dark_exptime)),  # X = tuple
            persistence_array,
            p0=[2, 1e-8, 1e-4]  # Initial guesses
        )
        
        # Extract parameters
        A_fit, B_fit, C_fit = popt
        
        # determine error
        perr = np.sqrt(np.diag(pcov))
        
        A = (A_fit,perr[0])
        B = (B_fit,perr[1])
        C = (C_fit,perr[2])
        
        if visualize is not False:
        
        
            ### Modeling Persistence ###
            
            # creating linspaces for better data resolution for model
            F = np.linspace(np.min(fluence_array), np.max(fluence_array), 200)
            T = np.linspace(np.min(time_array), np.max(time_array), 60)
            points = [] # array to store synthesizxed points
            
            
            # synthesize data points using the model
            for f in F:
                for t in T:
                    X = (f,t,self.dark_exptime)
                    p = model(X,*popt)
                    
                    points.append((f,t,p))
            
            points = np.array(points) # convert to ndarray
            
            # Plot model
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Correct order for trisurf: x = time, y = fluence, z = persistence
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                                   cmap='viridis', linewidth=0.2)
            
            ax.set_xlabel('Fluence [e⁻]')
            ax.set_ylabel('Time [s]')
            ax.set_zlabel('Persistence [e⁻/s]')
            ax.set_title(f'{self.Obj} Persistence Model')
            
            plt.tight_layout()
            plt.show()
            
            # save rotating 3d gif
            rot = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 3), interval=100)
            
            rot.save(f'persistence_model_{self.Obj}.gif', dpi=100, writer='pillow')
            
            
            ### Checking model ###
            
            points = [] # list to store data points
            
            
            for i in range(persist_matricies):  # loop over each observation set
                for j in range(bins[i]):  # fluence bins
                    for k in range(dt_arrs[i]):  # time steps
                        fluence = bins[i, j]
                        time = dt_arrs[i, k]
                        persistence = persist_matricies[i, j, k]
                        X = (fluence,time,self.dark_exp)
                        if np.isfinite(persistence):
                            points.append((fluence, time, persistence)) # synthesize data points to create 3d meshgrid
            
            
            points = np.array(points) # convert to ndarray
            
            # Plot model residuals
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Correct order for trisurf: x = time, y = fluence, z = persistence
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                                   cmap='viridis', linewidth=0.2)
            
            ax.set_xlabel('Fluence [e⁻]')
            ax.set_ylabel('Time [s]')
            ax.set_zlabel('Persistence [e⁻/s]')
            ax.set_title(f'{self.Obj} Persistence Model Residuals')
            
            plt.tight_layout()
            plt.show()
            
            # define a new rotate function with lower angle to better see deviation from 0 of model residuals
            def rotate(angle):
                ax.view_init(elev=5, azim=angle)
                return fig,
            
            # save 3d gif
            rot = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 3), interval=100)
            
            rot.save(f'model_residuals_{self.Obj}.gif', dpi=100, writer='pillow')
    
        return A,B,C


    def model_initial_persistence(self,bin_centers,persistence_matrix):
        
        popt, pcov = self.cubefit(bin_centers,persistence_matrix[:, 0])
        
        self.init_persist_params = popt
        
        perr = np.sqrt(np.diag(pcov))
        
        return popt, perr
    

        
    def create_json(self,decay_params,init_params,med_fluence_arr,savepath):
        
        const_params, pcov = self.linfit(med_fluence_arr,decay_params)
        
        params = {
            'Detector': self.detector,
            'Initial Parameters': {'Coefficients': init_params,
                                 'Constant Parameters': 1},
            'Decay Parameters': decay_params
            }
  
        with open(savepath, "w") as f:
                json.dump(params, f, indent=4)



class persistence_correction:
    
    def __init__(self,json_path,image,sigma_thresh=5):
        
        with open(json_path, "r") as f:
            self.params = json.load(f)

        
        self.imgs = []
        
        self.series_ti = None
        
        self.master_mask = None
        self.master_med_fluence = None
        self.master_fluence
        self.master_time = None
        
        self.master_persist = None


    ################## PERSISTENCE CHARACTERIZATION ##################

    def init_persist(self, fluence, med_fluence):

        init_params_coef = self.params['Initial Parameters']['Coefficients']
        
        A_i, B_i, C_i = init_params_coef
        
        init_params_const = self.params['Initial Parameters']['Constant Parameters']
        
        D_m, D_b = init_params_const
        
        initial_persist_model = A_i * fluence**3 + B_i * fluence**2 + C_i * fluence + (D_m * med_fluence + D_b)
        
        return initial_persist_model
    
    
    
    def persist_decay(self, fluence, time):
    
        decay_params = self.params['Decay Parameters']
        
        a_d, b_d, c_d = decay_params
        
        decay_model = a_d * (np.exp(-(b_d*fluence + c_d)*(time-self.exptime)) - 1)
        
        return decay_model



    def persistence_model(self,fluence, med_fluence, time):
        
        p_0 = self.init_persist(fluence, med_fluence)
        
        p_t = self.persist_decay(fluence, time)
        
        return p_t + p_0
    
    
    
    def model_integral(self, fluence, med_fluence, time):
        
        decay_params = self.params['Decay Parameters']
        
        a_d, b_d, c_d = decay_params
        
        init_params_coef = self.params['Initial Parameters']['Coefficients']
        
        A_i, B_i, C_i = init_params_coef
        
        init_params_const = self.params['Initial Parameters']['Constant Parameters']
        
        D_m, D_b = init_params_const
        
        return time * (D_m * med_fluence + D_b + A_i*fluence**3 + B_i*fluence**2 + C_i*fluence - a_d) - (a_d*np.exp((-b_d*fluence - c_d)*time - (-b_d*fluence - c_d)*self.exptime)/(b_d*fluence+c_d))
        
        
        
    def analytical_integral(self, fluence, med_fluence, t_0, t_1):
        
        return self.model_integral(fluence,med_fluence,t_1) - self.model_integral(fluence,med_fluence,t_0)
        
        
    
    def visualize_model(self,time_arr=np.linspace(0,1800,31),save_path=None):
        
        fluence_bins = np.logspace(np.log10(np.min(self.imgdata)), np.log10(np.max(self.imgdata)), 200)
        
        points = [] # array to store synthesizxed points
        
        # synthesize data points using the model
        for f in fluence_bins:
            for t in time_arr:
                
                p = self.persistence_model(f,t)
                
                points.append((f,t,p))
        
        points = np.array(points) # convert to ndarray
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Correct order for trisurf: x = time, y = fluence, z = persistence
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                               cmap='viridis', linewidth=0.2)
        
        ax.set_xlabel('Fluence [e⁻]')
        ax.set_ylabel('Time [s]')
        ax.set_zlabel('Persistence [e⁻/s]')
        ax.set_title('Persistence Model')
        
        plt.tight_layout()
        plt.show()
        
        if save_path is not None:
            
            # rotation for 3d figure to save
            def rotate(angle):
                ax.view_init(elev=30, azim=angle)
                return fig,
            
            # save 3d gif
            rot = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 3), interval=100)
            
            rot.save(save_path, dpi=100, writer='pillow')
    
    
    
    ################## PERSISTENCE CORRECTION ##################     
    
    def correct_img(self, filename, mask_thresh=10000, save_path=False, save_persist=False, visualize=False):
        '''
        Function opens image and obtains basic characteristics. If the first in a series no further action is taken.
        Subsequent images are corrected for persistence. Instead of correcting for each prior image, only the values 
        which produce the greatest persistence are corrected for which should work examining the persistence characteristics 
        of light series images.
        
        Parameters:
                filename: str/path; the absolute path to the image being being corrected
                mask_thresh: int [ADU]; the count threshold for which persistence is corrected
                save_path: str/path; the absolute path leading to
        '''
        root, ext = os.path.splitext(filename)
        head, tail = os.path.split(filename)
        
        extn = 0
        if (ext == '.fz'):
            extn = 1
        
        # image characteristics
        self.hdul = fits.open(filename)
        
        self.gain = self.hdul[extn].header['EGAIN']
        self.exptime = self.hdul[extn].header['EXPTIME']
        
        if self.series_ti is None:
            self.series_ti = self.hdul[extn].header['JD']
            self.init_time = self.series_ti
        
        else:
            self.init_time = self.hdul[extn].headr['JD']
        
        self.imgdata = np.copy(self.hdul[extn].data)
        self.imgs.append(self.imgdata)
        
        self.med_fluence = np.median(self.imgdata)
        
        self.mask = (np.isfinite(self.imgdata) & (self.imgdata > mask_thresh))
        
        # define initial data arrays
        if self.master_mask is None:
            
            self.master_mask = self.mask.copy()
            
            self.master_med_fluence = np.full(self.mask.shape, -np.inf)
            self.master_med_fluence[self.mask] = self.med_fluence
            
            self.master_fluence = np.full(self.mask.shape, -np.inf)
            self.master_fluence[self.mask] = self.imgdata[self.mask]
            
            self.master_time = np.full(self.mask.shape, np.nan)
            self.master_time[self.mask] = self.init_time
            
            return self.imgdata


        ### correct persistence before updating master array
        dt0 = (self.init_time - self.master_time[self.master_mask]) * 24 * 3600
        dt1 = dt0 + self.exptime
    
        synth_values = np.abs(
            self.analytical_integral(
                self.master_fluence[self.master_mask],
                dt0,
                dt1
            )
        )
    
        self.persist_img = synth_values
        
        self.corrected_img = self.imgdata.copy()
    
        self.corrected_img[self.master_mask] -= synth_values
        
        if save_path is not False:
            
            if save_persist is True:
                hdu = fits.PrimaryHDU(data=self.persist_img)
                hdu.writeto(f'{save_path}/persist.fits', overwrite=True)
            
            hdu = fits.PrimaryHDU(data=self.imgdata)
            hdu.writeto(f'{save_path}/corrected.fits', overwrite=True)


        if visualize is True:
            
            fig, ax = plt.subplots(1,3,layout='constrained',dpi=1000)
            
            # ploting images
            ax[0].imshow(self.imgdata, cmap='gray', norm=LogNorm(), origin='lower')
            ax[0].set_title('Original Dark')
            ax[1].imshow(self.persist_img, cmap='gray', norm=LogNorm(), origin='lower')
            ax[1].set_title('Persistence Image')
            ax[2].imshow(self.corrected_img, cmap='gray', norm=LogNorm(), origin='lower')
            ax[2].set_title('Corrected Image')
    
            plt.show()
        

        ### Update master array
        # New pixels exibiting significant persistence
        new_pixels = self.mask & ~self.master_mask

        self.master_mask[new_pixels] = True
        self.master_fluence[new_pixels] = self.imgdata[new_pixels]
        self.master_med_fluence[new_pixels] = self.med_fluence
        self.master_time[new_pixels] = self.init_time

        # Repeat pixels
        repeat_pixels = self.mask & self.master_mask
        
        if np.any(repeat_pixels):

            # time since previous exposure for these pixels
            dt_old = self.init_time - self.master_time[repeat_pixels]
    
            # persistence from previous exposure evaluated now
            p_old = self.persistence_model(
                self.master_fluence[repeat_pixels],
                self.master_med_fluence[repeat_pixels],
                dt_old + self.exptime
            )
    
            # persistence from the new exposure (evaluated at t=0)
            p_new = self.persistence_model(
                self.imgdata[repeat_pixels],
                self.med_fluence,
                self.exptime
            )
    
            overwrite = p_new > p_old
    
            if np.any(overwrite):
                idx = np.where(repeat_pixels)
                sel = overwrite
    
                self.master_fluence[idx[0][sel], idx[1][sel]] = self.imgdata[idx][sel]
                self.master_med_fluence[idx[0][sel], idx[1][sel]] = self.med_fluence
                self.master_time[idx[0][sel], idx[1][sel]] = self.init_time

        
        return self.corrected_img
    