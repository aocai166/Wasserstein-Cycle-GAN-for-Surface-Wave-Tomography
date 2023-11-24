import numpy as np
import os
import pylab as pl

##### Discretize the region ######
def get_gridpoints(grid_size=0.05):
    lon_1d = np.arange(-121,-113.5+grid_size,grid_size)
    lat_1d = np.arange(32,36.5+grid_size,grid_size)
    nx = len(lon_1d)
    ny = len(lat_1d)
    return nx,ny,lon_1d,lat_1d

##### create the grid points
nx,ny,lon_1d,lat_1d = get_gridpoints()
lat_arr,lon_arr = np.meshgrid(lat_1d,lon_1d)

def post_process_inv_results():
    # read in the inversion results
    fout_corr = 'inv_CVM-H_topo_corr_WCGANinv_full.npz'
    if os.path.isfile(fout_corr):
        out_dict = np.load(fout_corr)
        h1d = out_dict['h1d']
        #Vs_final = out_dict['Vs_final']
        Vs_final = out_dict['Vs_wcgan_pos']
        Vs_perturb = out_dict['Vs_wcgan_pos_perturb']
        #Vs_CVM = out_dict['Vs_CVM']
        return h1d,Vs_final,Vs_perturb
    else:
        return RuntimeError("file "+fout_corr+'.npz not exist')
# This is step is the key: loading the velocity models
# if you can format your model in the same structure, you don't need to change
# anything in the later part of the code
h1d,Vs_final,Vs_perturb = post_process_inv_results()

#############
def load_topo_data(lon_arr,lat_arr):
    # This text file contains interpolated topography
    ftopo = 'SC_topo_for_corr.txt'

    topo = np.loadtxt(ftopo)
    lonin = topo[:,0]; latin = topo[:,1]; elev = topo[:,2]
    lon_in = np.reshape(lonin,[131,81],order='F')
    lat_in = np.reshape(latin,[131,81],order='F')
    elev_in = np.reshape(elev,[131,81],order='F')

    nx,ny = lon_arr.shape
    elev_out = np.zeros([nx,ny])

    for i in range(nx):
        for j in range(ny):
            lon = lon_arr[i,j]; lat = lat_arr[i,j]
            I = np.argmin(abs(lon-lon_in[:,0]))
            J = np.argmin(abs(lat-lat_in[0,:]))
            elev_out[i,j] = elev_in[I,J]

    return elev_out,lon_in[:,0],lat_in[0,:],elev_in

elev_out,lon_e,lat_e,elev_in = load_topo_data(lon_arr,lat_arr)


############## Cross sections #################
# define fault locations
def load_SJF():
    lon_f = [-117.5,-117.1,-116.5,-116]
    lat_f = [34.3,33.9,33.5,33]
    f_SJF = interp1d(lon_f,lat_f,bounds_error=False)
    return f_SJF
    
def load_SAF():
    lon_f = [-120.45,-119.5,-118.9,-117.5,-117.1,-116.6,-115.8,-115.5]
    lat_f = [35.9,35,34.8,34.3,34.1,34,33.4,32.9]
    f_SAF = interp1d(lon_f,lat_f,bounds_error=False)
    return f_SAF

def load_GF():
    lon_f = [-118.9,-118,-117]
    lat_f = [34.8,35.3,35.6]
    f_GF = interp1d(lon_f,lat_f,bounds_error=False)
    return f_GF
    
def load_EF():
    lon_f = [-117.7,-117.1,-116.6]
    lat_f = [34,33.4,33.1]
    f_EF = interp1d(lon_f,lat_f,bounds_error=False)
    return f_EF
###############
# when intersect the fault, give the locations
def cross_fault_loc(f_fault,gstart,gend):
    lon_in = [gstart[0],gend[0]]
    lat_in = [gstart[1],gend[1]]
    f_line = interp1d(lon_in,lat_in,bounds_error=False)
    x_interp = np.arange(lon_in[0],lon_in[1],0.01)
    y_interp = f_line(x_interp)
    
    y_fault = f_fault(x_interp)
    ind_nan = (~np.isnan(y_fault)) & (~np.isnan(y_interp))
    y_f_nan = y_fault[ind_nan]
    x_f_nan = x_interp[ind_nan]
    y_interp_nan = y_interp[ind_nan]
    if len(y_f_nan) == 0:
        return -1,[]
    
    diff = np.abs(y_f_nan-y_interp_nan)
    
    # no intersection found  
    if min(diff) > 0.05:
        return 0,diff
    # intersection found, return the point of intersection
    else:
        ind_min = np.argmin(diff)
        lon_out,lat_out = x_f_nan[ind_min],y_f_nan[ind_min]
        
        return 1,[lon_out,lat_out]

# find out the intersect of all given labels    
def check_all_faults(gstart,gend):
    f_all = [load_SJF(),load_SAF(),load_EF(),load_GF()]
    f_names = ['SJF','SAF','EF','GF']
    
    n_f = len(f_all)
    out_name = []; fault_loc = []
    j = 0
    for i in range(n_f):
        flag,out = cross_fault_loc(f_all[i],gstart,gend)
        if flag == 1:
            out_name.append(f_names[i])
            fault_loc.append(out)
            j += 1
    if j == 0:
        return 0,[]
    else:
        return 1,[out_name,fault_loc]
############
from mpl_toolkits.axes_grid1 import make_axes_locatable
def imshow_CS(x_cs,depth,elev_arr,Vsin,vmin,vmax,ax=[],if_cbar=1,if_fault=[],column=0,i_reset=0):
    Zm = ma.masked_where(np.isnan(Vsin),Vsin); Dist,Depth = np.meshgrid(x_cs,depth)
    if if_fault != []:
        [dist_f,f_name] = if_fault; flag = 1; n_f = len(f_name)
    else:
        flag = 0
        
    if ax == []:
        cax = pl.pcolormesh(Dist,Depth,Zm,cmap='jet_r',vmin=vmin,\
                            vmax=vmax,rasterized=True)
        pl.gca().invert_yaxis()
        if if_cbar == 1:
            cbar = pl.colorbar(); cbar.set_label('Inv Vs',fontsize=14)
            cbar.ax.tick_params(labelsize=14)
        pl.plot(x_cs,elev_arr-2.5,'k-')
        pl.tick_params(labelsize=14)
        
        if flag == 1:
            for i in range(n_f):
                ind_f = np.argmin(np.abs(x_cs-dist_f[i]))
                
                pl.text(dist_f[i],elev_arr[ind_f],f_name[i],ha='center',\
                        va='bottom',fontsize=12,rotation=30)
    else:
        
        im = ax[0].pcolormesh(Dist,Depth,Zm,cmap='jet_r',vmin=vmin,\
               vmax=vmax,rasterized=True)
        if column == 0:
            ax[0].set_ylabel('Elevation (km)',fontsize=14)
        if i_reset == 2:
            ax[0].set_xlabel('Distance to SAF (km)',fontsize=14)
        if if_cbar:
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right",size="2%",pad=0.1)
            cax.tick_params(labelsize=14)
            cbar = ax[1].colorbar(im,cax=cax)
            if column == 0:
                cbar.set_label('Vs (km/s)',fontsize=14)
            else:
                cbar.set_label('Perturbation (%)',fontsize=14)
        
        ax[0].plot(x_cs,elev_arr*2+1.5,'k-')
        ax[0].set_yticks([20,15,10,3,1.5,-0.5,-2.5,-4.5])
        ax[0].set_yticklabels([20,15,10,3,'','','',''])
        ax[0].tick_params(labelsize=14)
        
                
        if flag == 1:
            for i in range(n_f):
                ind_f = np.argmin(np.abs(x_cs-dist_f[i]))
                
                ax[0].text(dist_f[i],-5.1,f_name[i],ha='center',va='bottom',\
                  fontsize=12,style='italic')
                ax[0].plot([dist_f[i],dist_f[i]],[2*elev_arr[ind_f]+1.2,-5.0],\
                  '-',color='gray')
                ax[0].plot(dist_f[i],elev_arr[ind_f]*2+1.,'v',markersize=8,\
                  color='gray')
        
###############
from scipy.interpolate import interp1d,interp2d
def fill_nan_value(h1d,Vsin):
    nx,ny,nh = Vsin.shape
    Vs_out = Vsin.copy()
    # get num of nan in each grid
    flag = np.ones_like(Vsin)
    flag[np.isnan(Vsin)] = 0
    N_nonan = np.sum(flag,axis=-1)
    # do not use grid inside the sea
    elev_tmp = elev_out.copy(); elev_tmp[90:,:] = 1
    N_nonan[elev_tmp < -0.05] = -100
       
    for i in range(nx):
        for j in range(ny):
            if N_nonan[i,j] < 50:
                Vs_out[i,j,:] = np.nan
                continue
            Vtar = Vsin[i,j,:]
            ind_tar = ~np.isnan(Vtar)
            ftar = interp1d(h1d[ind_tar],Vtar[ind_tar],kind='nearest',fill_value='extrapolate')
            Vs_out[i,j,:] = ftar(h1d)
    return Vs_out

Vs_interp = fill_nan_value(h1d,Vs_final)
Vs_perturb_interp = fill_nan_value(h1d,Vs_perturb)

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy.ma as ma

import pyproj
g = pyproj.Geod(ellps='WGS84')

def sphere_dist(elon,elat,slon,slat,elev=0):
    az1,az2,dist = g.inv(elon,elat,slon,slat)
    return dist/1000.

def get_and_interp_CS_1dep(Vsin,slon,slat,elon,elat,sdist,edist,num=1000,if_plot=0):
    x = lon_arr.T.copy(); y = lat_arr.T.copy(); tmp = Vsin.T.copy()
    z = tmp.copy(); z[np.isnan(z)] = np.nanmedian(z)
    
    # Coordinates of the line we'd like to sample along
    line = [(slon,slat), (elon,elat)]
    
    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(zip(*line))
#    print line
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y_world - y.min()) / y.ptp()
    
    # Interpolate the line at "num" points...
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]
    x_cs = np.linspace(sdist,edist,num)
    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((row, col)))
    xi = scipy.ndimage.map_coordinates(x, np.vstack((row, col)))
    
    if if_plot:
        # Plot...
        fig, axes = plt.subplots(nrows=2)
        axes[0].pcolormesh(x, y, tmp,cmap='jet_r',rasterized=True)
        axes[0].plot(x_world, y_world, 'ro-')
        axes[0].axis('image')
        
        axes[1].plot(x_cs,zi)
        pl.tight_layout()
        plt.show()
    return x_cs,zi

# performs interpolation to make the cross section look smooth
def get_CS_for_3Dmodel(Vs3d,slon,slat,elon,elat,sdist,edist,Zmax=max(h1d),num=2000,if_plot=0):
    ######
    gstart = [slon,slat]; gend = [elon,elat]
    flag,out = check_all_faults(gstart,gend)
    if flag == 1:
        f_name = out[0]; f_loc = out[1]
        n_f = len(f_name)
        dist_f = np.zeros(n_f)
        for i in range(n_f):
            dist_f[i] = sphere_dist(f_loc[i][0],f_loc[i][1],gstart[0],gstart[1],0)+sdist
    #######
    x_cs,elev = get_and_interp_CS_1dep(elev_out,slon,slat,elon,elat,sdist,\
                                       edist,num=num,if_plot=if_plot)
    elev_arr = -np.array(elev)
    
    ind_dep = h1d<=Zmax; Z = h1d[ind_dep]
    nh = len(Z); Vs_out = np.empty([nh,num]); Vs_out[:] = np.nan
    for i in range(nh):
        Vsin = Vs3d[:,:,i]
        x_cs,Vs_out[i,:] = get_and_interp_CS_1dep(Vsin,slon,slat,elon,elat,\
                   sdist,edist,num=num,if_plot=0)
        
    ## interpolate along depth
    f2d = interp2d(x_cs,Z,Vs_out)
    depth = np.linspace(min(Z),max(Z),1000)
    Vs_iout = f2d(x_cs,depth)
    #########
    Dist,Depth = np.meshgrid(x_cs,depth)
    
    Elev,trash = np.meshgrid(elev_arr,depth)
    ind_above = Elev >= Depth
    
    Vs_iout[ind_above] = np.nan
    if if_plot:
        pl.figure(1,figsize=(12,4))
        vmin,vmax = np.nanpercentile(Vs_iout,[5,95])
        if flag == 1:
            imshow_CS(x_cs,depth,elev_arr,Vs_iout,vmin,vmax,if_fault=[dist_f,f_name],column=0)
        else:
            imshow_CS(x_cs,depth,elev_arr,Vs_iout,vmin,vmax,column=0)
        
        pl.show()
    return x_cs,depth,elev_arr,Vs_iout

# subfunction that plots with a brunch of inputs
def comp_2model_CS(Vs_final,Vs_perturb,slon,slat,elon,elat,sdist,edist,fig,\
                   ax,ax1,Zmax=max(h1d),num=2000,if_plot=0,label=[],i_reset=0):
    ######
    gstart = [slon,slat]; gend = [elon,elat]
    flag,out = check_all_faults(gstart,gend)
    if flag == 1:
        f_name = out[0]; f_loc = out[1]
        n_f = len(f_name)
        dist_f = np.zeros(n_f)
        for i in range(n_f):
            dist_f[i] = sphere_dist(f_loc[i][0],f_loc[i][1],gstart[0],gstart[1],0)+sdist
    #######
    x_cs,depth,elev_arr,Vs_f = get_CS_for_3Dmodel(Vs_final,slon,slat,elon,elat,\
                                                  sdist,edist,Zmax=Zmax,num=num,if_plot=0)
    x_cs,depth,elev_arr,Vs_p = get_CS_for_3Dmodel(Vs_perturb,slon,slat,elon,elat,\
                                                  sdist,edist,Zmax=Zmax,num=num,if_plot=0)
    ind_color = depth > 3.0
    ind_colorc = depth > 3.0
    Vs_f[~ind_color,:] = np.nan; Vs_p[~ind_color,:] = np.nan
    vmin,vmax = np.nanpercentile(Vs_f[ind_colorc,:],[5,95])
    if flag == 1:
        imshow_CS(x_cs,depth,elev_arr,Vs_f,vmin,vmax,ax=[ax,fig],if_cbar=1,\
                  if_fault=[dist_f,f_name],column=0,i_reset=i_reset)
        # Draw the interpretations
        #ax.plot([-1,9],[3,20])
        #ax.plot([-6,-2],[3,20])
    else:
        imshow_CS(x_cs,depth,elev_arr,Vs_f,vmin,vmax,ax=[ax,fig],if_cbar=1,column=0,i_reset=i_reset)

    pmin,pmax = np.nanpercentile(Vs_p[ind_colorc,:],[0,100]); ptmp = max([-pmin,pmax])
    #pmin = -ptmp; pmax=ptmp
    if flag == 1:
        imshow_CS(x_cs,depth,elev_arr,Vs_p,pmin,pmax,ax=[ax1,fig],if_cbar=1,\
                  if_fault=[dist_f,f_name],column=1,i_reset=i_reset)
        # Draw the interpretations
        #ax1.plot([-1, 9], [3, 20])
        #ax1.plot([-5, -1], [3, 20])
    else:
        imshow_CS(x_cs,depth,elev_arr,Vs_p,pmin,pmax,ax=[ax1,fig],if_cbar=1,column=1,i_reset=i_reset)

    
    y_lim = ax.get_ylim(); ymin = -5
    ax.set_ylim([max(y_lim),ymin])
    ax1.set_ylim([max(y_lim),ymin])
    if label != []:
        
        ax.text(sdist,ymin,label,fontsize=14,ha='center',va='bottom')
        ax.text(edist,ymin,label+'\'',fontsize=14,ha='center',va='bottom')
        ax1.text(sdist,ymin,label,fontsize=14,ha='center',va='bottom')
        ax1.text(edist,ymin,label+'\'',fontsize=14,ha='center',va='bottom')

# This is the main function that generates the final figures
def get_CS_6endpoints():
    # get the lon lat of each cross-section profile defined profile_elev.dat
    fname = open('profile_elev.dat','r')
    tmp = fname.read(); data = tmp.split('>>')
    # choose the indexes of profiles you want to plot
    i_arr = np.array([6,8,9])
    i_fig = 1; i_reset = 0; i_col = 0
    label = ['D','E','F']
    # Figure 1 plots the final model and perturbations
    fig1,ax_all1 = pl.subplots(ncols=2,nrows=3,sharex = False, sharey = True, figsize=(12,6))
    for i in i_arr:
        if i_reset == 3:

            i_reset = 0; i_col += 1
            
        ax = ax_all1[i_reset,i_col]
        ax1 = ax_all1[i_reset,i_col+1]
        if i == 0:
            tmp2 = data[i][0:-1].split('\n')
        else:
            tmp2 = data[i][1:-1].split('\n')
        
        point_e = tmp2[-1].split(); point_s = tmp2[0].split()
        print(point_s,point_e)
        slon,slat,sdist = float(point_s[0]),float(point_s[1]),float(point_s[2])
        elon,elat,edist = float(point_e[0]),float(point_e[1]),float(point_e[2])
        
        comp_2model_CS(Vs_interp,Vs_perturb_interp,slon,slat,elon,elat,sdist,\
                       edist,fig1,ax,ax1,Zmax=20,num=2000,\
                       if_plot=1,label=label[i_fig-1],i_reset=i_reset)
        i_fig += 1; i_reset += 1
    fig1.tight_layout()

    pl.show()
    fig1.savefig('Cross_section_final.png',format='png',bbox_inches='tight',dpi=1200)
get_CS_6endpoints()
###############