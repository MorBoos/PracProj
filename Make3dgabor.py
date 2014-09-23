# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 15:06:19 2014

@author: moritz
"""

def make3dgabor(xytsize,params):
    """ported from STRFLAB
    INPUT:
    xytsize     -   3 element array
    params      -   dict w/ potential elements
                center_x
                center_y
                direction
                spat_freq
                tmp_freq
                spat_env_size
                tmp_env_size
                phase
                
    OUTPUT
    gabor function of size X x Y x T
    quadrature pair of gabor function"""
    from math import pi
    import numpy as np
    
    if "phase" in params.keys():
        params["phase"] = params["phase"] * pi/180
    else:
        params["phase"] = 0
    
    for key in params.keys():
        if not isinstance(params[key],float):
            params[key] = float(params[key])
    
    dx = np.linspace(0,1,xytsize[0])
    dy = np.linspace(0,1,xytsize[1])
    dt = np.linspace(0,1,xytsize[2]) if xytsize[2] > 1 else 0.5
    
    iy,ix,it = np.meshgrid(dx,dy,dt,indexing="ij")
    
    gauss = np.exp(- ((ix-params["center_x"])**2+(iy-params["center_y"])**2)/(2*params["spat_env_size"]**2) - (it-0.5)**2/(2*params["tmp_env_size"]**2) ) 
    
    fx = -params["spat_freq"]*np.cos(params["direction"]/180*pi)*2*pi
    fy = params["spat_freq"]*np.sin(params["direction"]/180*pi)*2*pi
    ft = params["tmp_freq"]*2*pi

    grat = np.sin( (ix-params["center_x"])*fx + (iy-params["center_y"])*fy + (it-0.5)*ft + params["phase"])
    gabor = gauss*grat

    grat = np.cos( (ix-params["center_x"])*fx + (iy-params["center_y"])*fy + (it-0.5)*ft + params["phase"])
    gabor90 = gauss*grat
    
    if np.max(np.abs(gabor))==0:
        gabor = -gabor90
        
    
    
    return (gabor,gabor90)
    
    