# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:36:55 2014

@author: moritz
"""

#port cris' simulations file

#new strategy: use mywav2aud on the stimuli, save these stimuli, then use the end product
#problem: there exists no direct corresponding file to the one used in cris' file
#workaround that could work for now

#wave files were filtered by mywav2aud

from scipy.io import loadmat
from Make3dgabor import make3dgabor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import ElasticNetCV,ElasticNet
from sklearn import preprocessing
from sklearn import grid_search
from sklearn.neighbors import KNeighborsRegressor,KernelDensity
from scipy.interpolate import UnivariateSpline
from sklearn.cross_validation import KFold
#%%
#create gabor RF

angle = 45
gparams = dict()

gparams["center_x"] = 0.5
gparams["center_y"] = 0.5
gparams["direction"] = angle
gparams["spat_freq"] = 1
gparams["tmp_freq"] = 1
gparams["spat_env_size"] = 0.07
gparams["tmp_env_size"] = 0.07
gparams["phase"] = 0

gabortup = make3dgabor([32,25,1],gparams)
RF = np.squeeze(gabortup[0])

#%%

srv_path = "/home/mboos/Simulations/stim_matrix.mat"

stim = loadmat(srv_path)
fname_list = stim["fnamelist"]
stim = np.squeeze(stim["stim"])

#%%

#building the time blocks vector
#what are we trying to do:
#create a list of arrays frames X lags*Features
#for each stimulus

nlags = RF.shape[1]
nfeat = RF.shape[0]

#create lagged stimuli arrays
#WARNING: this creates dependent observations
#better model these hierarchically
lagged_stimuli = [  np.vstack( np.concatenate((np.zeros((nlags-i)*nfeat),stimulus[:,max(0,i-nlags):i].flatten(order='F'))) if i < nlags else stimulus[:,max(0,i-nlags):i].flatten(order='F') for i in xrange(1,stimulus.shape[1]+1) ) for stimulus in stim ]

#%%
#now stack these vertically
lagged_stimuli = np.vstack(lagged_stimuli)


        
#%%
#now create the data
        
#scale by the max of the features
#might this be a problem since the data are dependent?
#maybe change this
#lagged_stimuli = lagged_stimuli / np.max(lagged_stimuli,axis=0)

#normalize with standardscaler
scaler = preprocessing.StandardScaler()
scaler.fit(lagged_stimuli)

lagged_stimuli = scaler.transform(lagged_stimuli)

Y = lagged_stimuli.dot(RF.flatten(order="F"))

#%%
#now add a non-linearity
#--------------------------------------
##this is the one cris uses:
#a = 1; 
#% suggestion: choose an irrational exponent, like a = 1.357; or a = 0.777;
#sig = @(x,a) (x.*(x>0)).^a - abs(x.*(x<=0)).^a; 
#Y = sig(Y,a);
#
#% add noise
#snr = [];
#signal = std(Y); 
#perc = 0.05*std(Y); noise  = perc*randn(size(Y));
#snr    = signal./std(noise);
#Y = Y + noise;
#-------------------------------------

#a = 1.357

#Y = (Y*(Y>0))**a - np.abs(Y*(Y<=0))**a

a=3

Y = Y**a

#%%
#add noise

perc = 0.05*np.std(Y)
Y = Y+perc*np.random.rand(Y.size)

#%%
#now for the estimation
#split into test & train set
train_X,test_X,train_Y,test_Y = train_test_split(lagged_stimuli,Y,test_size=0.3)

#%%
#try elastic net CV
#enet_model = ElasticNetCV([.1,.3,.7,.9,.99],cv=3,n_jobs=-1)
#enet_model.train(train_X,train_Y)
#pred_Y = enet_model.predict(train_X)

#for now
enet = ElasticNet(l1_ratio=0.7)
enet.fit(train_X,train_Y)
pred_Y = enet.predict(train_X)

#%%
#non-linearity first by CV NN

#much too slow
parameters_NN = { 'weights' : ('uniform','distance') , 'n_neighbors' : [5,10,20,40]}
NN_nonl = KNeighborsRegressor(n_neighbors=10)
gs_NN = grid_search.RandomizedSearchCV(NN_nonl,parameters_NN,verbose=1)

#try kernel estimation
#parameters_kernel = { 'bandwidth' : [1.0,3.0,5.0,10.0]}
#kde = KernelDensity()
#kde_cv = grid_search.RandomizedSearchCV(kde,parameters_kernel,n_jobs=-1)

#%%

#does not work well - need CV probably
#order = np.argsort(pred_Y)
#spl = UnivariateSpline(pred_Y[order],train_Y[order],s=pred_Y.size)
#KFold(5)

#%%

