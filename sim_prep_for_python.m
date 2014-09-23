% SIMULATION framework
% This script generates pseudo-brain data, by taking the real experimental stimuli,
% calculating their spectrograms (diverse methods available), multipling
% the result by a known receptive field (RF) in freq and time, and adding a non
% linearity and gaussian random noise (known snr).
% 
% See also: /data/cmicheli/speech_encoding/simulations/generate_data.m and
% generate_data_old.m

% add path to the simulation folder:
%addpath /data/cmicheli/speech_encoding/simulations/
whitenyes = false; % whitens the Features
fakewhitenoise = false; % uses only gaussian white noise as input (disregarding hte stimuli)

% generate a 3d gabor pattern (xytsize = xy and time), two lobes only, for now
gabor = []; 
RF    = []; % receptive field used in the simulation
angle = 45; %90;  % tilt (in degrees) of the RF

% the next step requires one function of STRFlab to be in the path
%addpath /data/TOOLBOXES/strflab/preprocessing/

% meaning: 
% params: [center_x center_y tilt space_f temp_f sigmas sigmat phase_st]
% units: [ poswint poswins degrees cycles/sigmas cycles/sigmat widwint widwins degrees]
gparams = [0.5 0.5 angle 1 1 .07 .07  0]';
gabor   = make3dgabor([32 25 1], gparams);
RF      = squeeze(gabor(:,:,1));

% use real stimuli (n=210)
%load(sprintf('/data/cmicheli/speech_encoding/_results/%sDataMerged.mat','YAR'),'stimuli')
% clear speech data

% ..and calculate their spectrograms (loop over all stimuli
%addpath /data/cmicheli/speech_encoding/stimuli/MTF

%need to load stimuli
%here they are in 3 1x70 stimuli with sampling freq of 48 Khz
fsample = 48000;


stim = [];
for nstim = 1:numel(stimuli.trial)
  paras  = [10 25 0 fsample];
  frmlen = paras(2);
  method = 'STFT';
  [y,CF,Fs,tim] = mywav2aud(stimuli.trial{nstim},paras,'STFT'); %'TANDEM' 'STFT'
  if strcmp(method,'STFT')
    stim{nstim} = y(1:8:end-1,:);
  else
    stim{nstim} = y(1:4:end,:);
  end
  disp(nstim)
end

if 0
  % visualize spectrogram
  tim = [1:size(stim{1},2)]*(10e-3);
  nstim = 1;
  figure,imagesc(tim,CF,stim{nstim})
  axis xy
end

% build the time-blocks vector
assign = [];
lags = size(RF,2);
for i=1:numel(stim)
  len = size(stim{i},2)-lags+1;
  assign{i} = i*ones(len,1);
end
assign = cat(1,assign{:});

if fakewhitenoise
  % Creating a random noise version of the input matrix
  X = randn(length(assign),numel(RF));
else
  % Creating lagged version of the feature matrix
  % Here using lag(0) + 50 (@ 10msec)
  X = zeros(length(assign),numel(RF)); 
  for nstim = 1:numel(stim)
    tmp = buffering(stim{nstim},lags);
    sel = find(assign==nstim);
    X(sel,:) = tmp;
    disp(nstim)
  end  
end

% scale by the norm
% Xn = X./norm(X(:));

% scale by the max
Xn = X./max(X(:));

if whitenyes
  % whiten it (dividing for std for each freq is getting rid of 1/f shape)
  %   clear stim
  tmp = std(X);
  Xw = X./repmat(tmp,[size(X,1) 1]);
end


% create the brain response
Y = Xn*RF(:);
% add a parametric static non-linearity
% NOTE: numbers between 0 and 1 compress, a>1 expands the signal, 1 is the linear case
a = 1; 
% suggestion: choose an irrational exponent, like a = 1.357; or a = 0.777;
sig = @(x,a) (x.*(x>0)).^a - abs(x.*(x<=0)).^a; 
Y = sig(Y,a);

% add noise
snr = [];
signal = std(Y); 
perc = 0.05*std(Y); noise  = perc*randn(size(Y));
snr    = signal./std(noise);
Y = Y + noise;
clear stimuli stim
% END of simulation


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Now the focus is more on the necessity of whitening (resp or stimuli)

% From here the idea is to estimate the RF with two methods (e.g. GLM and SVR)
% one of which assumes a linear relation input-output, and the other that
% takes care of estimating the non-linearity


% Task1: find the RFs in both cases
% Task2: Estimate the non-linearity

% You can also use a heuristic approach to estimate the non-linearity based
% on the predictive power metric (e.g. R^2) by making assumptions on the inverse of the
% linearity (assumed here invertible) and estimating the exponent of that
% via cross-validation

% estimate model:
addpath(genpath('/data/TOOLBOXES/DMLT/'))

if 0
  % exclude missing values
  Ytr = Y; % managed by bias 
  Xtr = X;
  mis_data=find(isnan(sum(Y,2)));
  Ytr(mis_data,:) = [];
  Xtr(mis_data,:) = [];
end

nfolds = 5;
% lambdas = fliplr(linspace(90,105,5));
lambdas = fliplr(logspace(-1,7,10));
% m = dml.gridsearch('validator',dml.crossvalidator('mva',{dml.glm_sgd},'stat','R2', ...
%   'type','nfold','folds',nfolds), ...
%   'vars','lambda','vals',lambdas,'verbose',true); 

m = dml.gridsearch('validator',dml.crossvalidator('mva',{dml.enet('family','gaussian','alpha',0)},'stat','R2', ...
  'type','nfold','folds',nfolds), ...
  'vars','lambda','vals',lambdas,'verbose',true); 

m = m.train(Xn(10000:21000,:),Y(10000:21000));

tmp=[];
for i=1:nfolds
  tmp(i,:) = m.models{m.optimum}{i}.weights';
end
tmp2=nanmean(tmp)';
%   strf{k} = tmp;

strf = reshape(tmp2',[size(RF,1) size(RF,2)]);
figure,imagesc(strf)

% Pb1) what happens to the RF estimation if i introduce WN as stimuli?
% Pb2) what happens if i whithen?
% Pb3) what happens if i apply a non linearity (compressive/expanding)?
%   or better how do i estimate my link function (see methods)
% Pb4) which transformation do i have to apply to my inputs (X, or Y) to
%   have a better RF estimation?
