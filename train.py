import cv2
import math
import numpy as np
import os
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy.special import gamma
from scipy.ndimage import correlate
import scipy.sparse.linalg as sla
import scipy.io
from scipy.stats import exponweib
from scipy.optimize import fmin
import time

def MyPCA(sampleData, reservedRatio):
    principleVectors = []
    meanOfSampleData = np.mean(sampleData, axis=1, keepdims=False)
    meanMatrix = np.tile(meanOfSampleData,(1,sampleData.shape[1]))
    sampleData = sampleData[:,:,0]
    centerlizedData = sampleData - meanMatrix

    covarianceMatrix = np.matmul(centerlizedData.T, centerlizedData)
    subSpaceDim = min(sampleData.shape)
    reservedPCs = math.floor(subSpaceDim*reservedRatio)
    d, tmpEigVectors = sla.eigs(covarianceMatrix,subSpaceDim)
    tmpEigVectors = tmpEigVectors[:, [0,2,3,1]] # keep similar to Matlab

    eigVectors = np.matmul(centerlizedData, tmpEigVectors[:,0:reservedPCs])
    for pcIndex in range(reservedPCs):
        tmpVector = eigVectors[:, pcIndex]
        tmpVector = tmpVector / np.linalg.norm(tmpVector,2)
        principleVectors.append(tmpVector)

    principleVectors = np.array(principleVectors).T
    projectionOfTrainingData = np.matmul(principleVectors.T, centerlizedData)

    return principleVectors, meanOfSampleData, projectionOfTrainingData

def fitweibull(x):
   def optfun(theta):
      return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)

def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)

def compute_mean(feature, block_posi):
    data = feature[block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
    return np.mean(data)

def compute_feature(feature_list, block_posi):
    """Compute features.

    Args:
        feature_list(list): feature to be processed.
        block_posi (turple): the location of 2D Image block.

    Returns:
        list: Features with length of 234.
    """
    feat = []
    data = feature_list[0][block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
    alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(data)
    feat.extend([alpha_data, (beta_l_data + beta_r_data) / 2])
    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(data, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(data * shifted_block)
        # Eq. 8 in NIQE
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])

    for i in range(1,4):
        data = feature_list[i][block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
        shape, scale = fitweibull(data.flatten('F'))
        feat.extend([scale, shape])

    for i in range(4,7):
        data = feature_list[i][block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
        mu = np.mean(data)
        sigmaSquare = np.var(data.flatten('F'))
        feat.extend([mu, sigmaSquare])

    for i in range(7,85):
        data = feature_list[i][block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
        alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(data)
        feat.extend([alpha_data, (beta_l_data + beta_r_data) / 2])

    for i in range(85,109):
        data = feature_list[i][block_posi[0]:block_posi[1], block_posi[2]:block_posi[3]]
        shape, scale = fitweibull(data.flatten('F'))
        feat.extend([scale, shape])

    return feat

def matlab_fspecial(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gauDerivative(sigma):
    halfLength = math.ceil(3*sigma)

    x, y = np.meshgrid(np.linspace(-halfLength, halfLength, 2*halfLength+1), np.linspace(-halfLength, halfLength, 2*halfLength+1))

    gauDerX = x*np.exp(-(x**2 + y**2)/2/sigma/sigma)
    gauDerY = y*np.exp(-(x**2 + y**2)/2/sigma/sigma)

    return gauDerX, gauDerY

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def logGabors(rows, cols, minWaveLength, sigmaOnf, mult, dThetaOnSigma):
    nscale          = 3    # Number of wavelet scales.
    norient         = 4    # Number of filter orientations.
    thetaSigma = math.pi/norient/dThetaOnSigma  # Calculate the standard deviation of the angular Gaussian function used to construct filters in the freq. plane.
    if cols % 2 > 0:
        xrange = np.linspace(-(cols-1)/2, (cols-1)/2, cols)/(cols-1)
    else:
        xrange = np.linspace(-cols/2, cols/2-1, cols)/cols

    if rows % 2 > 0:
        yrange = np.linspace(-(rows-1)/2, (rows-1)/2, rows)/(rows-1)
    else:
        yrange = np.linspace(-rows/2, rows/2-1, rows)/rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-y,x)
    radius = np.fft.ifftshift(radius)
    theta  = np.fft.ifftshift(theta)
    radius[0,0] = 1
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    logGabor = []
    for s in range(nscale):
        wavelength = minWaveLength*mult**(s)
        fo = 1.0/wavelength
        logGabor_s = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
        logGabor_s[0,0] = 0
        logGabor.append(logGabor_s)

    spread = []
    for o in range(norient):
        angl = o*math.pi/norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = abs(np.arctan2(ds,dc))
        spread.append(np.exp((-dtheta**2) / (2 * thetaSigma**2)))

    filter = []
    for s in range(nscale):
        o_list=[]
        for o in range(norient):
            o_list.append(logGabor[s] * spread[o])
        filter.append(o_list)
    return filter

def train(data_path):
    # This function trains the pristine model
    # Parameters
    block_size_h    = 84
    block_size_w    = 84
    blockrowoverlap = 0
    blockcoloverlap = 0
    sh_th = 0.78
    sigmaForGauDerivative = 1.66
    KforLog = 0.00001
    normalizedWidth = 524
    minWaveLength = 2.4
    sigmaOnf = 0.55
    mult = 1.31
    dThetaOnSigma = 1.10
    scaleFactorForLoG = 0.87
    scaleFactorForGaussianDer = 0.28
    reservedRatio = 0.92
    sigmaForDownsample = 0.9
    gaussian_window = matlab_fspecial((5,5),5/6)
    gaussian_window = gaussian_window/np.sum(gaussian_window)

    trainingFiles = os.listdir(data_path)

    pic_features = []
    pic_sharpness = []
    for img_file in trainingFiles:
        img = cv2.imread(os.path.join(data_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64)
        img = img.round()
        img = cv2.resize(img, (normalizedWidth, normalizedWidth),interpolation=cv2.INTER_AREA)

        h, w, _ = img.shape

        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)
        img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

        O1 = 0.3*img[:,:,0] + 0.04*img[:,:,1] - 0.35*img[:,:,2]
        O2 = 0.34*img[:,:,0] - 0.6*img[:,:,1] + 0.17*img[:,:,2]
        O3 = 0.06*img[:,:,0] + 0.63*img[:,:,1] + 0.27*img[:,:,2]

        RChannel = img[:,:,0]
        GChannel = img[:,:,1]
        BChannel = img[:,:,2]

        sharpness = []
        distparam = []  # dist param is actually the multiscale features
        for scale in (1, 2):  # perform on two scales (1, 2)
            mu = convolve(O3, gaussian_window, mode='nearest')
            sigma = np.sqrt(np.abs(convolve(np.square(O3), gaussian_window, mode='nearest') - np.square(mu)))
            # normalize, as in Eq. 1 in the paper
            structdis = (O3 - mu) / (sigma + 1)

            dx, dy = gauDerivative(sigmaForGauDerivative/(scale**scaleFactorForGaussianDer));
            compRes = conv2(O1, dx + 1j*dy, 'same')
            IxO1 = np.real(compRes)
            IyO1 = np.imag(compRes)
            GMO1 = np.sqrt(IxO1**2 + IyO1**2) + np.finfo(O1.dtype).eps

            compRes = conv2(O2, dx + 1j*dy, 'same')
            IxO2 = np.real(compRes)
            IyO2 = np.imag(compRes)
            GMO2 = np.sqrt(IxO2**2 + IyO2**2) + np.finfo(O2.dtype).eps

            compRes = conv2(O3, dx + 1j*dy, 'same')
            IxO3 = np.real(compRes)
            IyO3 = np.imag(compRes)
            GMO3 = np.sqrt(IxO3**2 + IyO3**2) + np.finfo(O3.dtype).eps

            logR = np.log(RChannel + KforLog)
            logG = np.log(GChannel + KforLog)
            logB = np.log(BChannel + KforLog)
            logRMS = logR - np.mean(logR)
            logGMS = logG - np.mean(logG)
            logBMS = logB - np.mean(logB)

            Intensity = (logRMS + logGMS + logBMS) / np.sqrt(3)
            BY = (logRMS + logGMS - 2 * logBMS) / np.sqrt(6)
            RG = (logRMS - logGMS) / np.sqrt(2)

            compositeMat = [structdis, GMO1, GMO2, GMO3, Intensity, BY, RG, IxO1, IyO1, IxO2, IyO2, IxO3, IyO3]

            h, w = O3.shape

            LGFilters = logGabors(h,w,minWaveLength/(scale**scaleFactorForLoG),sigmaOnf,mult,dThetaOnSigma)
            fftIm = np.fft.fft2(O3)

            logResponse = []
            partialDer = []
            GM = []
            for scaleIndex in range(3):
                for oriIndex in range(4):
                    response = np.fft.ifft2(LGFilters[scaleIndex][oriIndex]*fftIm)
                    realRes = np.real(response)
                    imagRes = np.imag(response)

                    compRes = conv2(realRes, dx + 1j*dy, 'same')
                    partialXReal = np.real(compRes)
                    partialYReal = np.imag(compRes)
                    realGM = np.sqrt(partialXReal**2 + partialYReal**2) + np.finfo(partialXReal.dtype).eps
                    compRes = conv2(imagRes, dx + 1j*dy, 'same')
                    partialXImag = np.real(compRes)
                    partialYImag = np.imag(compRes)
                    imagGM = np.sqrt(partialXImag**2 + partialYImag**2) + np.finfo(partialXImag.dtype).eps

                    logResponse.append(realRes)
                    logResponse.append(imagRes)
                    partialDer.append(partialXReal)
                    partialDer.append(partialYReal)
                    partialDer.append(partialXImag)
                    partialDer.append(partialYImag)
                    GM.append(realGM)
                    GM.append(imagGM)

            compositeMat.extend(logResponse)
            compositeMat.extend(partialDer)
            compositeMat.extend(GM)

            feat = []
            for idx_w in range(num_block_w):
                for idx_h in range(num_block_h):
                    # process each block
                    block_posi = [idx_h * block_size_h // scale, (idx_h + 1) * block_size_h // scale,
                                          idx_w * block_size_w // scale, (idx_w + 1) * block_size_w // scale]
                    feat.append(compute_feature(compositeMat, block_posi))

            if scale == 1:
                for idx_w in range(num_block_w):
                    for idx_h in range(num_block_h):
                        # process each block
                        block_posi = [idx_h * block_size_h // scale, (idx_h + 1) * block_size_h // scale,
                                              idx_w * block_size_w // scale, (idx_w + 1) * block_size_w // scale]
                        sharpness.append(compute_mean(sigma, block_posi))

            distparam.append(np.array(feat))
            gauForDS = matlab_fspecial([math.ceil(6*sigmaForDownsample), math.ceil(6*sigmaForDownsample)], sigmaForDownsample)
            filterResult = convolve(O1, gauForDS, mode='nearest')
            O1 = filterResult[0::2,0::2]
            filterResult = convolve(O2, gauForDS, mode='nearest')
            O2 = filterResult[0::2,0::2]
            filterResult = convolve(O3, gauForDS, mode='nearest')
            O3 = filterResult[0::2,0::2]

            filterResult = convolve(RChannel, gauForDS, mode='nearest')
            RChannel = filterResult[0::2,0::2]
            filterResult = convolve(GChannel, gauForDS, mode='nearest')
            GChannel = filterResult[0::2,0::2]
            filterResult = convolve(BChannel, gauForDS, mode='nearest')
            BChannel = filterResult[0::2,0::2]

        distparam = np.concatenate(distparam, axis=1)
        pic_features.append(np.array(distparam))
        pic_sharpness.append(sharpness)

    prisparam = []
    for i in range(len(pic_features)):
        cur_distparam = pic_features[i]
        cur_sharpness = pic_sharpness[i]
        InfIndicator = np.sum(np.isinf(feat),axis=1)
        InfIndicator = np.where(InfIndicator>0, 1, 0)
        cur_sharpness = np.array(cur_sharpness)*(1-InfIndicator)
        feat = cur_distparam[np.where(cur_sharpness>sh_th*np.max(cur_sharpness))]
        prisparam.append(feat)

    dataInHighDim = np.array(prisparam).T
    principleVectors, meanOfSampleData, projectionOfTrainingData = MyPCA(dataInHighDim,reservedRatio)

    prisparam = projectionOfTrainingData.T
    mu_prisparam = np.nanmean(prisparam, axis=0)
    prisparam_no_nan = prisparam[~np.isnan(prisparam).any(axis=1)]
    cov_prisparam = np.cov(prisparam_no_nan, rowvar=False)

    templateModel = []
    templateModel.append(mu_prisparam)
    templateModel.append(cov_prisparam)
    templateModel.append(meanOfSampleData)
    templateModel.append(principleVectors)

    scipy.io.savemat('./python_templateModel.mat', {'templateModel':[templateModel]})

if __name__ == '__main__':
    import warnings
    img_path = '../pristine/'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        time_start = time.time()

        train(img_path)

        time_used = time.time() - time_start
    print(f'\t time used in sec: {time_used:.4f}')
