import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import rescale, pyramid_reduce

def sreCal(Xref,X):
    '''Calculate signal to reconstructed error (SRE) between reference image and reconstructed image
    Input: Xref, X: reference and reconstructed images in shape [h,w,d]
    Output: aSRE average SRE in dB
            SRE_vec: SRE of each band'''
    mSRE=0
    if len(Xref.shape)==3:
        Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1],Xref.shape[2])
        X=X.reshape(X.shape[0]*X.shape[1],X.shape[2])
        SRE_vec=np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            SRE_vec[i]=10*np.log(np.sum(Xref[:,i]**2)/np.sum((Xref[:,i]-X[:,i])**2))/np.log(10)
        mSRE=np.mean(SRE_vec)
    else:
        pass
    return mSRE,SRE_vec

def get_s2_fkernel(shape=(7,7),band=20,r=2):
    def get_gauss2D(shape=(7,7),sigma=0.5):
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

    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
    # mtf = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    sdf = r*np.sqrt(-2*np.log(mtf)/np.square(np.pi))
    if band==20:
        sigma = sdf[d==2]
        L= len(sigma)
        ker = np.ones((shape[0],shape[1],L))
        for i in range(L):
            ker[:,:,i] = get_gauss2D(shape=shape,sigma=sigma[i])
    elif band==10:
        sigma = sdf[d==1]
        L= len(sigma)
        ker = np.ones((shape[0],shape[1],L))
        for i in range(L):
            ker[:,:,i] = get_gauss2D(shape=shape,sigma=sigma[i])
    else:
        sigma = sdf[d == 6]
        L = len(sigma)
        ker = np.ones((shape[0], shape[1], L))
        for i in range(L):
            ker[:, :, i] = get_gauss2D(shape=shape, sigma=sigma[i])

    return ker.astype(np.float32)


class downsampler(tf.keras.layers.Layer):
    def __init__(self, factor=2,kernel='gaussian',band=20):
        super(downsampler, self).__init__()
        self.factor =  factor
        self.kernel = kernel
        self.band = band

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'factor': self.factor,
            'kernel': self.kernel,
            'band': self.band,
        })
        return config

    def call(self, inputs):
        if self.kernel == 'gaussian':
            if self.band==20:
                x=get_s2_fkernel(shape=(7,7),band=self.band,r=self.factor)
                gauss_kernel=x[:,:,:,tf.newaxis]
                gauss_kernel=tf.cast(gauss_kernel,dtype=tf.float32)
                out=tf.nn.depthwise_conv2d(inputs, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            elif self.band==10:
                x=get_s2_fkernel(shape=(7,7),band=self.band,r=self.factor)
                gauss_kernel=x[:,:,:,tf.newaxis]
                gauss_kernel=tf.cast(gauss_kernel,dtype=tf.float32)
                out=tf.nn.depthwise_conv2d(inputs, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            else:
                x = get_s2_fkernel(shape=(7, 7), band=self.band,r=self.factor)
                gauss_kernel = x[:, :, :, tf.newaxis]
                gauss_kernel = tf.cast(gauss_kernel, dtype=tf.float32)
                out = tf.nn.depthwise_conv2d(inputs, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        elif self.kernel == None:
            out = inputs
        y = out[:,::self.factor,::self.factor,:]
        return y

class bicubicdown(tf.keras.layers.Layer):
    def __init__(self, factor=2):
        super(bicubicdown, self).__init__()
        self.factor =  factor

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'factor': self.factor,
        })
        return config

    def call(self, inputs):
        size = [inputs.shape[1]//self.factor,inputs.shape[2]//self.factor]
        return tf.image.resize(images=inputs,size=size,method='bicubic')

def load_S2_data(filepath):
    '''Load S2 data matlab file contains:
    Xm_im: 12 bands of reference images
    Yim: 1x12 cell observed images
    outputs: Xm_im: reference
            X20: 20 m bands (target)
            X20X10: 10m + interpolated 20m: input)'''
    import h5py
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    data=h5py.File(filepath)
    Xm_im = data['Xm_im']
    Y = data['Yim']
    Xm_im = np.transpose(data['Xm_im'])
    # X20ref = Xm_im[:,:,d==2]
    X20ref = Xm_im

    Ym =[]
    for i in range(4,7):
        Ym.append(data[Y[0,i]].value)
    Ym.append(data[Y[0,8]].value)
    Ym.append(data[Y[0,10]].value)
    Ym.append(data[Y[0,11]].value)
    Y20=np.transpose(np.asarray(Ym)) #--> 20 m bands
    Y20in = rescale(Y20,scale = 2, multichannel = True) # --> interpolation 20 m bands
    Y10 = []
    for i in range(1,4):
        Y10.append(data[Y[0,i]].value)
    Y10.append(data[Y[0,7]].value)
    Y10np = np.transpose(np.asarray(Y10)) #--> 10 m bands
    Yin = np.concatenate((Y10np,Y20in),axis=-1) # concatenate with 10 m bands as input
    return X20ref.astype(np.float32), Yin.astype(np.float32),Y20.astype(np.float32), Y10np.astype(np.float32)


def degradation(filepath):
    '''Create degratded images
    input: Yim: 12x1 matlab cell arrays observed images
    output: Yd: 12x1 downsample of Yim, Xim: observed img that can be used as reference'''
    import h5py
    f = h5py.File(filepath)
    Y10 = []
    for i in range(1, 4):
        Y10.append(f[f['Yim'][0, i]].value)
    Y10.append(f[f['Yim'][0, 7]].value)
    Y10np = np.transpose(np.asarray(Y10)).astype(np.float32)  # --> 10 m bands

    Y20 = []
    for i in range(4, 7):
        Y20.append(f[f['Yim'][0, i]].value)
    Y20.append(f[f['Yim'][0, 8]].value)
    Y20.append(f[f['Yim'][0, 10]].value)
    Y20.append(f[f['Yim'][0, 11]].value)
    Y20np = np.transpose(np.asarray(Y20)).astype(np.float32)  # --> 20 m bands

    Y10d = np.squeeze(downsampler(factor=2, kernel='gaussian', band=10)(Y10np[tf.newaxis, :, :, :]))
    Y20d = np.squeeze(downsampler(factor=2, kernel='gaussian', band=20)(Y20np[tf.newaxis, :, :, :]))
    return Y10d, Y20d, Y20np, Y10np


def loadmatlabcell(filepath):
    '''Create degratded images
    input: Yim: 12x1 matlab cell arrays observed images
    output: Yd: 12x1 downsample of Yim, Xim: observed img that can be used as reference'''
    import h5py
    f = h5py.File(filepath)
    Y10 = []
    for i in range(1, 4):
        Y10.append(f[f['Yim'][0, i]].value)
    Y10.append(f[f['Yim'][0, 7]].value)
    Y10np = np.transpose(np.asarray(Y10)).astype(np.float32)  # --> 10 m bands

    Y20 = []
    for i in range(4, 7):
        Y20.append(f[f['Yim'][0, i]].value)
    Y20.append(f[f['Yim'][0, 8]].value)
    Y20.append(f[f['Yim'][0, 10]].value)
    Y20.append(f[f['Yim'][0, 11]].value)
    Y20np = np.transpose(np.asarray(Y20)).astype(np.float32)  # --> 20 m bands

    Y60 = []
    Y60.append(f[f['Yim'][0, 0]].value)
    Y60.append(f[f['Yim'][0, 9]].value)
    Y60np = np.transpose(np.asarray(Y60)).astype(np.float32)  # --> 60 m bands
    return Y10np,Y20np,Y60np