import numpy as np
from numpy import sqrt, pi
import scipy as sp
import tensorflow as tf
from keras.layers import BatchNormalization
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import unitary_group
from scipy.stats import ortho_group


# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# (n,k) = 2^k messages over n complex-valued channel uses
k = 4 # either 4 or 8
n = 1 # complex-valued dimensions
R = k/n # bits / 2 dims

M = 2**k # constellation size
N = 2*n # real-valued dimensions

Tx = 2 # number of transmit antennas
Rx = 2 # number of receive antennas

Tx_NN = 64 # neurons in hidden layer at transmitter
Rx_NN = 512 # neurons in hidden layer at receiver
batch_norm = False

enc_dims = [k, Tx_NN, Tx_NN, Tx_NN, N*Tx]
dec_dims = [N*Rx+2*Tx*Rx, Rx_NN, Rx_NN, Rx_NN, k]

B = 65536 # minibatch size
lr = 0.001 # learning rate

EsNo_dB_r = [5,10,15,27]


def gen_msgs(a):
  block_length = k
  numbers = tf.range(2**block_length)
  binary_tensor = tf.bitwise.right_shift(
      tf.expand_dims(numbers, axis=-1), tf.range(block_length)
  ) & 1
  binary_tensor = tf.reverse(binary_tensor, axis=[-1])
  messages = tf.tile(binary_tensor,[a//M,1])
  messages = tf.cast(messages, dtype=tf.int64)
  return messages

def normalization(x): # power per tx antenna is 1
    """ x has shape [B, Tx*N]
    """
    Bt = x.shape[0] # assumed to be a multiple of M
    x = tf.reshape(x, [Bt//M, M, Tx*N])
    x = x / tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=2, keepdims=True), axis=1, keepdims=True)/(Tx*n))
    return tf.reshape(x, [Bt, Tx*N])

def normalization_old(x): # power per tx antenna is 1
    """ x has shape [B, Tx*N]
    """
    return x / tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=1, keepdims=True), axis=0, keepdims=True)/(Tx*n))
    
def mimo_channel(xr, xi, Hr, Hi, sigma2):
    """ xr,xi has shape [B, Tx*N]
        Hr,Hi have shape [B, Rx, Tx]
        y = H*x + n
    """
    yr = tf.matmul(Hr,xr) - tf.matmul(Hi,xi)
    yi = tf.matmul(Hr,xi) + tf.matmul(Hi,xr)
    yr = yr + tf.random.normal(tf.shape(yr), mean=0.0, stddev=tf.sqrt(tf.cast(sigma2, tf.float32)))
    yi = yi + tf.random.normal(tf.shape(yi), mean=0.0, stddev=tf.sqrt(tf.cast(sigma2, tf.float32)))
    return yr, yi

class Encoder(tf.Module):
    def __init__(self, dims, batch_norm=False):
        super().__init__()
        if len(dims) < 2: raise ValueError("Input list has to be at least length 2")
        
        self.layers = []
        for i in range(len(dims)-2):
            #if batch_norm == True:
            #    self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.Dense(dims[i+1], activation="relu"))
        #if batch_norm == True:
        #    self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Dense(dims[-1], activation=None))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return normalization(x)
        #return normalization_old(x)

class Decoder(tf.Module):
    def __init__(self, dims, batch_norm=False):
        super().__init__()
        if len(dims) < 2: raise ValueError("Input list has to be at least length 2")
        
        self.layers = []
        for i in range(len(dims)-2):
            if batch_norm == True:
                self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.Dense(dims[i+1], activation="relu"))
        if batch_norm == True:
            self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Dense(dims[-1], activation="sigmoid"))
        
    def __call__(self, x, training):
        for layer in self.layers:
            if "batch_normalization" in layer.name:
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

class AE(tf.Module):
    """ AE: communications autoencoder
    
    Args:
        enc_dims: dimensions (neurons per layer) for encoder
        dec_dims: dimensions (neurons per layer) for decoder
    
    """
    def __init__(self, enc_dims, dec_dims, batch_norm=False):
        super().__init__()
        self.Encoder = Encoder(enc_dims, batch_norm)
        self.Decoder = Decoder(dec_dims, batch_norm)
    
    def __call__(self, ohv, Hr, Hi, sigma2, training=False):
        Hr = tf.repeat(Hr, M, axis=0)
        Hi = tf.repeat(Hi, M, axis=0)
        HrR = tf.reshape(Hr, [-1, Tx*Rx])
        HiR = tf.reshape(Hi, [-1 ,Tx*Rx])
        
        x = self.Encoder(ohv)
        
        xr = tf.reshape(x[:,0:N*Tx//2], [-1, Tx, N//2])
        xi = tf.reshape(x[:,N*Tx//2:N*Tx], [-1, Tx, N//2])
    
        yr,yi = mimo_channel(xr, xi, Hr, Hi, sigma2)
        
        yr = tf.reshape(yr, [-1, Rx*N//2])
        yi = tf.reshape(yi, [-1, Rx*N//2])
        y = tf.concat([yr, yi], axis=1)
        
        return self.Decoder(tf.concat([y, HrR, HiR], axis=1), training)
    
bce_loss = tf.keras.losses.BinaryCrossentropy()

@tf.function # debug in eager mode, then compile into a static graph (faster)
def train_step_eager(sigma2):
    #indices = tf.random.uniform(shape=(B,), minval=0, maxval=M, dtype=tf.dtypes.int32)
    messages = gen_msgs(B)
    messages = tf.cast(messages, tf.float32)
    Hr = tf.random.normal([B//M, Rx, Tx])/np.sqrt(2)
    Hi = tf.random.normal([B//M, Rx, Tx])/np.sqrt(2)
    
    with tf.GradientTape() as tape:
        qxy = model(messages, Hr, Hi, sigma2, training=True)
        epsilon = 1e-10  # Small value to avoid log(0)
        L = bce_loss(messages, qxy)
        
    grads = tape.gradient(L, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))    
    return L

model = AE(enc_dims, dec_dims, batch_norm)

for _ in range(1):
	iterations = 500
	print_interval = 100

	log_dir = "logs/open_loop/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	train_summary_writer = tf.summary.create_file_writer(log_dir)

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
	EsNo_dB_r = [10]
	loss_sv = []
	total_iter = 0 
	for EsNo_dB in EsNo_dB_r:
		print("EsNo = {:.1f} dB".format(EsNo_dB), flush=True)
		EsNo_r = 10**(EsNo_dB/10)
		sigma2 = 1/(2*EsNo_r) # noise power per real dimension
		t = tqdm(range(1, iterations+1), desc="loss")
		for i in t:
			L = train_step_eager(sigma2)
			total_iter = total_iter + 1

			if i%print_interval==0 or i==1:
				with train_summary_writer.as_default():
					tf.summary.scalar('loss', L, step=total_iter)

				t.set_description("loss={:.5f}".format(L))
				t.refresh() # to show immediately the update
				loss_sv.append(L)
		break