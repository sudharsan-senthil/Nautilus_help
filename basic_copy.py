from tensorflow.keras.layers import Dense, Input, GaussianNoise, add, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

eb_no = list(range(0,9,2))
start = 0
jump = -0.1
R = 5

optimizer = 'adam'
loss = 'categorical_crossentropy'

Ks = [4]
input_dims = [2**k for k in Ks]
rate = 1/2
encoder_dims = [int(k/rate) for k in Ks]
train_samples = 10000
b_size = 16
epochs = 400

def AutoEncDec(input_dim, encoder_dim):
    main_input = Input(shape=(input_dim,), name='input')
    encoder_layer = Dense(encoder_dim, activation='relu')(main_input)
    encoder_layer = Dense(encoder_dim, activation='linear')(encoder_layer)
    encoder_layer = BatchNormalization(center=False, scale=False)(encoder_layer)

    noise = Input(shape=(encoder_dim,), name='input2')
    merged = add([encoder_layer,noise])

    decoder_layer = Dense(encoder_dim, activation='relu')(merged)
    decoder_layer = Dense(encoder_dim, activation='linear')(decoder_layer)
    decoder_layer = Dense(input_dim, activation='softmax')(decoder_layer)

    encoder = Model(inputs=[main_input],outputs=[encoder_layer])
    decoder = Model(inputs=[merged],outputs=[decoder_layer])
    model = Model(inputs=[main_input,noise], outputs=[decoder_layer])
    model.compile(optimizer=optimizer, loss=loss)
    return model, encoder, decoder


def Test(enc1,dec1,b):
    BLER = [0]*R
    cnt = 1000
    test_samples = 10**7
    t_samples = int(test_samples/cnt)
    I = np.eye(2**b)
    for _ in range(cnt):
        data_test_ind = np.random.randint(2**b,size=(t_samples))
        data_test = I[data_test_ind]
        enc_sig = enc1(data_test)
        for i in range(R):
            err = 0
            sigma = 10**(start+i*jump)
            noise = np.random.normal(0, sigma, (t_samples,2*b))
            rcv_sig = enc_sig + noise
            dec_sig = dec1(rcv_sig)
            est_sig = np.array(dec_sig).argmax(axis=1)

            for x in range(t_samples):
                if est_sig[x] != data_test_ind[x]:
                    err += 1
            BLER[i] += (err/test_samples)   
    print(BLER)

Encs, Decs = [], []
BLERs = []
for m in range(10):
    for k in range(len(Ks)):
        I = np.identity(input_dims[k])
        net,enc,dec = AutoEncDec(input_dims[k],encoder_dims[k])
        data_train_ind = np.random.randint(input_dims[k],size=(train_samples))
        data_train = I[data_train_ind]

        for t in range(epochs):
            sigma = -1*(np.random.choice(list(range(0,7)))/10)
            # sigma = random.uniform(0, -0.6)
            noise = np.random.normal(0, 10**sigma, (train_samples,encoder_dims[k]))
            net.fit([data_train,noise],[data_train],validation_split=0.1,batch_size=b_size,verbose=0)
        
        Encs.append(enc)
        Decs.append(dec)
        Test(enc,dec,4)
        enc.save("data/enc"+str(m))
        dec.save("data/dec"+str(m))
            