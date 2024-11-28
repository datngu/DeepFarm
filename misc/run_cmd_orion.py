# singularity shell --nv /mnt/users/ngda/sofware/singularity/ndatth-deepsea-v0.0.0.img


import argparse
import os, sys
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping


import glob
num_threads = 32
train_files = glob.glob('*tfr')
val_files = glob.glob('*val')
out = "transformer_model"
#!nvidia-smi




out_model = out + '.h5'
out_hist = out + '.csv'

# Decoding function
def parse_record(record):
    name_to_features = {
        'seq': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)



def decode_record(record):
    seq = tf.io.decode_raw(
        record['seq'], out_type=tf.float16, little_endian=True, fixed_length=None, name=None
    )
    label = tf.io.decode_raw(
        record['label'], out_type=tf.int8, little_endian=True, fixed_length=None, name=None
    )
    seq = tf.reshape(seq, [-1,4])
    #label = tf.cast(label, tf.float16)
    return (seq, label)



def get_dataset(record_file, num_threads = 8, batch = 512):
    dataset = tf.data.TFRecordDataset(record_file, num_parallel_reads = num_threads, compression_type = 'GZIP')
    dataset = dataset.map(parse_record, num_parallel_calls = num_threads)
    dataset = dataset.map(decode_record, num_parallel_calls = num_threads)
    dataset = dataset.shuffle(buffer_size = batch*10).batch(batch)
    return dataset
    


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




    
    
    
def positional_encoding(max_position, d_model):
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return position * angle_rates



class ConvBlock(layers.Layer):
    def __init__(self, seq_len = 1000, filters = 512, kernel_size = 11, drop_rate = 0.2, out_embeding = 100):
        super().__init__()
        self.conv1 =  layers.Conv1D(filters = filters , kernel_size = kernel_size, strides=1, padding='same', activation='relu', name = 'conv1')
        self.conv2 =  layers.Conv1D(filters = filters*2 , kernel_size = kernel_size, strides=1, padding='same', activation='relu', name = 'conv2')
        self.conv3 =  layers.Conv1D(filters = out_embeding , kernel_size = 1, strides=1, padding='same', activation='relu', name = 'conv3')
        self.max1 = layers.MaxPool1D(pool_size=2,strides=2, padding='valid', name = 'maxpool2')
        self.max2 = layers.MaxPool1D(pool_size=4,strides=4, padding='valid', name = 'maxpool1')
        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.pos_encode = positional_encoding( int(seq_len/8), out_embeding)
    
    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        return x + self.pos_encode



data_dim = get_dataset(val_files, batch = 1)
dim_it = iter(data_dim)
dim_sample = next(dim_it)

DENSE_UNITS = 925
NUM_INPUT = dim_sample[0].shape[1]
NUM_OUTPUT = dim_sample[1].shape[1]




def build_model(seq_len, n_track, filters = 512, num_heads = 8, ff_dim = 128, embeding = 64):
    
    inp = layers.Input(shape=(seq_len, 4))
    x = ConvBlock(seq_len = seq_len, filters = filters, out_embeding = embeding)(inp)
    x = TransformerBlock(embeding, num_heads, ff_dim)(x)
    x = TransformerBlock(embeding, num_heads, ff_dim)(x)
    x = TransformerBlock(embeding, num_heads, ff_dim)(x)
    x = TransformerBlock(embeding, num_heads, ff_dim)(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    out = layers.Dense(n_track, activation="sigmoid", name="classifier")(x)

    model = keras.Model(inputs=[inp], outputs=[out])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
        metrics = ['binary_accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model



# model = build_model(NUM_INPUT, NUM_OUTPUT, 512)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = build_model(NUM_INPUT, NUM_OUTPUT, filters = 512, num_heads = 11, ff_dim = 512, embeding = 64)


model.summary()




checkpointer = ModelCheckpoint(filepath = out_model, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
batch_size = 1024

train = get_dataset(train_files, batch= batch_size, num_threads = num_threads)
val = get_dataset(val_files, batch= batch_size, num_threads = num_threads)

history = model.fit(train, epochs=200, validation_data=val, callbacks=[checkpointer, earlystopper])