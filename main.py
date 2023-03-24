import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.layers.core import *
from keras.models import Model
from data_helper import *
from evaluation import *
from numpy.random import seed
from keras.optimizers import adam_v2
from keras.optimizers import adagrad_v2
from run_metrics import *
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.ops import array_ops
import MiniAttention.MiniAttention as MA
from attention import Attention
import tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed(1)
# tf.random.set_seed(2)

tf.set_random_seed(2)





if __name__ == '__main__':
    # df_train = pd.read_csv('F:/ChatEO/Chhh/data/answer/clojurians/train.csv')
    # df_train = pd.read_csv('F:/ChatEO/Chhh/data/answer/elmlangB_2017/train.csv')
    # df_train = pd.read_csv('F:/ChatEO/Chhh/data/answer/pythondev/train.csv')
    df_train = pd.read_csv('../../data/answer/train_set.csv')
    df_test = pd.read_csv('../../data/answer/test_set.csv')
    #df_test = pd.read_csv('F:/ChatEO/Chhh/data/answer/pythondev/test.csv')
    # df_test = pd.read_csv('F:/ChatEO/Chhh/data/answer/clojurians/test.csv')
    #df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/clojurians/test.csv')
    #df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/elmlangB_2017/test.csv')

    #df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/pythondev/test.csv')
    #df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/irc_angularjs/test.csv')
    #df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/irc_opengl/test.csv')
    df_tt = pd.read_csv('F:/ChatEO/Chhh/data/answer/irc_c++-general/test.csv')


    def binary_focal_loss(gamma, alpha):
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

            p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
            return K.mean(focal_loss)

        return binary_focal_loss_fixed


    def DSC_loss(y_true, y_pred):  # https://www.cnblogs.com/hotsnow/p/10954624.html
        soomth = 0.5
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred_rev = tf.subtract(1.0, y_pred)
        y_true = tf.cast(y_true, dtype=tf.float32)
        nominator = tf.multiply(tf.multiply(2.0, y_pred_rev), y_pred) * y_true
        denominator = tf.multiply(y_pred_rev, y_pred) + y_true
        dsc_coe = tf.subtract(1.0, tf.divide(nominator, denominator))
        return tf.reduce_mean(dsc_coe)

    embedding_dim = 200  # dim of our learned embedding model
    X_train, Y_train, X_test, Y_test, sample_weights_train, sample_weights_test = prep_data(df_train, df_test,
                                                                                            embedding_dim, df_tt)
    utterance_length = 40  # avg QA-combination utterance length
    conversation_length = 20  # avg conversation length
    inputs = Input(shape=(conversation_length, utterance_length, embedding_dim))

    # bilstm = MA.MiniAttentionBlock(keras.initializers.he_uniform, None, None, keras.regularizers.L2(l2=0.02), None,None, None, None, None)(inputs)
    # x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
    # CNN Layer
    num_filters = 50
    reshape = Reshape((conversation_length, utterance_length, embedding_dim, 1))(inputs)
    conv_0_3 = TimeDistributed(Conv2D(num_filters, kernel_size=(2, embedding_dim), activation='relu'),
                               input_shape=(1, conversation_length, utterance_length, embedding_dim, 1))(reshape)
    maxpool_0_3 = TimeDistributed(MaxPool2D(pool_size=(2, 1), padding='valid'))(conv_0_3)
    conv_1_3 = TimeDistributed(Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu'),
                               input_shape=(1, conversation_length, utterance_length, embedding_dim, 1))(reshape)
    maxpool_1_3 = TimeDistributed(MaxPool2D(pool_size=(2, 1), padding='valid'))(conv_1_3)
    conv_2_3 = TimeDistributed(Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu'),
                               input_shape=(1, conversation_length, utterance_length, embedding_dim, 1))(reshape)
    maxpool_2_3 = TimeDistributed(MaxPool2D(pool_size=(3, 1), padding='valid'))(conv_2_3)
    concatenated_tensor = Concatenate(axis=2)([maxpool_0_3, maxpool_1_3, maxpool_2_3])
    flatten = TimeDistributed(Flatten())(concatenated_tensor)
    output = Dropout(0.5)(flatten)

    # biLSTM Layer
    #bilstm = MA.MiniAttentionBlock(keras.initializers.glorot_uniform, None, None, keras.regularizers.L2(l2=0.02), None,None, None, None, None)(output)
    #bilstm = MA.MiniAttentionBlock(None, None, None, None, None, None, None, None, None)(output)
    #bilstm = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(bilstm)
    bilstm = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(output)  # variational biLSTM)
    bilstm = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(bilstm)
    #bilstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(bilstm)
    #bilstm = MA.MiniAttentionBlock(keras.initializers.he_uniform, None, None, keras.regularizers.L2(l2=0.02), None, None, None, None, None)(bilstm)
    bilstm = MA.MiniAttentionBlock(keras.initializers.glorot_uniform, None, None, keras.regularizers.L2(l2=0.02), None,None, None, None, None)(bilstm)
    #bilstm = MA.MiniAttentionBlock(None, None, None, None, None,None, None, None, None)(bilstm)
    # bilstm = attention_3d_block(bilstm)
    # bilstm = TimeDistributed(Flatten())(bilstm)
    outputs = TimeDistributed(Dense(1, activation="sigmoid"))(bilstm)
    #outputs = Dense(1, activation="sigmoid")(bilstm)
    model = Model(inputs=inputs, outputs=outputs)
    #model = load_model("F:/ChatEO/ReplicationPackage-ChatEO/model_file_path.h5", custom_objects={'DSC_loss': DSC_loss})

    opt1 = adagrad_v2.Adagrad(learning_rate=0.005)
    opt = adam_v2.Adam(lr=0.001)
    #model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"], sample_weight_mode='temporal')
    model.compile(optimizer=opt, loss=[binary_focal_loss(alpha=.60, gamma=2)], metrics=["binary_accuracy"])
    #model.compile(optimizer=opt, loss=[DSC_loss], metrics=["binary_accuracy"])
    print(model.summary())
    # model_save_path = "F:/ChatEO/Chhh/change_model.h5"
    # 保存模型
    # model.save(model_save_path)
    # model = load_model(model_save_path)
    # Evaluation
    train_test_validate(model, X_train, Y_train, X_test, Y_test, sample_weights_train, sample_weights_test)
