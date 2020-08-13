import json
import os
import keras
from keras.layers import Dense, Input, LSTM, Embedding, Reshape, Dropout, Lambda
from keras.optimizers import RMSprop
from keras.models import Model, load_model
import keras.backend as K
import numpy as np
import datetime

c2n = open('./dataset/char2num.json', 'r', encoding='utf-8')
n2c = open('./dataset/num2char.json', 'r', encoding='utf-8')

char2num = json.load(fp=c2n)
num2char = json.load(fp=n2c)

del c2n, n2c


def argmax(inp):
    out = K.argmax(inp, axis=-1)
    return out


def inverse_argmax(num,l):
    temp=[0]*l
    temp[num]=1
    return temp


class dataGenerator:
    def __init__(self, fp='./dataset/dataset.txt'):
        self.content = open(fp, 'r', encoding='utf-8').readlines()

    def get(self, batch_size):
        idx = np.random.randint(0, len(self.content), size=batch_size)
        out = []
        l = len(char2num)
        for i in idx:
            temp = []
            s = self.content[i].replace('\n', '')
            for c in s:
                temp.append(inverse_argmax(int(char2num.get(c, '0')), l))
            out.append(temp)
        return np.array(out)


class GAN:
    def __init__(self,
                 lr=0.0001,
                 batch_size=500,
                 log_path='./log.txt',
                 word_count=len(char2num)):
        self.lr = lr
        self.batch_size = batch_size
        self.log = open(log_path, 'w', encoding='utf-8')
        self.word_count = word_count
        self.optimizer = RMSprop(lr=self.lr)

    def build(self):
        if os.path.exists('./model/gen.h5') and os.path.exists(
                './model/dis.h5'):
            self.gen = load_model('./model/gen.h5')
            self.dis = load_model('./model/dis.h5')
            self.print_log('Load Models')
        else:
            self.gen = self.build_gen()
            self.dis = self.build_dis()
            self.print_log('New Models')

        self.gen.summary()
        self.dis.summary()

        self.gen.trainable = False
        self.dis.trainable = True
        noise = Input(shape=(8))
        fake_poem = self.gen(noise)
        real_poem = Input(shape=(7, self.word_count))
        fake_out = self.dis(fake_poem)
        real_out = self.dis(real_poem)
        self.d_train_model = Model(inputs=[real_poem, noise],
                                   outputs=[real_out, fake_out, real_out])
        self.d_train_model.compile(optimizer=self.optimizer,
                                   loss=['mse', 'mse', 'binary_crossentropy'],
                                   loss_weights=[1, 1, 10])

        self.gen.trainable = True
        self.dis.trainable = False
        noise_gen = Input(shape=(8))
        out = self.dis(self.gen(noise_gen))
        self.g_train_model = Model(noise_gen, out)
        self.g_train_model.compile(optimizer=self.optimizer, loss='mse')

        noise_p = Input(shape=(8))
        pre = self.gen(noise_p)
        pre = Lambda(argmax)(pre)
        self.predict_model = Model(noise_p, pre)

    def train(self, epoch=10000, sample_interval=5, sample_num=8):
        data = dataGenerator()
        start_time = datetime.datetime.now()
        self.print_log('Training Start At {}'.format(start_time))
        real_out = np.ones([self.batch_size, 1])
        fake_out = np.zeros([self.batch_size, 1])
        gp_out = np.ones([self.batch_size, 1])
        for i in range(epoch):
            real_poem = data.get(self.batch_size)
            noise = np.random.normal(0.0, 0.1, size=[self.batch_size,
                                                     8]).astype('float32')
            d_loss = self.d_train_model.train_on_batch(
                [real_poem, noise], [real_out, fake_out, gp_out])
            g_loss = self.g_train_model.train_on_batch(noise, real_out)
            if i % sample_interval == 0:
                self.print_log('=== Epoch {} ==='.format(i))
                self.print_log('d loss: {}'.format(d_loss))
                self.print_log('g loss: {}'.format(g_loss))
                self.gen.save('./model/gen.h5')
                self.dis.save('./model/dis.h5')
                self.print_log(self.sample(sample_num))
        end_time = datetime.datetime.now()
        self.print_log('Training End At {}'.format(end_time))
        self.print_log('Time Cost : {}'.format(end_time - start_time))
        self.print_log('=== Final Sample ===')
        self.print_log(self.sample(sample_num))

    def build_gen(self):
        inp = Input(shape=(8))  # (8)
        out = Dense(16)(inp)  # (16)
        out = Dense(32)(out)  # (32)
        out = Dense(64)(out)  # (64)
        out = Dense(56)(out)  # (56)
        out = Reshape((7, 8))(out)  # (7,8)
        out = LSTM(1024, return_sequences=True)(out)  # (7, 1024)
        out = Dense(self.word_count)(out)  # (7, word_count)
        gen = Model(inputs=inp, outputs=out)
        return gen

    def build_dis(self):
        inp = Input(shape=(7, self.word_count))  # (7, word_count)
        out = LSTM(1024)(inp)  # (1024)
        out = Dense(1024)(out)  # (1024)
        out = Dense(512)(out)  # (512)
        out = Dropout(0.6)(out)  # (64)
        out = Dense(1)(out)  # (1)
        dis = Model(inputs=inp, outputs=out)
        return dis

    def sample(self, size):
        noise = np.random.normal(0.0, 0.1, size=[size, 8]).astype('float32')
        pred = self.predict(noise=noise)
        ret = ''
        for i in range(size):
            for j in range(7):
                ret += num2char.get(str(pred[i][j]), '0')
            ret += '\n'
        return ret

    def print_log(self, s):
        print(s)
        self.log.write(s + '\n')
        self.log.flush()

    def predict(self, noise):
        return self.predict_model.predict(noise)


if __name__ == '__main__':
    gan = GAN(batch_size=5)
    gan.build()
    gan.train()
