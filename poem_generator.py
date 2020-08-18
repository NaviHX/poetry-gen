from keras.models import load_model, Model
from keras.layers import Reshape, Input
from tkinter import *
import json
import numpy as np

c2n = open('./dataset/char2num.json', 'r', encoding='utf-8')
n2c = open('./dataset/num2char.json', 'r', encoding='utf-8')

char2num = json.load(fp=c2n)
num2char = json.load(fp=n2c)

amount = len(char2num)

del c2n, n2c

class poemGenerator:
    def __init__(self):
        self.word_count=amount
        self.model=load_model('./model/gen.h5')
        noise_p = Input(shape=(7))
        pre = self.model(noise_p)
        pre = Reshape((7, ))(pre)
        self.predict_model = Model(noise_p, pre)

    def predict(self, noise):
        return self.predict_model.predict(noise)

    def get(self,count=8,temperature=1.00,seed=None):
        ret = ''
        if seed is None:
            seed = np.random.randint(0, self.word_count, size=[1, 7])
        else:
            noise = [char2num.get(c, 0) for c in seed]
            if len(noise) < 7:
                noise.extend([0] * (7 - len(noise)))
            elif len(noise) > 7:
                noise = noise[0:7]
            seed = np.array([noise])
        l = [seed]
        for i in range(count):
            pred = self.predict(l[i])
            l.append(np.abs(np.rint(pred * self.word_count)))
            for j in range(7):
                ret += num2char.get(
                    str(
                        int(
                            np.abs(
                                np.rint(pred[0][j] * self.word_count *
                                        temperature)))), ' ')
            ret += '\n'
        return ret

class GUI:
    def __init__(self):
        self.window=Tk()
        self.gen=poemGenerator()

    def set_window(self):
        self.window.title("PoemGenerator By Keras")
        self.window.geometry('300x200+10+10')
        
        self.input_label=Label(self.window,text='输入起始诗句')
        self.input_label.grid(row=0,column=0)

        self.output_label=Label(self.window,text='预测诗句')
        self.output_label.grid(row=0,column=12)

        self.input_text=Text(self.window, width=15,height=2)
        self.input_text.grid(row=1,column=0)

        self.output_text=Text(self.window, width=15,height=10)
        self.output_text.grid(row=1,column=12,rowspan=10,columnspan=10)

        self.button=Button(self.window,text='生成',bg='lightblue',width=10,command=self.generate)
        self.button.grid(row=2,column=0)

    def generate(self):
        seed=self.input_text.get(1.0,END).strip().replace("\n","")
        if seed:
            try:
                ret=self.gen.get(count=7,temperature=1.00,seed=seed)
                self.output_text.delete(1.0,END)
                self.output_text.insert(1.0,ret)
            except:
                self.output_text.delete(1.0,END)
                self.output_text.insert(1.0,'Unexpected Error !!!')
        else:
            self.output_text.delete(1.0,END)
            self.output_text.insert(1.0,'请输入起始诗句')

gui=GUI()
gui.set_window()
gui.window.mainloop()
