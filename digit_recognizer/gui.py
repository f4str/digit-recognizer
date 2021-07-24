import argparse
import io
import os
import sys
import tkinter as tk

from PIL import Image
import models
import numpy as np
import torch
import torch.nn.functional as F


class Canvas:
    def __init__(self, model: torch.nn.Module):
        self.root = tk.Tk()
        self.root.title('Digit Recognizer')
        self.root.configure(background='black', padx=5)
        self.root.resizable(False, False)

        self.predict_button = tk.Button(
            self.root,
            text='Predict',
            command=self.predict,
            bg='black',
            fg='white',
            width=20,
            padx=0,
            bd=4,
        )
        self.predict_button.config(font=('Calibri', 12))
        self.predict_button.grid(row=0)

        self.clear_button = tk.Button(
            self.root,
            text='Clear',
            command=self.clear,
            bg='black',
            fg='white',
            width=20,
            padx=0,
            bd=4,
        )
        self.clear_button.config(font=('Calibri', 12))
        self.clear_button.grid(row=0, column=1)

        self.canvas = tk.Canvas(self.root, bg='white', width=300, height=300)
        self.canvas.grid(row=1, columnspan=2)

        self.label = tk.Label(
            self.root,
            bg='black',
            fg='white',
            bd=4,
            height=1,
            width=10,
            justify=tk.CENTER,
            text='Prediction: ',
        )
        self.label.config(font=('Calibri', 20))
        self.label.grid(row=2, column=0)

        self.predict_label = tk.Label(
            self.root, bg='black', fg='white', bd=4, height=1, width=10, justify=tk.CENTER
        )
        self.predict_label.config(font=('Calibri', 20))
        self.predict_label.grid(row=2, column=1)

        self.old_x = None
        self.old_y = None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<B3-Motion>', self.erase)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.canvas.bind('<ButtonRelease-3>', self.reset)

        self.model = model

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def predict(self):
        ps = self.canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))

        data = np.array(img.resize((28, 28)).convert('L'))
        data = (255 - data) / 255
        tensor = torch.tensor([data], dtype=torch.float32)

        with torch.no_grad():
            out = self.model(tensor).squeeze()
        prediction = torch.argmax(out).item()
        confidence = int(F.softmax(out, dim=0)[prediction].item() * 100)

        self.predict_label.config(text=f'{prediction} ({confidence}%)')

    def clear(self):
        self.canvas.delete('all')
        self.predict_label.config(text='')

    def paint(self, event):
        self.predict_label.config(text='')
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=10.0,
                fill='black',
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
                splinesteps=36,
            )
        self.old_x = event.x
        self.old_y = event.y

    def erase(self, event):
        self.predict_label.config(text='')
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=20.0,
                fill='white',
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
                splinesteps=36,
            )
        self.old_x = event.x
        self.old_y = event.y

    def start(self):
        self.root.mainloop()


def get_args():
    parser = argparse.ArgumentParser(description='GUI for testing MNIST')
    parser.add_argument('--name', '-n', type=str, required=True, help='name of the model')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='convolutional',
        help='type of model to use',
        choices=['feedforward', 'ffnn', 'convolutional', 'cnn', 'recurrent', 'rnn'],
    )

    args = parser.parse_args()
    return args


def load_model(model_name, name):
    path = os.path.join('saved_models', name)
    model = models.get_model(model_name)

    if os.path.exists(os.path.join(path, 'model.pt')):
        ckpt = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(ckpt['state_dict'])
    else:
        sys.exit('Saved model not found')

    return model


def main(args):
    print('Loading model')
    model = load_model(args.model, args.name)

    print('Opening canvas')
    canvas = Canvas(model)
    canvas.start()


if __name__ == '__main__':
    args = get_args()
    main(args)
