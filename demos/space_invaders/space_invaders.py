#!/usr/bin/env python

import math, time
import tkinter as tk
from tkinter import Tk, Canvas, PhotoImage, Label, Button

class Game(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        parent.title("Space Invaders")

        canvas, aliens, lasers = Canvas(parent, width=800, height=400, bg='black'), {}, {}
        canvas.pack()
        i1, i2 = PhotoImage(format = 'gif', file = "alien.gif"), PhotoImage(format = 'gif', file = "laser.gif")
        for x, y, p in [(100+40*j, 160-20*i, 100*i) for i in range(8) for j in range(15)]:
            aliens[canvas.create_image(x, y, image = i1)] = p
        canvas.bind('<Button-1>', lambda e: lasers.update({canvas.create_image(e.x, 390, image=i2): 10}))
        while aliens:
            try:
                for l in lasers:
                    canvas.move(l, 0, -5)
                    if canvas.coords(l)[1]<0:
                        canvas.delete(l); del lasers[l]
                for a in aliens:
                    canvas.move(a, 2.0*math.sin(time.time()),0)
                    p = canvas.coords(a)
                    items = canvas.find_overlapping(p[0]-5, p[1]-5, p[0]+5, p[1]+5)
                    for i in items[1:2]:
                        canvas.delete(a); del aliens[a]; canvas.delete(i); del lasers[i]
                time.sleep(0.02); root.update()
            except: pass

if __name__ == "__main__":
    root = tk.Tk()
    Game(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
