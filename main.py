import numpy as np
from numpy import linalg as LA
import math
from random import random, gauss
import itertools
from bokeh.plotting import figure, show, output_file, save
from bokeh.io import curdoc, export_png
from bokeh.layouts import gridplot, column, row
from bokeh.models import Range1d, Title, TabPanel, Tabs, ColumnDataSource, Legend
from bokeh.models.tools import BoxZoomTool, ResetTool, PanTool
from copy import copy, deepcopy
from time import time

# Темы для графиков
# themes: caliber, dark_minimal, ligth_minimal, nigth_sky, and contrast
# curdoc().theme = "dark_minimal"

def all_subsets(array):
    
    n = len(array)
    M = []
    for j in range(2**n):
        sbin = bin(j)[2:]
        if len(sbin) < n:
            sbin = '0'*(n-len(sbin)) + sbin
        S = []
        for i in range(n):
            if sbin[i] == '1':
                S.append(array[i])
        S = tuple(S)
        M.append(S)
    return M


def matrix_check(h, w):
    
    if len(h) != 2 or w[0] > 0:
        return
    h1, h2 = h
    w1, w2 = w
    if  h1 > h2:
        if (h1 - h2) % (h2 - w2) == 0:
            print('Unstable matrix! Case 1')
            return
        if (h1 - h2 - w1) % (h2 - w2) == 0:
            print('Unstable matrix! Case 2')
            return
    else:
        if h1 - h2 == w1:
            print('Unstable matrix! Case 3')
            return
        if (h2 - h1) % (h1 - w1) == 0:
            print('Unstable matrix! Case 4')
            return
        
        

class Spike_Matrix:
    
    def __init__(self, dim=2, exc=0):
        
        self.dim = dim
        self.exc = exc
        self.noise = False
        self.generate()
    
    
    def generate(self):
        
        n = self.dim
        B = np.zeros((n, n))
        H, w = 100 * np.random.random(n), np.random.random(n)
        if self.exc > 0:
            w_exc = -100*np.random.random(self.exc)
            for i in range(self.exc):
                w[i] = w_exc[i]
                B[i][i] = H[i]
                for j in range(n):
                    if j != i:
                        B[i][j] = w_exc[i]
        for i in range(self.exc, n):
            w[i] *= H[i] 
            B[i][i] = H[i]
            for j in range(n):
                if j!=i:
                    B[i][j] = w[i]
        
        B = np.transpose(B)
        
        for i in range(n):
            H[i] = round(H[i], 2)
            w[i] = round(w[i], 2)
            for j in range(n):
                B[i][j] = round(B[i][j], 2)
        
        self.matrix = B
        self.h = H
        self.w = w
    
    def set_manually(self, h, w):
        
        h = np.array(h)
        w = np.array(w)
        n = len(w)
        exc = 0
        while w[exc] < 0 and exc < n:
            exc += 1
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                B[i][j] = w[i]
            B[i][i] = h[i]
        
        B = np.transpose(B)
        self.dim = n
        self.exc = exc
        self.h = h
        self.w = w
        self.matrix = B
        
        matrix_check(h, w)
        
    
    def get_noise(self):
        
        if not self.noise:
            return 0
        
        noise = 2*random()-1
        # noise = gauss(0, 1)
        # scale = 1e1
        # scale = 0.5
        scale = 1
        return noise / scale
        
    
    def h_pulse(self, index):
        
        pulse = self.h[index] + self.get_noise()
        return pulse
    
    def w_pulse(self, index):
        
        pulse = self.w[index] + self.get_noise()
        return pulse
    
    def show(self):
        
        print('Spike Matrix:\n', self.matrix)
        print('H:', self.h)
        print('w:', self.w)
        
  

class Pies:
    
    def __init__(self, dim=2, subsets=all_subsets(range(0, 2))):
        
        self.pi_subsets = dict.fromkeys(subsets, 0)
        self.pi_self = np.zeros((dim,))
        self.pi_exc_all = 0
        self.pi_exc_it = np.zeros((dim,))
        self.pi_exc_notit = np.zeros((dim,))
        self.pi_exc_itnotit = np.zeros((dim, dim))
    
    def info(self):
        
        s = 'Pies:\n'
        dim = len(self.pi_exc_it)
        exc = dim
        for subset in self.pi_subsets:
            if len(subset) > 0 and exc > min(subset):
                exc = min(subset)
                
        s += f'Inh neurons: {list(range(exc, dim))}\n'
        
        s += f'pi_self = {list(map(lambda x: round(x, 4), self.pi_self))}\n'
        if exc == 0:
            return s
        for subset in self.pi_subsets:
            s += str(subset) + f': {round(self.pi_subsets[subset], 4)}\n'
        s += f'pi_exc_it = {list(map(lambda x: round(x, 4), self.pi_exc_it))}\n'
        s += f'pi_exc_notit = {list(map(lambda x: round(x, 4), self.pi_exc_notit))}\n'
        s += 'pi_exc_itnotit:\n'
        s += f'{self.pi_exc_itnotit[exc:,exc:]}\n'
        s += f'exc_all: {round(self.pi_exc_all, 4)}'
        # for i in range(exc, dim):
        #     s += f'pi_self_{i}: {round(self.pi_self[i], 4)}\n'
        # if exc == 0:
        #     return s
        # s += '\n'
        # for subset in self.pi_subsets:
        #     s += str(subset) + f': {round(self.pi_subsets[subset], 4)}\n'
        # s += '\n'
        # for i in range(exc, dim):
        #     s += f'pi_exc_it_{i}: {round(self.pi_exc_it[i], 4)}\n'
        # s += '\n'
        # for i in range(exc, dim):
        #     s += f'pi_exc_notit_{i}: {round(self.pi_exc_notit[i], 4)}\n'
        # s += '\n'
        # for i in range(exc, dim):
        #     for j in range(exc, dim):
        #         if i != j:
        #             s += f'pi_exc_itnotit_{i}{j}: {round(self.pi_exc_itnotit[i][j], 4)}\n'
        # s += '\n'
        # s += f'pi_exc_all: {round(self.pi_exc_all, 4)}'
        
        return s
        
        

class Testalt:

    def __init__(self, dim=3, time=10, step=0.1):

        self.matrix = Spike_Matrix(dim, 0)
        self.time = time
        self.step = step
        self.nsp = 0
        self.noise_scale = 1
        self.update()

    def get_noise(self):
        return self.noise_scale * gauss(0, self.step)

    def update(self):
        self.dim = self.matrix.dim
        self.exc = self.matrix.exc
        if 'start' not in dir(self):
            self.start = 10 * np.random.random(self.dim)

    def do(self):

        self.update()
        nsteps = int(self.time // self.step + 1)
        self.potentials = np.zeros((self.dim, nsteps + 1))
        self.potentials[:, 0] = self.start
        for i in range(nsteps):
            spiked = False
            for n in range(self.dim):
                if self.potentials[n][i] < self.step:
                    self.potentials[n][i] = 0
                    for m in range(self.dim):
                        self.potentials[m][i+1] = self.potentials[m][i] - self.step + self.get_noise()
                        if n == m:
                            self.potentials[m][i+1] += self.matrix.h_pulse(n)
                        else:
                            self.potentials[m][i+1] += self.matrix.w_pulse(n)
                    spiked = True
                    self.nsp += 1
                    break
            if not spiked:
                for n in range(self.dim):
                    self.potentials[n][i+1] = self.potentials[n][i] - self.step + self.get_noise()


    def explicit_show(self):

        dynamics = []
        # n = self.number_of_spikes
        # x_axis = list(range(n + 1))
        for i in range(self.dim):
            f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")

            title = f'Neuron {i}:\n'
            title += f'H = {self.matrix.h[i]}, w = {self.matrix.w[i]}\n'
            temp = figure()
            temp.line(legend_label=title)
            legend = temp.legend[0]
            color = 'deepskyblue'
            x_axis = []
            ltime = 0
            while ltime <= self.time:
                x_axis.append(ltime)
                ltime += self.step
            f.line(x=x_axis, y=self.potentials[i], color=color)
            x_range = Range1d(0, x_axis[-1])
            f.x_range = x_range
            f.add_layout(legend, 'right')
            f.title_location = 'right'

            # if self.dim == 2:
            #     y = [0] * len(self.spikes_subempty)
            #     f.circle(self.spikes_subempty, y, color='coral', alpha=0.8)
            #     y = [0] * len(self.spikes_2)
            #     f.circle(self.spikes_2, y, color='lightskyblue', alpha=0.8)
            #     y = [0] * len(self.spikes_sub2)
            #     f.circle(self.spikes_sub2, y, color='red', alpha=0.8)

            # f.title = title
            dynamics.append([f])

        grid = gridplot(dynamics, width=2500, height=2500)

        # boxes = Information_Boxes(self)
        # up = boxes.up
        # legend = boxes.legend

        grid.toolbar_location = 'below'
        grid.sizing_mode = 'scale_both'
        grid.rows = str(100 // self.dim) + '%'
        grid.cols = str(100 // 1) + '%'
        # for i in range(self.number_of_rows):
        #     for j in range(self.number_of_cols):
        #         if self.graphic_grid_for_show[i][j] != None:
        #             self.graphic_grid_for_show[i][j].width = 10000
        #             self.graphic_grid_for_show[i][j].height = 10000

        # grid.toolbar_location = None

        # self.content = column(up, row(grid, legend, sizing_mode='scale_both'))
        # self.content.sizing_mode = 'scale_both'

        # show(self.content)
        show(grid)


class Test:
    
    def __init__(self, dim=2, exc=0):
        
        self.matrix = Spike_Matrix(dim, exc)
        self.number_of_spikes = 1000
        self.update()
    
    def update(self):
        
        self.dim = self.matrix.dim
        self.exc = self.matrix.exc
        if 'start' not in dir(self):
            self.start = 10*np.random.random(self.dim)
        self.subsets = all_subsets(range(self.exc, self.dim))
        self.pies = Pies(self.dim, self.subsets)
    
    def do(self):
        
        self.update()
        self.potentials = np.zeros((self.dim, self.number_of_spikes + 1))
        self.potentials[:,0] = self.start
        if self.dim == 2:
            self.spikes_subempty = []
            self.spikes_sub2 = []
            self.spikes_2 = []
        self.times = np.zeros((self.number_of_spikes + 1,))
        
        for n in range(1, self.number_of_spikes + 1):
            
            index = self.potentials[:,n-1].argmin()
            interspike_time = self.potentials[index][n-1]
            self.times[n] = self.times[n-1] + interspike_time
            self.potentials[:,n] = self.potentials[:,n-1] - interspike_time
            if index >= self.exc:
                # print(index, self.potentials[:,n])
                if self.dim == 2:
                    self.spikes_2.append(n)
                if n >= self.number_of_spikes // 10:
                    self.pies.pi_self[index] += 1
                self.potentials[:,n] += self.matrix.w_pulse(index)
                self.potentials[index,n] = self.matrix.h_pulse(index)
            else:
                if n >= self.number_of_spikes // 10:
                    self.pies.pi_exc_all += 1
                all_neurons = set(range(self.dim))
                exc_neurons = list(range(self.exc))
                spiked_neurons = {index}
                self.potentials[index][n] = self.matrix.h_pulse(index)
                self.potentials[list(all_neurons.symmetric_difference(spiked_neurons)),n] += self.matrix.w_pulse(index)
                for _ in range(self.exc):
                    index = self.potentials[exc_neurons,n].argmin()
                    if self.potentials[index][n] <= 0:
                        spiked_neurons.add(index)
                        self.potentials[index][n] = self.matrix.h_pulse(index)
                        self.potentials[list(all_neurons.symmetric_difference(spiked_neurons)),n] += self.matrix.w_pulse(index)
                    else:
                        break
                
                spiked_inh_neurons = set()
                for index in range(self.exc, self.dim):
                    if self.potentials[index][n] <= 0:
                        spiked_inh_neurons.add(index)
                        spiked_neurons.add(index)
                        self.potentials[index][n] = self.matrix.h_pulse(index)
                
                if self.dim == 2:
                    if len(spiked_inh_neurons) == 0:
                        self.spikes_subempty.append(n)
                    else:
                        self.spikes_sub2.append(n)
                if n >= self.number_of_spikes // 10:
                    self.pies.pi_subsets[tuple(sorted(list(spiked_inh_neurons)))] += 1
                self.potentials[list(all_neurons.symmetric_difference(spiked_neurons)),n] += sum(self.matrix.w[list(spiked_inh_neurons)])
        
        self.pies.pi_self /= self.times[-1] - self.times[self.number_of_spikes//10 - 1]
        self.pies.pi_exc_all /= self.times[-1] - self.times[self.number_of_spikes//10 - 1]
        for subset in self.subsets:
            self.pies.pi_subsets[subset] /= self.times[-1] - self.times[self.number_of_spikes//10 - 1]
        for i in range(self.exc, self.dim):
            for subset in self.subsets:
                if i in subset:
                    self.pies.pi_exc_it[i] += self.pies.pi_subsets[subset]
                else:
                    self.pies.pi_exc_notit[i] += self.pies.pi_subsets[subset]
                
                for j in range(self.exc, self.dim):
                    if i in subset and j not in subset:
                        self.pies.pi_exc_itnotit[i][j] += self.pies.pi_subsets[subset]
    
    
    def show(self):
        
        dynamics = []
        n = self.number_of_spikes
        x_axis = list(range(n + 1))
        x_range = Range1d(n//10 , n + 1)
        for i in range(self.dim):
            f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
            
            title = f'Neuron {i}:\n'
            title += f'H = {self.matrix.h[i]}, w = {self.matrix.w[i]}\n'
            title += f'self = {round(self.pies.pi_self[i], 4)}\n'
            title += f'exc_it = {round(self.pies.pi_exc_it[i], 4)}\n'
            title += f'exc_notit = {round(self.pies.pi_exc_notit[i], 4)}\n'
            temp = figure()
            temp.line(legend_label=title)
            legend = temp.legend[0]
            color = 'orangered' * (i < self.exc) + 'deepskyblue' * (i >= self.exc)
            f.line(x=x_axis, y=self.potentials[i], color=color)
            f.x_range = x_range
            f.add_layout(legend, 'right')
            f.title_location = 'right'
            
            if self.dim == 2:
                y = [0] * len(self.spikes_subempty)
                f.circle(self.spikes_subempty, y, color='coral', alpha=0.8)
                y = [0] * len(self.spikes_2)
                f.circle(self.spikes_2, y, color='lightskyblue', alpha=0.8)
                y = [0] * len(self.spikes_sub2)
                f.circle(self.spikes_sub2, y, color='red', alpha=0.8)
            
            # f.title = title
            dynamics.append([f])
        
        grid = gridplot(dynamics, width=2500, height=2500)
        
        boxes = Information_Boxes(self)
        up = boxes.up
        legend = boxes.legend
        
        grid.toolbar_location = 'below'
        grid.sizing_mode = 'scale_both'
        grid.rows = str(100//self.dim) + '%'
        grid.cols = str(100//1) + '%'
        # for i in range(self.number_of_rows):
        #     for j in range(self.number_of_cols):
        #         if self.graphic_grid_for_show[i][j] != None:
        #             self.graphic_grid_for_show[i][j].width = 10000
        #             self.graphic_grid_for_show[i][j].height = 10000
        
        # grid.toolbar_location = None
        
        self.content = column(up, row(grid, legend, sizing_mode = 'scale_both'))
        self.content.sizing_mode = 'scale_both'
        
        show(self.content)
    
    def explicit_show(self):
        
        dynamics = []
        n = self.number_of_spikes
        x_axis = list(range(n + 1))
        # x_range = Range1d(n//10 , n + 1)
        for i in range(self.dim):
            f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
            
            title = f'Neuron {i}:\n'
            title += f'H = {self.matrix.h[i]}, w = {self.matrix.w[i]}\n'
            title += f'self = {round(self.pies.pi_self[i], 4)}\n'
            title += f'exc_it = {round(self.pies.pi_exc_it[i], 4)}\n'
            title += f'exc_notit = {round(self.pies.pi_exc_notit[i], 4)}\n'
            temp = figure()
            temp.line(legend_label=title)
            legend = temp.legend[0]
            color = 'orangered' * (i < self.exc) + 'deepskyblue' * (i >= self.exc)
            potentials = [self.start[i]]
            x_axis = [0]
            for s in range(n - 1):
                step = min(self.potentials[:, s])
                x_axis.append(x_axis[-1] + step)
                potentials.append(potentials[-1] - step)
                x_axis.append(x_axis[-1])
                potentials.append(self.potentials[i][s+1])
            # print(x_axis)
            f.line(x=x_axis, y=potentials, color=color)
            x_range = Range1d(0, x_axis[-1])
            f.x_range = x_range
            f.add_layout(legend, 'right')
            f.title_location = 'right'
            
            # if self.dim == 2:
            #     y = [0] * len(self.spikes_subempty)
            #     f.circle(self.spikes_subempty, y, color='coral', alpha=0.8)
            #     y = [0] * len(self.spikes_2)
            #     f.circle(self.spikes_2, y, color='lightskyblue', alpha=0.8)
            #     y = [0] * len(self.spikes_sub2)
            #     f.circle(self.spikes_sub2, y, color='red', alpha=0.8)
            
            # f.title = title
            dynamics.append([f])
        
        grid = gridplot(dynamics, width=2500, height=2500)
        
        boxes = Information_Boxes(self)
        up = boxes.up
        legend = boxes.legend
        
        grid.toolbar_location = 'below'
        grid.sizing_mode = 'scale_both'
        grid.rows = str(100//self.dim) + '%'
        grid.cols = str(100//1) + '%'
        # for i in range(self.number_of_rows):
        #     for j in range(self.number_of_cols):
        #         if self.graphic_grid_for_show[i][j] != None:
        #             self.graphic_grid_for_show[i][j].width = 10000
        #             self.graphic_grid_for_show[i][j].height = 10000
        
        # grid.toolbar_location = None
        
        self.content = column(up, row(grid, legend, sizing_mode = 'scale_both'))
        self.content.sizing_mode = 'scale_both'
        
        show(self.content)
        
        

class Information_Boxes:
    
    """ Блоки с информацией для вкладки """
    
    def __init__(self, test):
        
        # Сверху
        self.up = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        self.up.height = 20
        self.up.sizing_mode = 'scale_width'
        # self.up.title = 'title_text'
        self.up.title = 'Neurons dynamic'
        self.up.toolbar_location = None
        self.up.title.text_font_size = '25px'
        self.up.line(x=[], y=[])
        
        # # Ниже - информация по строкам и столбцам в таблице
        # self.for_table = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        # self.for_table.height = 30
        # self.for_table.sizing_mode = 'scale_width'
        # self.for_table.title = self.table_text()
        # self.for_table.title.align = 'center'
        # self.for_table.title.text_align = 'center'
        # self.for_table.toolbar_location = None
        # self.for_table.title.text_font_size = '15px'
        # self.for_table.line(x=[], y=[])
        
        # Справа - вся остальная информация
        self.legend = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        self.legend.width = 250
        self.legend.sizing_mode = 'stretch_height'
        self.legend.toolbar_location = None
        self.legend.line(x=[], y=[])
        self.legend.title = self.make_legend(test)
        self.legend.title.text_font_size = '15px'
    
    def make_legend(self, test=Test()):
        
        title = 'Information:\n'
            
        title += f'Dim = {test.matrix.dim}\n'
        title += f'Exc = {test.matrix.exc}\n'
        # title += f'Spike Matrix = {test.matrix.matrix}\n'
        title += f'H = {test.matrix.h}\n'
        title += f'w = {test.matrix.w}\n'
        title += f'Spikes = {test.number_of_spikes}\n'
        pies = test.pies.info()
        if test.dim > 3:
            pies = pies.split('pi_exc')
            pies = pies[0] + 'pi_exc' + pies[-1]
        title += pies
        
        return title
        



def t():
    # start = time()
    
    # test = Test(4, 1)
    # test = Test(3, 1)
    test = Testalt(2, 60)
    # test.number_of_spikes = 50
    # test.matrix.noise = True
    # test.matrix.set_manually([0.5, 91.5], [-0.1, 2])
    # test.matrix.set_manually([25, 70], [-0.1, 20])
    # test.matrix.set_manually([22, 20], [-1, 16.5])
    # test.matrix.set_manually([2, 5], [-3, 4])
    # test.matrix.set_manually([1, 6, 6, 6], [-8, 5, 5, 5])
    # test.start = [1, 6, 6, 20]
    # test.matrix.set_manually([4, 9, 9], [-3, 8, 8])
    # test.start = [1, 10, 50]
    # test.matrix.set_manually([4, 4, 4], [3, 3, 3])

    test.matrix.set_manually([4, 4], [3, 3])
    test.noise_scale = 3
    test.do()
    # test.show()
    test.explicit_show()
    # if test.matrix.h[0] % 1 != 0:
    #     print(f'H = {test.matrix.h}')
    #     print(f'w = {test.matrix.w}')
    # print(test.pies.info())
    # # test.show()
    # f = figure(sizing_mode='stretch_both')
    # f.circle(test.potentials[0], test.potentials[1])
    # show(f)
    
    # print(f'\n{round(time()-start, 3)} sec')
    print(test.nsp)
    return test
    
    
    
    
    
    
if __name__ == '__main__':
    
    t()
    
    pass
    
    
    
    
    
    
    
    

