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

curdoc().theme = "dark_minimal"
# deepskyblue, orangered, coral, red

class DDS:
    
    def __init__(self, d=2, h=3):
        
        self.d = d
        self.h = h
        self.step = 1e-1
    
    def single_dynamic(self, start):
        
        d = self.d
        h = self.h
        # start = round(start, 5)
        orbit = [start]
        reset = False
        n = 0
        point = start
        while n <= 1e4:
            if point < 0:
                point += d
            elif point > 1:
                point -= h
            else:
                reset = True
                break
            # point = round(point, 5)
            if point in orbit:
                cycle_length = n - orbit.index(point)
                break
            else:
                orbit.append(point)
            n += 1
        
        if n >= 1e4:
            print('WTF-point:', start)
        
        if reset:
            return reset, n
        else:
            return reset, cycle_length
        
    
    def test(self):
        
        d = self.d
        h = self.h
        step = self.step
        rad = 2*max(d, h)
        x_axis_reset = []
        x_axis_cycle = []
        y_axis_reset_length = []
        y_axis_cycle_length = []
        x = -rad
        while x <= rad:
            reset, n = self.single_dynamic(x)
            if reset:
                x_axis_reset.append(x)
                y_axis_reset_length.append(n)
            else:
                x_axis_cycle.append(x)
                y_axis_cycle_length.append(n)
            x += step
        
        self.x_axis_reset = x_axis_reset
        self.x_axis_cycle = x_axis_cycle
        self.y_axis_reset_length = y_axis_reset_length
        self.y_axis_cycle_length = y_axis_cycle_length
        
        
    
    def show(self):
        
        d = self.d
        h = self.h
        
        f = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        
        rad = 2*max(d, h)
        
        f.line([-rad, rad], [-rad, rad], color='blue')
        
        f.line([-rad, 0], [-rad + d, d], color='orangered')
        f.line([1, rad], [1-h, rad - h], color='orangered')
        
        if 'x_axis_reset' not in dir(self):
            self.test()
            
        f.circle(self.x_axis_reset, [-rad]*len(self.x_axis_reset), color='red', size=1)
        f.circle(self.x_axis_cycle, [-rad]*len(self.x_axis_cycle), color='deepskyblue', size=1)
        
        hysto = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        
        hysto.vbar(x=self.x_axis_reset, top=self.y_axis_reset_length, color='red', width=0.7*self.step)
        hysto.vbar(x=self.x_axis_cycle, top=self.y_axis_cycle_length, color='deepskyblue', width=0.7*self.step)
        
        
        grid = gridplot([[f, hysto]], width=2500, height=2500)
        
        boxes = Information_Boxes(self)
        up = boxes.up
        legend = boxes.legend

        grid.toolbar_location = 'below'
        grid.sizing_mode = 'scale_both'
        grid.rows = str(100//1) + '%'
        grid.cols = str(100//2) + '%'
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
    
    def __init__(self, dds=DDS()):
        
        # Сверху
        self.up = figure(tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover")
        self.up.height = 20
        self.up.sizing_mode = 'scale_width'
        self.up.title = 'title_text'
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
        self.legend.title = self.make_legend(dds)
        self.legend.title.text_font_size = '15px'
    
    def make_legend(self, dds=DDS()):
        
        title = 'Information:\n'
            
        title += f'd = {dds.d}\n'
        title += f'h = {dds.h}\n'
        
        return title
    
    
    
def test():
    
    d = 1 + 100*random()
    h = 1 + 100*random()
    # d = 6
    # h = 3
    
    dds = DDS(d, h)
    
    dds.show()









