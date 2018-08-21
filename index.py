#!/usr/bin/env python
# -*- coding: utf-8 -*-
#************************************************************
#*                    KID-ANALYSER                          *
#*                        INAOE                             *
#*                       index.py                           *
#*                 Programa principal                       *
#*               Marcial Becerril Tapia                     *
#*                    23/julio/2018                         *
#************************************************************

import os
import os.path
import sys
import time

from MAPx import movAproTrans

from PyQt4 import QtCore, QtGui,uic
from PyQt4.QtCore import *
from PyQt4.QtGui import QPalette,QWidget,QFileDialog,QMessageBox, QTreeWidgetItem, QIcon

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import fft

from dataRed import dataRed
from getQ import get_Q_factor
from enableThread import enableWindow
from loadThread import loadThread

from detector_peaks import detect_peaks
from scipy import signal

import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import(
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
          
#Main Window
class MainWindow(QtGui.QMainWindow):

    def __init__(self):

        QtGui.QMainWindow.__init__(self)

        # Original window size
        size_x_Original = 895
        size_y_Original = 564

        # Load of main window GUI
        # The GUI was developed in QT Designer 
        self.ui = uic.loadUi("gui/main.ui")

        # Full Screen
        self.ui.showMaximized()
        screen = QtGui.QDesktopWidget().screenGeometry()  

        # Screen dimensions
        self.size_x = screen.width()
        self.size_y = screen.height()

        # Ratio imagen/screen to scale widgets
        rel_x = float(self.size_x)/float(size_x_Original)
        rel_y = float(self.size_y)/float(size_y_Original)

        self.ui.setWindowFlags(self.ui.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.ui.setWindowFlags(self.ui.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)

        # Adjust frame to be responsive
        # TODO. It doesn't work correctly, is not responsive as I expect
        x_plot = 175
        y_plot = 200

        nx = self.size_x - x_plot - 55
        ny = self.size_y - y_plot

        self.ui.plotFrame.resize(int(nx),int(ny))
        self.ui.fileBar.resize(int(nx),41)
        self.ui.currentDiry.resize(int(nx)-45,31)

        # Frame Control
        x_ctrl = 10
        y_ctrl = 10

        cx = 201
        cy = self.size_y - y_ctrl - 295

        # Menu bar
        # File
        # Open directory
        self.ui.actionAbrir.triggered.connect(self.openApp)
        # Exit
        self.ui.actionSalir.triggered.connect(self.closeApp)

        # MAP
        self.ui.actionMAP.triggered.connect(self.loadMAP)

        # Tools bar
        # Open directory
        self.ui.actionOpen.triggered.connect(self.openApp)
        # Load KIDs files
        self.ui.actionLoad.triggered.connect(self.loadFullData)
        # Clear Tree
        self.ui.actionClear.triggered.connect(self.clearKidTree)
        # Load MAP (Moving Aproximation Transform) functions 
        self.ui.actionMAPbar.triggered.connect(self.loadMAP)

        # Plot
        self.ui.actionPlot.triggered.connect(self.getPlot)

        # Plot settings
        # Many colors
        self.ui.manyColors.mousePressEvent = self.manyCol
        # One color, different tranparencies
        self.ui.oneColor.mousePressEvent = self.oneCol

        self.ui.manyColors.setChecked(True)

        # Plot sweep, speed and IQ circle
        self.ui.actionVNA_Plots.triggered.connect(self.groupPlotsVNA)
        # Plot time stream I, Q and df
        self.ui.actionHomodynePlots.triggered.connect(self.groupPlotsHomo)
        # Plot all time stream (before the median)
        self.ui.actionAll_time_stream.triggered.connect(self.plotAllTS)
        # Plot I/Q High resolution 
        self.ui.actionIQ_HR.triggered.connect(self.IQ_HR)
        # Plot I/Q Low resolution 
        self.ui.actionIQ_LR.triggered.connect(self.IQ_LR)

        # Plot group 
        self.ui.actionVNA_Sweep.triggered.connect(self.plotVnaSweep) 

        # Save figures for all the directories files
        self.ui.actionCreateImages.triggered.connect(self.saveFigures)

        # Class to reduce data
        self.dataRedtn = dataRed() 

        # Class to get Q factor
        self.getQ = get_Q_factor()

        self.path = []
        self.allPaths = []
        self.allNames = []
        self.allShortNames = []
        self.KIDS = []
        self.nKIDS = 0

        self.flagResize = True
        self.now = ""
        self.last = ""

        self.ui.actionCosRay.setChecked(True)
        
        # Creation of Plot
        self.fig1 = Figure()
        self.addmpl(self.fig1)

        # To use LATEX in plots
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

        # Thread to show a diferent process while the machine is working
        self.enableW = enableWindow() 
        self.enableW.flagEnable.connect(self.plotData)

        self.threadLoad = loadThread() 
        self.threadLoad.flagLoad.connect(self.loadThread)

        self.ui.show() 

    # --- Plot full VNA sweep
    def plotVnaSweep(self):

        if len(self.allPaths) == 0:
            self.messageBox("Warning","Directory not selected")
            self.ui.setEnabled(True)
            return
        self.ui.setEnabled(False)
        self.messageBox("Info","The VNA sweep will be loaded, it would takes several minutes")

        try:
            self.fig1.clf()
        except:
            pass

        p = ""
        if self.ui.pointPlot.isChecked():
            p = "*"

        try:
            self.f1 = self.fig1.add_subplot(111)
            for path in self.path:
                vnaPath = path + "/VNA_Sweeps/"
                files = os.listdir(vnaPath)
                for file in files:
                    name = ""
                    for l in file:
                        if l == "_":
                            name = name + ","
                        else:
                            name = name + l  

                    freq, mag = self.dataRedtn.get_full_vna(str(vnaPath + file))

                    self.f1.set_ylabel(r'$S_{21}[V]$')
                    
                    if self.ui.ylog.isChecked():
                        mag_n = 20*np.log10(mag) + 10
                        self.f1.set_ylabel(r'$S_{21}[dBm]$')
                        self.f1.plot(freq,mag_n,p+'-',alpha=1,label=r"$"+str(name)+"$")
                    else:
                        self.f1.plot(freq,mag,p+'-',alpha=1,label=r"$"+str(name)+"$")
                    self.f1.set_xlabel(r'$Frequency [Hz]$')

            if self.ui.actionFindResonance.isChecked():

                ind = self.findResonance(mag, 0.02, 100)
                self.ui.statusText.setText("KIDS = " + str(len(ind)) + " founded")

                for i in ind:
                    self.f1.axvline(freq[i],color='r',linewidth=0.75)  
                    self.f1.plot(freq[i],mag[i],'bo')
                    self.f1.annotate(r"$"+str(freq[i]/1e6)+"MHz$",xy=(freq[i],mag[i]))

            if self.ui.squareLeg.isChecked():
                self.f1.legend(loc='best')

            self.f1.figure.canvas.draw()
        except:
            self.messageBox("Error","Error loading VNA sweep")

        self.ui.setEnabled(True)

        if self.flagResize == False:
            self.resizeTree(self.flagResize)

    # --- Plot resonance frequencies.
    #   Created by Salvador Ventura
    def findResonance(self, mag, mph, mpd):
        # Try negate mag, i.e. -mag
        ind = detect_peaks(signal.detrend(mag),valley=True,mph=mph,mpd=mpd)

        return ind 

    def plotAllTS(self):
        p = ""
        c1 = ""
        c2 = ""
        alpha = 1
        alpha_on = 0.75

        plotPar = [c1,c2,alpha_on,p]

        paths, shortName, namePlot, headPath = self.kidSelected()

        if len(paths) == 0:
            self.messageBox("Error","Select at least one KID to plot")    
            self.ui.setEnabled(True)
            return

        try:
            self.fig1.clf()
        except:
            pass

        self.f1 = self.fig1.add_subplot(111)

        num = 0
        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                self.ui.setEnabled(True)
                return

            ind_cut_freq = np.where(psd[0]>1000.)[0][0]
           
            for i in range(len(psd_low[1])):

                psdON, psdOFF, psdFreqON, psdFreqOFF = [], [], [], []

                psdON = np.concatenate((psd_low[1][i][1:],psd[1][i][ind_cut_freq:-1]),axis=0)
                psdOFF = np.concatenate((psd_low_OFF[1][i][1:],psd_OFF[1][i][ind_cut_freq:-1]),axis=0)
                
                psdFreqON = np.concatenate(( psd_low[0][1:], psd[0][ind_cut_freq:-1]),axis=0)
                psdFreqOFF = np.concatenate(( psd_low_OFF[0][1:], psd_OFF[0][ind_cut_freq:-1]),axis=0) 
                
                psdON = 10*np.log10(psdON)
                psdOFF = 10*np.log10(psdOFF)

                y = self.f1.semilogx(psdFreqON, psdON, c1+p+'-', alpha=alpha, label=r"$"+"Plot "+str(i)+"$")
                c2 = y[0].get_color()
                
                self.f1.semilogx(psdFreqOFF,psdOFF, color=c2,marker=p, alpha=alpha_on)

                self.f1.set_xlabel(r'$Frequency [Hz]$')
                self.f1.set_ylabel(r'$PSD \ df[dB]$')

            num += 1

        self.f1.figure.canvas.draw()

        if self.flagResize == False:
            self.resizeTree(self.flagResize)

    def resizeTree(self, doRes):

        size_x = 201
        size_y = 361

        size_x_tree = 191
        size_y_tree = 351

        # Show IQ colors
        size_x_IQ = 201
        size_y_IQ = 361

        size_x_IQ_tree = 191
        size_y_IQ_tree = 351

        if doRes == True:
            # Resize KID tree to show Color widget
            self.ui.treeKID.resize(size_x_tree,size_y_tree/2 - 5)
            self.ui.controlFrame.resize(size_x,size_y/2)

            self.ui.treeIQ.resize(size_x_IQ_tree,size_y_IQ_tree/2 - 10)
            self.ui.controlIQ.resize(size_x_IQ,size_y_IQ/2 - 5)
            self.ui.controlIQ.move(10,size_y_IQ/2 + 15)

            self.flagResize = False
        else:
            # Resize KID tree to hide Color widget
            self.ui.treeKID.resize(size_x_tree,size_y_tree)
            self.ui.controlFrame.resize(size_x,size_y)

            self.ui.treeIQ.resize(size_x_IQ_tree,size_y_IQ_tree)
            self.ui.controlIQ.resize(size_x_IQ,size_y_IQ)
            self.ui.controlIQ.move(10,10)

            try:
                self.clearTree(self.ui.treeIQ, False)
            except:
                pass

            self.flagResize = True

    # Plot IQ time stream High Resolution  
    def plotIQ_HR_TS(self):

        paths, shortName, namePlot, headPath = self.kidSelected()

        if len(paths) == 0:
            self.messageBox("Error","Select at least one KID to plot")    
            self.ui.setEnabled(True)
            return

        create_Tree = False

        if self.flagResize == True:
            self.resizeTree(self.flagResize)
            create_Tree = True

        try:
            self.fig1.clf()
        except:
            pass

        num = 0
        tree = self.ui.treeIQ
        tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                self.ui.setEnabled(True)
                return

            # I
            i_HR_ON = psd[10]
            i_HR_OFF = psd_OFF[10]

            # Q
            q_HR_ON = psd[11]
            q_HR_OFF = psd_OFF[11]

            self.f1 = self.fig1.add_subplot(121)

            number = len(i_HR_ON)
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, number)]

            print "ON"
            print "++++"
            print len(i_HR_ON),len(q_HR_ON)
            print "++++"

            for i in range(len(i_HR_ON)):

                if create_Tree == False:
                    item = self.ui.treeIQ.invisibleRootItem()
                    signal_count = item.childCount()
                    child = item.child(i)
                    if child.isSelected():
                        self.f1.plot(psd[3], i_HR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                        self.f1.plot(psd[3], q_HR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                else:

                    item = QTreeWidgetItem(tree)
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)
                    item.setText(0, "Signal HR ON:" + str(i))  
                    item.setBackground(0, QtGui.QColor(int(255*colors[i][0]),int(255*colors[i][1]),int(255*colors[i][2]),int(255*colors[i][3])))
                    
                    self.f1.plot(psd[3], i_HR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                    self.f1.plot(psd[3], q_HR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")

            self.f1.set_title(r'$High \ Resolution \ ON$')
            self.f1.set_xlabel(r'$Tiempo [s]$')
            self.f1.set_ylabel(r'$Amplitude [V]$')

            self.f1 = self.fig1.add_subplot(122)

            number = len(i_HR_OFF)
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, number)]

            print "OFF"
            print "++++"
            print len(i_HR_OFF),len(q_HR_OFF)
            print "++++"

            for j in range(len(i_HR_OFF)):

                if create_Tree == False:
                    item = self.ui.treeIQ.invisibleRootItem()
                    signal_count = item.childCount()
                    child = item.child(j + len(i_HR_ON))
                    if child.isSelected():
                        self.f1.plot(psd[3], i_HR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                        self.f1.plot(psd[3], q_HR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                else:

                    item = QTreeWidgetItem(tree)
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)
                    item.setText(0, "Signal HR OFF:" + str(j))  
                    item.setBackground(0, QtGui.QColor(int(255*colors[j][0]),int(255*colors[j][1]),int(255*colors[j][2]),int(255*colors[j][3])))

                    self.f1.plot(psd[3], i_HR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                    self.f1.plot(psd[3], q_HR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
            
            self.f1.set_title(r'$High \ Resolution \ OFF$')
            self.f1.set_xlabel(r'$Tiempo [s]$')
            self.f1.set_ylabel(r'$Amplitude [V]$')

            num += 1

        self.f1.figure.canvas.draw()

    # Plot IQ time stream Low Resolution  
    def plotIQ_LR_TS(self):

        paths, shortName, namePlot, headPath = self.kidSelected()

        create_Tree = False

        if self.flagResize == True:
            self.resizeTree(self.flagResize)
            create_Tree = True

        try:
            self.fig1.clf()
        except:
            pass

        num = 0
        tree = self.ui.treeIQ
        tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        if len(paths) == 0:
            self.messageBox("Error","Select at least one KID to plot")    
            self.ui.setEnabled(True)
            return

        try:
            self.fig1.clf()
        except:
            pass

        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                self.ui.setEnabled(True)
                return

            # I
            i_LR_ON = psd_low[10]
            i_LR_OFF = psd_low_OFF[10]

            # Q
            q_LR_ON = psd_low[11]
            q_LR_OFF = psd_low_OFF[11]

            self.f1 = self.fig1.add_subplot(121)

            number = len(i_LR_ON)
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, number)]

            print "0N"
            print "++++"
            print len(i_LR_ON),len(q_LR_ON)
            print "++++"

            for i in range(len(i_LR_ON)):

                if create_Tree == False:
                    item = self.ui.treeIQ.invisibleRootItem()
                    signal_count = item.childCount()
                    child = item.child(i)
                    if child.isSelected():
                        self.f1.plot(psd_low[3], i_LR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                        self.f1.plot(psd_low[3], q_LR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                else:

                    item = QTreeWidgetItem(tree)
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)
                    item.setText(0, "Signal LR ON:" + str(i))  
                    item.setBackground(0, QtGui.QColor(int(255*colors[i][0]),int(255*colors[i][1]),int(255*colors[i][2]),int(255*colors[i][3])))
                    
                    self.f1.plot(psd_low[3], i_LR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")
                    self.f1.plot(psd_low[3], q_LR_ON[i], color=colors[i], label=r"$"+"Plot "+str(i)+"$")

            self.f1.set_title(r'$Low \ Resolution \ ON$')
            self.f1.set_xlabel(r'$Tiempo [s]$')
            self.f1.set_ylabel(r'$Amplitude [V]$')

            self.f1 = self.fig1.add_subplot(122)

            print "OFF"
            print "++++"
            print len(i_LR_OFF),len(q_LR_OFF)
            print "++++"

            number = len(i_LR_OFF)
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, number)]

            for j in range(len(i_LR_OFF)):

                if create_Tree == False:
                    item = self.ui.treeIQ.invisibleRootItem()
                    signal_count = item.childCount()
                    child = item.child(j+len(i_LR_ON))
                    if child.isSelected():
                        self.f1.plot(psd_low[3], i_LR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                        self.f1.plot(psd_low[3], q_LR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                else:

                    item = QTreeWidgetItem(tree)
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)
                    item.setText(0, "Signal LR OFF:" + str(j))  
                    item.setBackground(0, QtGui.QColor(int(255*colors[j][0]),int(255*colors[j][1]),int(255*colors[j][2]),int(255*colors[j][3])))

                    self.f1.plot(psd_low[3], i_LR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
                    self.f1.plot(psd_low[3], q_LR_OFF[j], color=colors[j], label=r"$"+"Plot "+str(j)+"$")
            
            self.f1.set_title(r'$Low \ Resolution \ OFF$')
            self.f1.set_xlabel(r'$Tiempo [s]$')
            self.f1.set_ylabel(r'$Amplitude [V]$')

            num += 1

        self.f1.figure.canvas.draw()

    def IQ_HR(self):
        self.now = "HR"
        if self.last != self.now:
            if self.flagResize == False:
                self.resizeTree(self.flagResize)
        self.last = self.now

        self.plotIQ_HR_TS()

    def IQ_LR(self):
        self.now = "LR"
        if self.last != self.now:
            if self.flagResize == False:
                self.resizeTree(self.flagResize)
        self.last = self.now

        self.plotIQ_LR_TS()

    # --- Plot Sweep, speed and IQ Circle in one figure
    def plotVNA(self,cnt,exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, namePlot,nPlots,leg, f0_leg, fit):
        c1, c2, alpha, p = plotPar

        #---S21---
        self.f1 = self.fig1.add_subplot(311)
        mag_vna = np.sqrt(sweep_i**2+sweep_q**2)
        mag_vna = 20*np.log10(mag_vna) + 10

        z = self.f1.plot(freq,mag_vna,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        c2 = z[0].get_color()

        if self.ui.actionHR.isChecked():
            mag_hr = np.sqrt((sweep_i_hr**2)+(sweep_q_hr**2))
            mag_hr = 20*np.log10(mag_hr) + 10
            self.f1.plot(freq_hr,mag_hr,color=c2,dashes=[12, 2],marker="o")

        if self.ui.actionQ_Factor.isChecked():
            fit_curve_par = fit[0:3]
            fit_curve = -1*self.getQ.lorentz(freq,*fit_curve_par) + np.max(mag_vna)
            self.f1.plot(freq, fit_curve,color=c2,dashes=[6, 2])
            self.f1.plot(fit[2],np.min(fit_curve),"yo")
            self.f1.annotate(r"$"+str(fit[2]/1e6)+"MHz, Q="+str(fit[4])+"$",xy=(fit[2],np.min(fit_curve)))

        if self.ui.actionFindResonance.isChecked() or f0_leg == True:
            self.f1.plot([f0_meas, f0_meas],[np.min(mag_vna),np.max(mag_vna)],'c-')
            self.f1.plot([f0_fits, f0_fits],[np.min(mag_vna),np.max(mag_vna)],'g--')
            self.f1.plot(f0_meas,np.min(mag_vna),'ro')
            self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(mag_vna)))
            self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(mag_vna)))

        self.f1.set_xlabel(r'$Frequency [Hz]$')
        self.f1.set_ylabel(r'$S_{21}[dBm]$')

        if leg == False:
            xp = freq[cnt*(len(freq)/2)/nPlots + 1]
            yp = mag_vna[cnt*(len(mag_vna)/2)/nPlots + 1]
            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #---Speed---
        self.f1 = self.fig1.add_subplot(312)

        di_df = np.diff(sweep_i)/np.diff(freq)
        dq_df = np.diff(sweep_q)/np.diff(freq)
        speed = np.sqrt(di_df**2 + dq_df**2)

        z = self.f1.plot(freq[:-1],speed,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        c3 = z[0].get_color()

        # Speed High Resolution
        """
        if self.ui.actionHR.isChecked():
            di_df_hr = np.diff(sweep_i_hr)/np.diff(freq_hr)
            dq_df_hr = np.diff(sweep_q_hr)/np.diff(freq_hr)
            speed_hr = np.sqrt(di_df_hr**2 + dq_df_hr**2)
            self.f1.plot(freq_hr[:-1],speed_hr,color=c3,dashes=[12, 2],marker="o")"""

        if self.ui.actionFindResonance.isChecked() or f0_leg == True:
            self.f1.plot([f0_meas, f0_meas],[np.min(speed),np.max(speed)],'c-')
            self.f1.plot([f0_fits, f0_fits],[np.min(speed),np.max(speed)],'g--')
            self.f1.plot(f0_meas,np.min(speed),'ro')
            self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(speed)))
            self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(speed)))

        self.f1.set_xlabel(r'$Frequency [Hz]$')
        self.f1.set_ylabel(r'$Speed[V/Hz]$')

        self.getStad(freq,fit)
        
        if leg == False:    
            xp = freq[cnt*(len(freq)/2)/nPlots + 1]
            yp = speed[cnt*(len(speed)/2)/nPlots + 1]
        
            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #---IQ---
        self.f1 = self.fig1.add_subplot(313)

        z = self.f1.plot(sweep_i,sweep_q,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        c3 = z[0].get_color()

        if self.ui.actionHR.isChecked():
            self.f1.plot(sweep_i_hr,sweep_q_hr,color=c3,dashes=[12, 2],marker="o")

        self.f1.axis('equal')
        self.f1.set_xlabel(r'$I[V]$')
        self.f1.set_ylabel(r'$Q[V]$')

        if leg == False:
            xp = sweep_i[cnt*(len(sweep_i)/2)/nPlots + 1]
            yp = sweep_q[cnt*(len(sweep_q)/2)/nPlots + 1]

            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #Plot parameters
        #Legend
        if self.ui.squareLeg.isChecked() or leg == True:
            self.f1.legend(loc='best')

        if self.flagResize == False:
            self.resizeTree(self.flagResize)

    # --- Plot time stream high resolution for I, Q and df
    def plotTimeStream(self,cnt,exists, freq, sweep_i, sweep_q, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, namePlot,nPlots, leg):
        c1, c2, alpha, p = plotPar

        #---I time---
        self.f1 = self.fig1.add_subplot(311)
        self.f1.plot(psd[3],psd[4],c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        self.f1.plot(psd_OFF[3],psd_OFF[4],c2+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        self.f1.set_xlabel(r'$Time [s]$')
        self.f1.set_ylabel(r'$I[V]$')

        if leg == False:
            xp = psd[3][cnt*(len(psd[0])/2)/nPlots + 1]
            yp = psd[4][cnt*(len(psd[4])/2)/nPlots + 1]
            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #---Q time---
        self.f1 = self.fig1.add_subplot(312)
        self.f1.plot(psd[3],psd[5],c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        self.f1.plot(psd_OFF[3],psd_OFF[5],c2+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        self.f1.axis('equal')
        self.f1.set_xlabel(r'$Time[s]$')
        self.f1.set_ylabel(r'$Q[V]$')

        if leg == False:
            xp = psd[3][cnt*(len(psd[3])/2)/nPlots + 1]
            yp = psd[5][cnt*(len(psd[5])/2)/nPlots + 1]
            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #---df---
        self.f1 = self.fig1.add_subplot(313)
        
        self.f1.plot(psd[3],psd[6],c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        self.f1.plot(psd_OFF[3],psd_OFF[6],c2+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        self.f1.set_xlabel(r'$Time [s]$')
        self.f1.set_ylabel(r'$df[V]$')
       
        if leg == False:
            xp = psd[3][cnt*(len(psd[3])/2)/nPlots + 1]
            yp = psd[6][cnt*(len(psd[6])/2)/nPlots + 1]
            if self.ui.plotLeg.isChecked():
                self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

        #Plot parameters
        #Legend
        if self.ui.squareLeg.isChecked() or leg == True:
            self.f1.legend(loc='best')

    # --- Plot Noise for ON/OFF resonance frequency
    def plotNoise(self, psd, psd_OFF, psd_low, psd_low_OFF, namePlot, cnt, plotPar, ctrl, clr):
        c1, c2, alpha, alpha_on, p = plotPar

        self.f1 = self.fig1.add_subplot(111)

        ind_cut_freq = np.where(psd[0]>1000.)[0][0]
        psdON = np.concatenate((psd_low[2][1:],psd[2][ind_cut_freq:-1]),axis=0)
        psdOFF = np.concatenate((psd_low_OFF[2][1:],psd_OFF[2][ind_cut_freq:-1]),axis=0)
        
        psdFreqON = np.concatenate(( psd_low[0][1:], psd[0][ind_cut_freq:-1]),axis=0)
        psdFreqOFF = np.concatenate(( psd_low_OFF[0][1:], psd_OFF[0][ind_cut_freq:-1]),axis=0) 
       
        y = self.f1.semilogx(psdFreqON, psdON, c1+p+'-', alpha=alpha, label=r"$"+namePlot[cnt]+"$")
        
        if ctrl == False:
            if self.ui.manyColors.isChecked():
                c2 = y[0].get_color()
        else:
            if clr == True:
                c2 = y[0].get_color()
        
        self.f1.semilogx(psdFreqOFF,psdOFF, color=c2,marker=p, alpha=alpha_on)

        self.f1.set_title(r"$Noise \ low \ and \ high \ resolution$")
        self.f1.set_xlabel(r'$frequency [Hz]$')
        self.f1.set_ylabel(r'$df[dB]$')
        
        xp = psd_low_OFF[0][cnt + 1]
        yp = psd_low_OFF[2][cnt + 1]

        return xp,yp

    # --- Plot sweep
    def plotSweep(self, freq, sweep_i, sweep_q, namePlot, cnt, plotPar, f0_leg, f0_meas, f0_fits, nPlots, fit):
        c1, c2, alpha, p = plotPar

        self.f1 = self.fig1.add_subplot(111)

        mag = np.sqrt(sweep_i**2+sweep_q**2)
        mag = 20*np.log10(mag) + 10

        z = self.f1.plot(freq,mag,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        if self.ui.actionQ_Factor.isChecked():
            fit_curve_par = fit[0:3]
            fit_curve = -1*self.getQ.lorentz(freq,*fit_curve_par) + np.max(mag)
            c2 = z[0].get_color()
            self.f1.plot(freq, fit_curve,color=c2,dashes=[6, 2])
            self.f1.plot(fit[2],np.min(fit_curve),"yo")
            self.f1.annotate(r"$"+str(fit[2]/1e6)+"MHz, Q="+str(fit[4])+"$",xy=(fit[2],np.min(fit_curve)))

        if self.ui.actionFindResonance.isChecked() or f0_leg == True:
            self.f1.plot([f0_meas, f0_meas],[np.min(mag),np.max(mag)],'c-')
            self.f1.plot([f0_fits, f0_fits],[np.min(mag),np.max(mag)],'g--')
            self.f1.plot(f0_meas,np.min(mag),'ro')
            self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(mag)))
            self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(mag)))

        self.f1.set_xlabel(r'$Frequency [Hz]$')
        self.f1.set_ylabel(r'$S_{21}[dBm]$')

        self.getStad(freq,fit)
          
        xp = freq[cnt*(len(freq)/2)/nPlots + 1]
        yp = mag[cnt*(len(mag)/2)/nPlots + 1]

        return xp, yp

    # --- Plot IQ Circles
    def plotIQ(self, sweep_i, sweep_q, namePlot, cnt, plotPar, nPlots):
        c1, c2, alpha, p = plotPar
        
        self.f1 = self.fig1.add_subplot(111)

        self.f1.plot(sweep_i,sweep_q,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        self.f1.axis('equal')
        self.f1.set_xlabel(r'$I[V]$')
        self.f1.set_ylabel(r'$Q[V]$')

        xp = sweep_i[cnt*(len(sweep_i)/2)/nPlots + 1]
        yp = sweep_q[cnt*(len(sweep_q)/2)/nPlots + 1]

        return xp, yp

    # --- To plot Time stream/Noise/Sweep/IQ Circle. The figure adapt to the plots selected
    def getPlot(self, event):
        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage("Loading Files")
        self.enableW.start()
    
    def plotData(self):

        if self.flagResize == False:
            self.resizeTree(self.flagResize)

        c1 = ""
        c2 = ""
        alpha = 1
        p = ""
        if self.ui.pointPlot.isChecked():
            p = "*"

        plotPar = [c1,c2,alpha,p]

        if self.ui.actionVNA_Plots.isChecked() or self.ui.actionHomodynePlots.isChecked():

            paths, shortName, namePlot, headPath = self.kidSelected()

            if len(paths) == 0:
                self.messageBox("Error","Select at least one KID to plot")    
                self.ui.setEnabled(True)
                return

            try:
                self.fig1.clf()
            except:
                pass

            cnt = 0
            for path in paths:
                try:
                    exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path,shortName[cnt], headPath[cnt], "all")
                except Exception as e:
                    self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                    self.ui.setEnabled(True)
                    return

                if self.ui.actionVNA_Plots.isChecked():
                    self.plotVNA(cnt, exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, namePlot, len(paths), False, False, fit)
                else:
                    self.plotTimeStream(cnt, exists, freq, sweep_i, sweep_q, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, namePlot, len(paths), False)

                self.f1.figure.canvas.draw()
                cnt += 1

        else:
            #Graphics flags
            flags = [False]*4

            if self.ui.actionTimeStream.isChecked():
                flags[0] = True
            if self.ui.actionNoise.isChecked():
                flags[1] = True
            if self.ui.actionSweep.isChecked():
                flags[2] = True
            if self.ui.actionIQCircle.isChecked():
                flags[3] = True

            #Adjust the plots
            nCheck = 0
            for i in flags:
                if i == True:
                    nCheck += 1

            if nCheck > 0:

                paths, shortName, namePlot, headPath = self.kidSelected()

                if len(paths) == 0:
                    self.messageBox("Error","Select at least one KID to plot")    
                    self.ui.setEnabled(True)
                    return

                try:
                    self.fig1.clf()
                except:
                    pass

                gs = gridspec.GridSpec(2, 2)
                cnt = 0

                for path in paths:
                    try:
                        if flags[0] == True or flags[1] == True:
                            if flags[2] == True or flags[3] == True:
                                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path,shortName[cnt], headPath[cnt], "all")
                            else:
                                allData = self.loadData(path,shortName[cnt], headPath[cnt], "homo")
                                if allData[0] == True:
                                    psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[4], allData[5], allData[6], allData[7]
                                else:
                                    psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = allData[7], allData[8], allData[9], allData[10], allData[11], allData[12], allData[13]
                        else:
                            if self.ui.actionIQcircleTS.isChecked():
                                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path,shortName[cnt], headPath[cnt], "all")   
                            else:
                                allData = self.loadData(path, shortName[cnt], headPath[cnt], "vna")
                                if allData[0] == True:
                                    freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[4], allData[5], allData[6], allData[7], allData[8], allData[9]
                                else:
                                    freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[4], allData[5], allData[6], allData[11], allData[12], allData[13]

                    except Exception as e:
                        self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                        self.ui.setEnabled(True)
                        return

                    #Banderas habilitadoras de gráficos
                    checked_0 = True
                    checked_1 = True
                    checked_2 = True
                    checked_3 = True

                    auxSub = 0

                    if self.ui.oneColor.isChecked():
                        c1 = "r"
                        c2 = "b"
                        alpha = 0.75-0.5*cnt/len(paths)
                        alpha_on = alpha
                    else:
                        alpha_on = 0.75

                    for nPlot in range(nCheck):
                        xp = 0
                        yp = 0

                        if nCheck%2 == 0: 
                            self.f1 = self.fig1.add_subplot(int(nCheck/2),2,nPlot + 1)
                        else:
                            if flags[1] == True and nCheck == 3:
                                if auxSub == 0:
                                    self.f1 = self.fig1.add_subplot(gs[0, :])
                                elif auxSub == 1:
                                    self.f1 = self.fig1.add_subplot(gs[-1, 0])
                                else:
                                    self.f1 = self.fig1.add_subplot(gs[-1, -1])
                                auxSub += 1
                            else:
                                self.f1 = self.fig1.add_subplot(1,nCheck,nPlot + 1)

                        if flags[0] == True and checked_0 == True:
                            
                            mag_OFF = np.sqrt(psd_low_OFF[4]**2 + psd_low_OFF[5]**2)
                            mag = np.sqrt(psd_low[4]**2 + psd_low[5]**2)

                            self.f1.plot(psd_low_OFF[3],mag_OFF,c2+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+",OFF$")
                            self.f1.plot(psd_low[3],mag,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+",ON$")

                            self.f1.set_title(r"$Time Stream$")
                            self.f1.set_xlabel(r'$Time [s]$')
                            self.f1.set_ylabel(r'$\sqrt{I^2+Q^2} [V]$')

                            xp = mag[cnt*(len(psd[0])/2)/len(paths) + 1]
                            yp = mag_OFF[cnt*(len(psd[0])/2)/len(paths) + 1]
                            checked_0 = False

                        elif flags[1] == True and checked_1 == True:

                            ind_cut_freq = np.where(psd[0]>1000.)[0][0]
                            psdON = np.concatenate((psd_low[2][1:],psd[2][ind_cut_freq:-1]),axis=0)
                            psdOFF = np.concatenate((psd_low_OFF[2][1:],psd_OFF[2][ind_cut_freq:-1]),axis=0)
                            
                            psdFreqON = np.concatenate(( psd_low[0][1:], psd[0][ind_cut_freq:-1]),axis=0)
                            psdFreqOFF = np.concatenate(( psd_low_OFF[0][1:], psd_OFF[0][ind_cut_freq:-1]),axis=0) 
                           
                            y = self.f1.semilogx(psdFreqON, psdON, c1+p+'-', alpha=alpha, label=r"$"+namePlot[cnt]+"$")
                            
                            if self.ui.manyColors.isChecked():
                                c2 = y[0].get_color()
                            
                            self.f1.semilogx(psdFreqOFF,psdOFF, color=c2,marker=p, alpha=alpha_on)
                    
                            #self.f1.legend(loc='best')
                            self.f1.set_title(r"$Noise \ low \ and \ high \ resolution$")
                            self.f1.set_xlabel(r'$frequency [Hz]$')
                            self.f1.set_ylabel(r'$PSD \ df[dB]$')
                            
                            xp = psd_low_OFF[0][cnt + 1]
                            yp = psd_low_OFF[2][cnt + 1]
                            checked_1 = False
                        
                        elif flags[2] == True and checked_2 == True:
                            mag = np.sqrt((sweep_i**2)+(sweep_q**2))
                            self.f1.set_ylabel(r'$S_{21}[V]$')

                            if self.ui.ylog.isChecked() or self.ui.actionQ_Factor.isChecked():
                                self.ui.ylog.setChecked(True)
                                mag = 20*np.log10(mag) + 10
                                self.f1.set_ylabel(r'$S_{21}[dBm]$')                           

                            z = self.f1.plot(freq,mag, c1+p+'-', alpha=alpha, label=r"$"+namePlot[cnt]+"$")
                            c3 = z[0].get_color()

                            if self.ui.actionHR.isChecked():
                                mag_hr = np.sqrt((sweep_i_hr**2)+(sweep_q_hr**2))
                                if self.ui.ylog.isChecked():
                                    mag_hr = 20*np.log10(mag_hr) + 10
                                self.f1.plot(freq_hr,mag_hr,color=c3,dashes=[12, 2],marker="o")
                            
                            if self.ui.actionQ_Factor.isChecked():
                                fit_curve_par = fit[0:3]
                                fit_curve = -1*self.getQ.lorentz(freq,*fit_curve_par) + np.max(mag)
                                self.f1.plot(freq, fit_curve,color=c3,dashes=[6, 2])
                                self.f1.plot(fit[2],np.min(fit_curve),"yo")
                                self.f1.annotate(r"$"+str(fit[2]/1e6)+"MHz, Q="+str(fit[4])+"$",xy=(fit[2],np.min(fit_curve)))

                            if self.ui.actionFindResonance.isChecked():
                                self.f1.plot([f0_meas, f0_meas],[np.min(mag),np.max(mag)],'c-')
                                self.f1.plot([f0_fits, f0_fits],[np.min(mag),np.max(mag)],'g--')
                                self.f1.plot(f0_meas,np.min(mag),'ro')
                                self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(mag))) 
                                self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(mag))) 

                            self.f1.set_title(r"$Sweep$")
                            self.f1.set_xlabel(r'$frequency [Hz]$')

                            self.getStad(freq, fit)

                            xp = freq[cnt*(len(mag)/2)/len(paths) + 1]
                            yp = mag[cnt*(len(mag)/2)/len(paths) + 1]
                            checked_2 = False

                        elif flags[3] == True and checked_3 == True:

                            z = self.f1.plot(sweep_i,sweep_q,c1+p+'-', alpha=alpha, label=r"$"+namePlot[cnt]+"$")
                            c3 = z[0].get_color()

                            if self.ui.actionHR.isChecked():
                                self.f1.plot(sweep_i_hr,sweep_q_hr,color=c3,dashes=[12, 2],marker="o")

                            # High and low resolution
                            if self.ui.actionIQcircleTS.isChecked():
                                # I
                                i_LR_ON = psd_low[10]
                                # Q
                                q_LR_ON = psd_low[11]

                                for i in range(len(i_LR_ON)):
                                    self.f1.plot(i_LR_ON[i], q_LR_ON[i],"ro")

                            self.f1.axis('equal')
                            self.f1.set_title(r"$IQ \ circle$")
                            self.f1.set_xlabel(r'$I$')
                            self.f1.set_ylabel(r'$Q$')

                            xp = sweep_i[cnt*(len(sweep_i)/2)/len(paths) + 1]
                            yp = sweep_q[cnt*(len(sweep_q)/2)/len(paths) + 1]
                            checked_3 = False  

                        #Plot parameters
                        #Legend
                        if self.ui.squareLeg.isChecked():
                            self.f1.legend(loc='best')
                        if self.ui.plotLeg.isChecked():
                            self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))

                    cnt += 1
                    #self.fig1.tight_layout()
                    self.f1.figure.canvas.draw()
                
                self.ui.statusbar.showMessage("KID data has been loaded")
            else:
                self.messageBox("Error","Select a graph to plot")
        self.ui.setEnabled(True)            

    #*********MENU*********
    # --- Open Directory
    def openApp(self,event):
        w = QWidget()
        w.resize(320, 240)    
        w.setWindowTitle("Select directory where KID files are ")

        self.path.append(QFileDialog.getExistingDirectory(self, "Select Directory"))

        diry = ""
        for i in self.path:
            diry = diry + i + "...  "
        self.ui.currentDiry.setText(diry)

        KIDS, self.allPaths, self.allNames, self.allHeads, self.allShortNames = self.findFiles(self.path)

        if KIDS == []:
            self.path.pop()
            self.messageBox("Warning","KID files not found!")
            return -1
        else:
            self.KIDS = KIDS
            tree = self.ui.treeKID

            for i in xrange(len(KIDS)):
                parent = QTreeWidgetItem(tree)
                parent.setText(0, KIDS[i][0])
                parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                
                for j in xrange(len(KIDS[i][1])):
                    child = QTreeWidgetItem(parent)
                    child.setIcon(0, QIcon('./resources/f0_tem.png'))
                    child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                    child.setText(0, KIDS[i][1][j][0][16:19] + KIDS[i][1][j][0][20:22])
                    
                    for k in xrange(len(KIDS[i][1][j][1])):
                        grandChild = QTreeWidgetItem(child)
                        grandChild.setIcon(0, QIcon('./resources/power.png'))
                        grandChild.setFlags(grandChild.flags() | Qt.ItemIsUserCheckable)
                        grandChild.setText(0, KIDS[i][1][j][1][k][16:20])
                        grandChild.setCheckState(0, Qt.Unchecked)  

    # --- Exit
    def closeApp(self):
        try:
            self.map.close()
        except:
            pass
        self.ui.close()

    # --- Get the files selected in the tree
    def kidSelected(self):
        checked = dict()
        root = self.ui.treeKID.invisibleRootItem()
        signal_count = root.childCount()

        paths = []
        shortName = []
        namePlot = []
        headPath = []

        for i in range(signal_count):
            signal = root.child(i)
            checked_sweeps = list()
            num_children = signal.childCount()

            for n in range(num_children):
                child = signal.child(n)
                num_grandChildren = child.childCount()

                for m in range(num_grandChildren):
                    grandChild = child.child(m)

                    if grandChild.checkState(0) == QtCore.Qt.Checked:
                        checked_sweeps.append(child.text(0))
 
                        # I don't like this block of code, but it works so while                            
                        if len(child.text(0)) > 4:
                            sub = "mK"
                        else:
                            sub = "K"

                        test = ""
                        s = 0
                        for j in signal.text(0):
                            if j == '_':
                                break
                            else:
                                test = test + j
                            s += 1

                        nKIDs = int(test[1:])

                        headPath.append(str(self.path[nKIDs]))
                        paths.append(str(self.path[nKIDs]) + '/' + str(signal.text(0))[s+1:] + "/Set_Temperature_" + str(child.text(0)[0:3]) + "_" + sub + "/Set_Attenuation_" + str(grandChild.text(0)) + '/')
                        shortName.append(str(signal.text(0))[-4:] + "_" + str(child.text(0)[:-2]) + "_" + str(grandChild.text(0)))
                        namePlot.append(test + ", " + str(signal.text(0))[-4:] + ", T = " + str(child.text(0)[0:3]) + " " + sub + ", A = " + str(grandChild.text(0)))

        return paths, shortName, namePlot, headPath

    def clearKidTree(self,event):
        self.clearTree(self.ui.treeKID, True)

    # --- Clear KID tree. Adapt it from: tcrownson https://gist.github.com/tcrowson/8273931
    def clearTree(self, tree, full):

        if full == True:
            self.ui.currentDiry.setText(" ")
            self.path = []
            self.nKIDS = 0

        iterator = QtGui.QTreeWidgetItemIterator(tree, QtGui.QTreeWidgetItemIterator.All)
        while iterator.value():
            iterator.value().takeChildren()
            iterator +=1
        
        i = tree.topLevelItemCount()
        while i > -1:
            tree.takeTopLevelItem(i)
            i -= 1

    # --- Get path from each KID file
    def findFiles(self, path):

        self.ui.statusbar.showMessage(u'Searching KIDs Files')

        m = 0
        KIDS = []
        allPaths = []
        allNames = []
        allHeads = []
        allShortNames = []

        try:
            kidFiles = os.listdir(path[-1])
            for kid in kidFiles:
                if kid[0:4] == "KID_":
                    KIDS.append(["T" + str(self.nKIDS) + "_" + kid])
                    temFiles = os.listdir(path[-1] + '/' + kid)
                    auxTemp = []
                    n = 0
                    
                    for tem in temFiles:
                        if tem[0:16] == "Set_Temperature_":
                            auxTemp.append([tem])
                            atteFiles = os.listdir(path[-1] + '/' + kid + '/' + tem)
                            auxAtte = []
                            l = 0

                            for att in atteFiles:
                                if att[0:16] == "Set_Attenuation_":
                                    auxAtte.append(att)
                                    allPaths.append(path[-1] + '/' + kid + '/' + tem + '/' + att)
                                    allNames.append(kid[4:8] + '_' + tem[16:19] + '_' + att[16:20])
                                    allShortNames.append(kid[4:8] + ', T = ' + tem[16:19] + ', A = ' + att[16:20])
                                    allHeads.append(path[-1])
                                    l = l + 1
                            auxTemp[n].append(auxAtte)
                            n = n + 1
                    KIDS[m].append(auxTemp)
                    m = m + 1

            if KIDS != []:
                self.nKIDS += 1 

        except:
            pass

        return KIDS, allPaths, allNames, allHeads, allShortNames

    # --- Functions to load all files in one or some directories
    def loadThread(self):
        if len(self.allPaths) == 0:
            self.messageBox("Warning","Directory not selected")
            self.ui.setEnabled(True)
            return
        try:
            i = 0
            for path in self.allPaths:
                self.loadData(str(path), str(self.allNames[i]), str(self.allHeads[i]), "all", False)
                i += 1
        except:
            self.messageBox("Warning","Error trying to open files, maybe files are missing")
            pass      

        self.ui.setEnabled(True) 
        self.ui.statusbar.showMessage("KID data has been loaded") 

    def loadFullData(self, event):
        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage("Loading Files")

        #Este hilo llama al ploteador
        self.threadLoad.start()        

    def loadData(self, path, name, headPath, inst, load=True):
        exists = False
        name = str(name)

        fullPath = str(headPath) + '/' + name

        if os.path.exists(fullPath):
            exists = True
            if load == False:
                return

            if inst == "vna":
                freq = np.load(fullPath + "/" + name + "_freq.npy")
                sweep_i = np.load(fullPath + "/" + name + "_sweep_i.npy")
                sweep_q = np.load(fullPath + "/" + name + "_sweep_q.npy")
                
                freq_hr = np.load(fullPath + "/" + name + "_freq_hr.npy")
                sweep_i_hr = np.load(fullPath + "/" + name + "_sweep_i_hr.npy")
                sweep_q_hr = np.load(fullPath + "/" + name + "_sweep_q_hr.npy")

                f0_meas,f0_fits = np.load(fullPath + "/" + name + "_stat.npy")
                fit = np.load(fullPath + "/" + name + "_qfactor.npy")

                status = fullPath + " Loaded \n**********"
                self.ui.statusText.append(status)

                return exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, f0_meas, f0_fits, fit

            elif inst == "homo":
                psd = np.load(fullPath + "/" + name + "_psd.npy")            
                psd_low = np.load(fullPath + "/" + name + "_psd_low.npy")
                psd_OFF = np.load(fullPath + "/" + name + "_psd_OFF.npy")
                psd_low_OFF = np.load(fullPath + "/" + name + "_psd_low_OFF.npy")
                f0_meas,f0_fits = np.load(fullPath + "/" + name + "_stat.npy")
                fit = np.load(fullPath + "/" + name + "_qfactor.npy")

                status = fullPath + " Loaded \n**********"
                self.ui.statusText.append(status)

                return exists, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas,f0_fits,fit
            
            elif inst == "all":
                freq = np.load(fullPath + "/" + name + "_freq.npy")
                sweep_i = np.load(fullPath + "/" + name + "_sweep_i.npy")
                sweep_q = np.load(fullPath + "/" + name + "_sweep_q.npy")

                freq_hr = np.load(fullPath + "/" + name + "_freq_hr.npy")
                sweep_i_hr = np.load(fullPath + "/" + name + "_sweep_i_hr.npy")
                sweep_q_hr = np.load(fullPath + "/" + name + "_sweep_q_hr.npy")

                psd = np.load(fullPath + "/" + name + "_psd.npy")            
                psd_low = np.load(fullPath + "/" + name + "_psd_low.npy")
                psd_OFF = np.load(fullPath + "/" + name + "_psd_OFF.npy")
                psd_low_OFF = np.load(fullPath + "/" + name + "_psd_low_OFF.npy")   
                f0_meas,f0_fits = np.load(fullPath + "/" + name + "_stat.npy")
                fit = np.load(fullPath + "/" + name + "_qfactor.npy")

                status = fullPath + " Loaded \n**********"
                self.ui.statusText.append(status)

                return exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit

        else:

            os.system('mkdir ' + fullPath)

            freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr,  psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits = self.dataRedtn.get_all_data(path,self.ui.actionCosRay.isChecked()) 

            mag = np.sqrt((sweep_i**2)+(sweep_q**2))
            mag = 20*np.log10(mag) + 10

            fit = self.getQFactor(freq,mag)

            # Low resolution
            np.save(fullPath + "/" + name + "_freq", freq)
            np.save(fullPath + "/" + name + "_sweep_i", sweep_i)
            np.save(fullPath + "/" + name + "_sweep_q", sweep_q)
            # High resolution
            np.save(fullPath + "/" + name + "_freq_hr", freq_hr)
            np.save(fullPath + "/" + name + "_sweep_i_hr", sweep_i_hr)
            np.save(fullPath + "/" + name + "_sweep_q_hr", sweep_q_hr)
            
            np.save(fullPath + "/" + name + "_psd", psd)
            np.save(fullPath + "/" + name + "_psd_low", psd_low)
            np.save(fullPath + "/" + name + "_psd_OFF", psd_OFF)
            np.save(fullPath + "/" + name + "_psd_low_OFF", psd_low_OFF)
            np.save(fullPath + "/" + name + "_stat",[f0_meas, f0_fits])
            np.save(fullPath + "/" + name + "_qfactor", fit)

            status = str(psd[7]) + ' Temp=' + str(psd[8]) + ' att= ' + str(psd[9]) + "\n**********"
            self.ui.statusText.append(status)

            return exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit

    # --- Functions of MAP (Moving Approximation Transform) based on the work of PhD Ildar Batyrshin 
    def loadMAP(self, event):
        self.map = uic.loadUi("gui/map.ui")

        self.minM = 5
        self.maxM = 50
        self.threshM = 0.85

        self.map.minMAP.setText(str(5))
        self.map.maxMAP.setText(str(50))
        self.map.threshMAP.setText(str(0.85))

        self.map.startMAP.mousePressEvent = self.MAP_function

        self.map.move(self.size_x/2-260/2,self.size_y/2-205/2)
        self.map.show()

    def MAP_function(self,event):
        #Cargamos el MAP
        self.MAP = movAproTrans()

        #Read of the files checked
        paths, shortName, namePlot, headPath = self.kidSelected()

        if len(paths) > 1:

            #Cluster
            r = 10.
            x = []
            y = []
            plt.figure('Cluster')
            for i in range(len(paths)):
                ang = i*2*(np.pi)/len(paths)
                x_aux = r*np.cos(ang)
                y_aux = r*np.sin(ang)
                x.append(x_aux)
                y.append(y_aux)

                plt.plot(x_aux,y_aux,'r*')
                plt.annotate(r"$"+namePlot[i]+"$",xy=(x_aux,y_aux))

            thresh = float(self.map.threshMAP.text())
            mini = int(self.map.minMAP.text())
            maxi = int(self.map.maxMAP.text()) 
            
            for cnt1 in range(len(paths) - 1):
                toCom = paths[cnt1]
                allDataCom = self.loadData(toCom,shortName[cnt1],headPath[cnt1],"vna")
                freq_C, sweep_i_C, sweep_q_C = allDataCom[1], allDataCom[2], allDataCom[3]

                for cnt2 in range(1 + cnt1,len(paths)):
                    vs = paths[cnt2]
                    allDataVS = self.loadData(vs,shortName[cnt2],headPath[cnt1],"vna")   
                    freq_V, sweep_i_V, sweep_q_V = allDataVS[1], allDataVS[2], allDataVS[3]

                    f = ((freq_C[1] - freq_C[0])+(freq_C[3] - freq_C[2]))/2
                    
                    mag_C = np.sqrt(sweep_i_C**2+sweep_q_C**2)
                    mag_VS = np.sqrt(sweep_i_V**2+sweep_q_V**2)

                    AFv = self.MAP.AF(mag_C,mag_VS,mini,maxi,f)
                    AMv = self.MAP.AM(AFv,'m')

                    if AMv > thresh:
                        plt.plot([x[cnt1],x[cnt2]],[y[cnt1],y[cnt2]],'c',linewidth=1.5)
                    
                    print toCom[-5:] + vs[-5:] + ": " + str(AMv)
            
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

        else:
            self.messageBox("Info","To apply MAP algorithm, two data signal are needed")
            return

    # --- Generate all the figures and save them
    def saveFigures(self, event):
        print "Create the Directories"
        
        self.ui.setEnabled(False)
        self.messageBox("Info","The program will create the figures for all the files, it would takes several minutes.")

        self.createVNAHomoFigures(self.allPaths, self.KIDS, self.path)
        
        self.ui.setEnabled(True)
        print "Directories created!"

    # --- Functions to create the tree structure with all the measurements
    def createVNAHomoFigures(self,path,diries,heads):

        for head in heads:
            kid = ''
            mainPath = str(head) + "/" + "KIDS_Plots"
            num = 0
            nAtt,nAtt_2,nAtt_3 = 0,0,0
            nTemp = 0
            if os.path.exists(mainPath) == False:
                os.system('mkdir ' + mainPath)

            for i in xrange(len(diries)):
                s_i = 0
                for kidname in diries[i][0]:
                    s_i = s_i + 1
                    if kidname == "_":
                        break

                kid = mainPath + "/" + diries[i][0][s_i:]
                if os.path.exists(kid) == False:
                    os.system('mkdir ' + kid)
             
                tem = ""
                for j in xrange(len(diries[i][1])):

                    temp = kid + "/" + diries[i][1][j][0][16:]                
                    if os.path.exists(temp) == False:
                        os.system('mkdir ' + temp)

                    att = ""
                    for k in xrange(len(diries[i][1][j][1])):

                        att = temp + "/" + diries[i][1][j][1][k][16:20] 
                        if os.path.exists(att) == False:
                            os.system('mkdir ' + att)

                        p = ""
                        c1 = "r"
                        c2 = "b"
                        alpha = 1
                        alpha_on = 1

                        plotPar = [c1,c2,alpha,p]

                        try:
                            self.fig1.clf()
                        except:
                            pass

                        try:
                            exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(str(path[num]),self.allNames[num], self.allHeads[num], "all")                    
                        except Exception as e:
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                            self.ui.setEnabled(True)
                            return

                        self.plotVNA(num, exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, self.allShortNames, 1, True, True, fit)                       
                        self.fig1.savefig(str(att) + '/' + self.allNames[num] + "_VNA.png")

                        self.fig1.clf()

                        self.plotTimeStream(num, exists, freq, sweep_i, sweep_q, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, plotPar, path, self.allShortNames, 1, True)
                        self.fig1.savefig(str(att) + '/' + self.allNames[num] + "_TS.png")

                        self.fig1.clf()

                        self.plotNoise(psd, psd_OFF, psd_low, psd_low_OFF, self.allShortNames, 1, [c1,c2,1,1,p], True, False)
                        self.f1.legend(loc='best')
                        self.fig1.savefig(str(att) + '/' + self.allNames[num] + "_noise.png")

                        self.fig1.clf()

                        self.plotSweep(freq, sweep_i, sweep_q, self.allShortNames, 1, plotPar, True, f0_meas, f0_fits, 1, fit)
                        self.f1.legend(loc='best')
                        self.fig1.savefig(str(att) + '/' + self.allNames[num] + "_sweep.png")                    

                        self.fig1.clf()

                        self.plotIQ(sweep_i, sweep_q, self.allShortNames, 1, plotPar, 1)
                        self.f1.legend(loc='best')
                        self.fig1.savefig(str(att) + '/' + self.allNames[num] + "_iq.png")                   

                        num = num + 1

                    self.fig1.clf()

                    ant = nAtt
                    c1 = ""
                    c2 = ""
                    alpha = 1

                    plotPar = [c1,c2,alpha,p]
                    
                    for eachAtt in xrange(len(diries[i][1][j][1])):
                        try:
                            allData = self.loadData(str(path[nAtt]),self.allNames[nAtt],self.allHeads[nAtt], "vna")
                            if allData[0] == True:
                                freq, sweep_i, sweep_q, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[7], allData[8], allData[9]
                            else:
                                freq, sweep_i, sweep_q, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[11], allData[12], allData[13]
                        except Exception as e:
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                            self.ui.setEnabled(True)
                            return                    

                        self.plotSweep(freq, sweep_i, sweep_q, self.allShortNames[ant:ant + len(diries[i][1][j][1])], eachAtt, plotPar, False, f0_meas, f0_fits, len(diries[i][1][j][1]),fit)
                        self.f1.legend(loc='best')
                        
                        nAtt += 1

                    self.fig1.savefig(str(temp) + '/' + "Sweep_allAtt.png")

                    self.fig1.clf()

                    ant_2 = nAtt_2
                    p = "*"
                    c1 = ""
                    c2 = ""
                    alpha = 1

                    plotPar = [c1,c2,alpha,p]
                    
                    for eachAtt in xrange(len(diries[i][1][j][1])):
                        try:
                            allData = self.loadData(str(path[nAtt_2]),self.allNames[nAtt_2],self.allHeads[nAtt_2], "vna")
                            if allData[0] == True:
                                freq, sweep_i, sweep_q, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[7], allData[8], allData[9]
                            else:
                                freq, sweep_i, sweep_q, f0_meas, f0_fits, fit = allData[1], allData[2], allData[3], allData[11], allData[12], allData[13]
                        except Exception as e:
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                            self.ui.setEnabled(True)
                            return                    

                        self.plotIQ(sweep_i, sweep_q, self.allShortNames[ant_2:ant_2 + len(diries[i][1][j][1])], eachAtt, plotPar, len(diries[i][1][j][1]))
                        self.f1.legend(loc='best')
                        
                        nAtt_2 += 1

                    self.fig1.savefig(str(temp) + '/' + "IQ_allAtt.png")

                    self.fig1.clf()

                    ant_3 = nAtt_3
                    p = ""
                    c1 = ""
                    c2 = ""

                    plotPar = [c1,c2,1,0.75,p]
                    
                    for eachAtt in xrange(len(diries[i][1][j][1])):                    
                        try:
                            exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(str(path[nAtt_3]),self.allNames[nAtt_3],self.allHeads[nAtt_3], "all")
                        except Exception as e:
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e))    
                            self.ui.setEnabled(True)
                            return                    

                        self.plotNoise(psd, psd_OFF, psd_low, psd_low_OFF, self.allShortNames[ant_3:ant_3 + len(diries[i][1][j][1])], eachAtt, plotPar, True, True)
                        self.f1.legend(loc='best')
                        nAtt_3 += 1

                    self.fig1.savefig(str(temp) + '/' + "Noise_allAtt.png")

                    nTemp += 1
                    ant,ant_2,ant_3= 0,0,0

    # --- Get basic information
    def getStad(self, freq, fit):
        fmin = np.min(freq)
        fmax = np.max(freq)
        bw = fmax - fmin
        res = freq[1] - freq[0]

        data = "Fmin = " + str(fmin/1e6) + "MHz\n" + "Fmax = " + str(fmax/1e6) + "MHz\n" + "BW = " + str(bw/1e6) + "MHz\n" + "Res = " + str(res) + "Hz"

        fit = "Q = " + str(fit[4]) + "\nF0 = " + str(fit[2]/1e6) + "MHz\nBW_Q = " + str(fit[3]) + "MHz"

        self.ui.statusText.append("**********")
        self.ui.statusText.append(data)
        self.ui.statusText.append(fit)
        self.ui.statusText.append("**********")


    def getQFactor(self, freq, mag):
        mag = np.abs(np.max(mag) - mag)
        I, G, f0_fit, bw_fit, q_fit = self.getQ.fit(freq,mag)

        return I, G, f0_fit, bw_fit, q_fit

    # --- Message window
    def messageBox(self, title, msg):
        w = QWidget()
        QMessageBox.critical(w, title, msg)

    # --- Functions for graphical settings
    def groupPlotsVNA(self, event):
        if self.ui.actionHomodynePlots.isChecked():
            self.ui.actionHomodynePlots.setChecked(False)

        if self.ui.actionVNA_Plots.isChecked():
            self.ui.actionNoise.setEnabled(False)
            self.ui.actionSweep.setEnabled(False)
            self.ui.actionTimeStream.setEnabled(False)
            self.ui.actionIQCircle.setEnabled(False)
        else:
            self.ui.actionNoise.setEnabled(True)
            self.ui.actionSweep.setEnabled(True)
            self.ui.actionTimeStream.setEnabled(True)
            self.ui.actionIQCircle.setEnabled(True)

    def groupPlotsHomo(self, event):
        if self.ui.actionVNA_Plots.isChecked():
            self.ui.actionVNA_Plots.setChecked(False)

        if self.ui.actionHomodynePlots.isChecked():
            self.ui.actionNoise.setEnabled(False)
            self.ui.actionSweep.setEnabled(False)
            self.ui.actionTimeStream.setEnabled(False)
            self.ui.actionIQCircle.setEnabled(False)
        else:
            self.ui.actionNoise.setEnabled(True)
            self.ui.actionSweep.setEnabled(True)
            self.ui.actionTimeStream.setEnabled(True)
            self.ui.actionIQCircle.setEnabled(True)

    def manyCol(self, event):
        if self.ui.oneColor.isChecked():
            self.ui.oneColor.setChecked(False)

        self.ui.manyColors.setChecked(True)

    def oneCol(self, event):
        if self.ui.manyColors.isChecked():
            self.ui.manyColors.setChecked(False)

        self.ui.oneColor.setChecked(True)

    def addmpl(self,fig):
        self.canvas = FigureCanvas(fig)
        self.ui.MainPlot.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
           self, coordinates=True)
        self.ui.MainPlot.addWidget(self.toolbar)

    def rmmpl(self):
        self.ui.MainPlot.removeWidget(self.canvas)
        self.canvas.close()
        self.ui.MainPlot.removeWidget(self.toolbar)
        self.toolbar.close()

#Ejecución del programa
app = QtGui.QApplication(sys.argv)
MyWindow = MainWindow()
sys.exit(app.exec_())