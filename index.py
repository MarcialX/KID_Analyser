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
from fitting_bounds import fitting_resonators
from enableThread import enableWindow
from loadThread import loadThread

from detector_peaks import detect_peaks
from scipy import signal

from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.stats import gamma

from scipy.misc import factorial
from scipy.optimize import curve_fit

import matplotlib
import scipy.stats
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import(
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


# Main Window
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
        x_plot = 175
        y_plot = 200

        nx = self.size_x - x_plot - 55
        ny = self.size_y - y_plot

        self.ui.plotFrame.resize(int(nx),int(ny))
        self.ui.fileBar.resize(int(nx),41)
        self.ui.currentDiry.resize(int(nx)-45,31)

        self.ui.statusFrame.resize(201,self.size_y - 690)
        self.ui.statusText.resize(191,self.size_y - 700)

        self.ui.plotFrame.setLayout(self.ui.MainPlot)

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

        # Draw line
        self.ui.drawBtn.mousePressEvent = self.drawLine
        # Delete line
        self.ui.eraseBtn.mousePressEvent = self.eraseLine

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
        self.ui.actionPlot_Q.triggered.connect(self.plotQanalysis)

        # Save figures for all the directories files
        self.ui.actionCreateImages.triggered.connect(self.saveFigures)

        # VNA Analysis
        self.ui.addPathVNA.mousePressEvent = self.openVNAFiles
        self.ui.delPathVNA.mousePressEvent = self.clearVNATree

        # Stadistics
        self.ui.actionStadistics.triggered.connect(self.VNAstad)

        # Get Qi and Qc
        self.ui.getQBtn.mousePressEvent = self.getQi_Qc
        self.ui.plotQBtn.mousePressEvent = self.plotQ_shapes
        self.ui.actionNonlinearity.triggered.connect(self.nonlinear_active_bar)
        self.ui.nonlinearBtn.clicked.connect(self.nonlinear_active_btn)

        # Temperature Analysis
        self.ui.actionF0_vs_T.triggered.connect(self.tempPlot)

        # Q Maps
        self.ui.actionMap_array.triggered.connect(self.paintQMap)

        # Filter
        self.ui.actionQ_Filter.triggered.connect(self.filter_analysis)
        self.ui.actionReset.triggered.connect(self.reset_values)

        # Delta functions
        self.ui.actionDelta_Plot.triggered.connect(self.delta_f0)

        # Responsibility
        self.ui.actionResponsivity.triggered.connect(self.respo)

        # Save variables
        self.ui.actionSave_Variables.triggered.connect(self.save_var)

        # Settings
        # ---Q Stadistics
        self.ui.actionQStadistics.triggered.connect(self.Q_StadSet)

        # Help
        self.ui.actionAcerca_de.triggered.connect(self.about)

        # Plot fnotes 
        self.ui.fnoteBtn.clicked.connect(self.fnote_active_btn)

        # Legends
        self.ui.squareLeg.clicked.connect(self.legend_btn)
        self.ui.plotLeg.clicked.connect(self.legend_btn)

        # Write tone list
        self.ui.actionWrite_Tone_List.triggered.connect(self.writeTones)

        # Class to reduce data
        self.dataRedtn = dataRed() 

        # Classes to get Q factor
        self.getQ = get_Q_factor()
        self.fit_S21 = fitting_resonators()

        self.path = []
        self.allPaths = []
        self.allNames = []
        self.allShortNames = []
        self.KIDS = []
        self.nKIDS = 0

        self.flag_ind = False
        self.legend_curve = None

        self.pathVNA = []
        self.vlines = []
        self.acumQ = []

        self.I_vna = []
        self.Q_vna = []
        self.freq_Q = []
        self.ind = []

        self.flagResize = True
        self.flagVNA = False
        self.prevPathVNA = []

        self.lockVNA = False

        self.prevNpCont = []
        self.prevOrderCont = []
        self.prevNp = []
        self.prevOrder = []
        self.prevMPD = []
        self.prevMPH = []

        self.base = []

        self.now = ""
        self.last = ""

        self.nbin = 30
        self.pdfFlag = False
        self.pdaFlag = False
        self.drawCurve = False
        self.writeLeg = False

        self.indexPath = []
        
        self.pdfType = "Gamma"

        self.ui.actionCosRay.setChecked(True)
        
        # Creation of Plot
        self.fig1 = Figure()
        self.f1 = self.fig1.add_subplot(111)
        self.addmpl(self.fig1)
        cid = self.f1.figure.canvas.mpl_connect('button_press_event', self.resDraw)

        # To use LATEX in plots
        try:
            matplotlib.rc('text', usetex=True)
            matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        except:
            pass

        # Thread to show a diferent process while the machine is working
        self.enableW = enableWindow() 
        self.enableW.flagEnable.connect(self.plotData)

        self.threadLoad = loadThread() 
        self.threadLoad.flagLoad.connect(self.loadThread)

        self.ui.show() 

    # --- VNA Analysis
    # ------- Open VNA Sweep
    def openVNAFiles(self,event):

        w = QWidget()
        w.resize(320, 240)    
        w.setWindowTitle("Select VNA files")

        pathVNA = QFileDialog.getOpenFileName(self, "Select VNA files")

        if pathVNA == "":
            self.messageBox("Error VNA file","VNA file couldn't be opened, verify if the file exists or has the right extension","error")
            return -1
        else:

            #Depth clean
            self.flagVNA = False
            self.lockVNA = True

            self.pathVNA.append(pathVNA)
            tree = self.ui.treeVNA

            self.I_vna = [[]]*len(self.pathVNA)
            self.Q_vna = [[]]*len(self.pathVNA)
            self.freq_Q = [[]]*len(self.pathVNA)

            self.prevMPH = [0.]*len(self.pathVNA)
            self.prevMPD = [0.]*len(self.pathVNA)
            self.prevNp = [0.]*len(self.pathVNA)
            self.prevOrder = [0.]*len(self.pathVNA)
            self.prevNpCont = [0.]*len(self.pathVNA)
            self.prevOrderCont = [0.]*len(self.pathVNA)

            # Parameters
            self.flagFind = [False]*len(self.pathVNA)
            self.flagCont = [False]*len(self.pathVNA)
            self.base = [0]*len(self.pathVNA)

            # Resonance frequencies
            self.ind = [[]]*len(self.pathVNA)
            self.vlines = [[]]*len(self.pathVNA)
            self.fnote_ann = [[]]*len(self.pathVNA)

            self.clearTree(tree, False)

            for k in xrange(len(self.pathVNA)):
                vnaFile = QTreeWidgetItem(tree)
                vnaFile.setIcon(0, QIcon('./resources/plot.png'))
                vnaFile.setFlags(vnaFile.flags() | Qt.ItemIsUserCheckable)

                fileVNA = self.getNameFile(self.pathVNA[k])

                vnaFile.setText(0, fileVNA)
                vnaFile.setCheckState(0, Qt.Unchecked)  

    # ------- Remove files
    def clearVNATree(self,event):
        self.clearTree(self.ui.treeVNA, False)
        self.pathVNA = []

        self.flagVNA = False
        self.lockVNA = True

        self.ind = []
        self.I_vna = []
        self.Q_vna = []
        self.freq_Q = []

    def getNameFile(self, path):
        file = ""
        for i in range(1,len(path)):
            if path[-i] == "/":
                break  
            else:
                file = path[-i] + file
        return file

    # ------- Select VNA files
    def selectedVNA(self):
        root = self.ui.treeVNA.invisibleRootItem()
        signal_count = root.childCount()

        pathVNA = []
        self.indexPath = []

        for i in range(signal_count):
            fileVNA = root.child(i)

            if fileVNA.checkState(0) == QtCore.Qt.Checked:
                pathVNA.append(self.pathVNA[i])
                self.indexPath.append(i)

        return pathVNA

    # --- Plot full VNA sweep
    def plotVnaSweep(self):

        plot_VNA = self.selectedVNA()

        if len(plot_VNA) == 0:
            self.messageBox("No directory","Directory not selected","warning")
            self.ui.setEnabled(True)
            return

        self.ui.setEnabled(False)
        self.messageBox("Loading VNA sweep","The VNA sweep will be loaded, it would takes several minutes","info")

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        p = ""
        if self.ui.pointPlot.isChecked():
            p = "*"

        if plot_VNA != self.prevPathVNA:
            self.flagVNA = True 
            self.prevPathVNA = plot_VNA
        else:
            self.flagVNA = False

        try:
            cnt = 0

            self.f1 = self.fig1.add_subplot(111)
            self.ui.editSweepBox.setMaximum(len(plot_VNA))

            for path in plot_VNA:
                
                m = self.indexPath[cnt]

                # Change the name, in case of have "_" replace for ","
                name = ""
                fileVNA = self.getNameFile(path)
                for l in fileVNA:
                    if l == "_":
                        name = name + ","
                    else:
                        name = name + l  

                if self.flagVNA or self.lockVNA:

                    freq, mag_norm, I, Q = self.dataRedtn.get_full_vna(str(path))

                    self.I_vna[m] = I
                    self.Q_vna[m] = Q
                    self.freq_Q[m] = freq
                else:
                    mag_norm  = np.sqrt(self.I_vna[m]**2 + self.Q_vna[m]**2)
                    freq = self.freq_Q[m]

                # Remove Continuos 
                if self.ui.remContBtn.isChecked():
                    # Delete the continuous
                    npointsCon = self.ui.nPointsCont.value()
                    orderCon = self.ui.orderCont.value()

                    # Cleaning flags
                    if npointsCon != self.prevNpCont[m] or orderCon != self.prevOrderCont[m]:
                        self.flagCont[m] = False 
                        self.prevNpCont[m] = npointsCon
                        self.prevOrderCont[m] = orderCon
                    else:
                        self.flagCont[m] = True

                    if self.flagCont[m] == False or self.lockVNA:
                        self.base[m] = savgol_filter(mag_norm, npointsCon, orderCon)
                        self.flagCont[m] = True

                    mag = mag_norm - self.base[m] 
                else:
                    mag = mag_norm

                # Find Resonators 
                if self.ui.findResBtn.isChecked():

                    # Filter less order
                    npoints = self.ui.nPointsFind.value()
                    order = self.ui.orderFind.value()

                    mag_wBL = savgol_filter(mag,npoints,order)
                    mag_wBL = -1*mag_wBL

                    if self.ui.AutoBtn.isChecked():
                        # Parameters MPH and MPD 
                        mph = np.max(mag_wBL)/8
                        mpd = len(freq)/500

                        self.ui.mphBox.setValue(mph)
                        self.ui.mpdBox.setValue(mpd)

                        if npoints != self.prevNp[m] or order != self.prevOrder[m]:
                            self.flagFind[m] = False 
                            self.prevNp[m] = npoints
                            self.prevOrder[m] = order
                        else:
                            self.flagFind[m] = True
                    else:
                        # Parameters
                        mph = self.ui.mphBox.value()
                        mpd = self.ui.mpdBox.value()

                        if npoints != self.prevNp[m] or order != self.prevOrder[m] or mph != self.prevMPH[m] or mpd != self.prevMPD[m]:
                            self.flagFind[m] = False 
                            self.prevNp[m] = npoints
                            self.prevOrder[m] = order
                            self.prevMPD[m] = mpd
                            self.prevMPH[m] = mph
                        else:
                            self.flagFind[m] = True

                    if self.flagFind[m] == False or self.lockVNA:
                        ind = detect_peaks(signal.detrend(mag_wBL),mph=mph,mpd=mpd)
                        self.ui.statusText.append("KIDS = " + str(len(ind)) + " founded")

                        # Store for Qi and Qc Calculus
                        self.ind[m] = ind
                        self.flagFind[m] = True

                    if len(self.ind) > 0:
                        self.flag_ind = True
                    else:
                        self.flag_ind = False

                colorline = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # Plotting VNA sweep 
                if self.ui.plotRemContBtn.isChecked():
                    mag_plot = np.max(mag_norm) + mag_norm - self.base[m] 
                else:
                    mag_plot = mag_norm

                if self.ui.ylog.isChecked():
                    mag_plot = 20*np.log10(mag_plot)
                    self.f1.set_ylabel(r'$S_{21}[dB]$')
                    self.f1.plot(freq,mag_plot,p+'-',alpha=1,label=r"$"+str(name)+"$",color=colorline[m])
                else:
                    self.f1.set_ylabel(r'$S_{21}[V]$')
                    self.f1.plot(freq,mag_plot,p+'-',alpha=1,label=r"$"+str(name)+"$",color=colorline[m])

                if len(self.ind) > 0:
                    line_full_sweep = []
                    aux_fnote_ann = []

                    for i in self.ind[m]:
                        i = int(i)
                        line = self.f1.axvline(freq[i],color=colorline[m], linewidth=0.75)
                        line_full_sweep.append(line)

                        fnote_ann = self.f1.annotate(r"$"+str(freq[i]/1e6)+"MHz$",xy=(freq[i],mag_plot[i]))
                        self.f1.plot(freq[i],mag_plot[i],'o',color=colorline[m])

                        if self.ui.fnoteBtn.isChecked():
                            fnote_ann.set_visible(True)
                        else:
                            fnote_ann.set_visible(False)

                        aux_fnote_ann.append(fnote_ann)

                    self.fnote_ann[m] = aux_fnote_ann
                    self.vlines[m] = line_full_sweep

                self.f1.set_xlabel(r'$Frequency [Hz]$')

                cnt = cnt + 1

            if self.ui.squareLeg.isChecked():
                self.f1.legend(loc='best')

            self.f1.figure.canvas.draw()

        except Exception as e:
            self.messageBox("Error","Error loading VNA sweep." + str(e),"error")

        self.lockVNA = False
        self.ui.setEnabled(True)

        if self.flagResize == False:
            self.resizeTree(self.flagResize)

    def resDraw(self,event):
        resLine = event.xdata

        y_lims = self.f1.get_ylim()
        mid_y_axis = (y_lims[1] - y_lims[0])/2
        mid_y_axis = mid_y_axis + y_lims[0]

        if len(self.freq_Q)>0:
            i = self.ui.editSweepBox.value()-1
            i = self.indexPath[i]
            f0_ind = np.argmin(np.abs(self.freq_Q[i] - resLine))

            if self.ui.drawBtn.isChecked():

                colorline = plt.rcParams['axes.prop_cycle'].by_key()['color']

                if not f0_ind in self.ind[i]:
                    self.ind[i] = np.append(self.ind[i],int(f0_ind))
                    line = self.f1.axvline(resLine,color=colorline[i],linewidth=0.75)
                    self.vlines[i].append(line)
                    self.f1.plot(resLine,mid_y_axis,'bo')
                    self.f1.annotate(r"$"+str(resLine/1e6)+"MHz$",xy=(resLine,mid_y_axis))

                self.f1.figure.canvas.draw()

            elif self.ui.eraseBtn.isChecked():

                if f0_ind in self.ind[i]:
                    index = np.where(self.ind[i]==f0_ind)[0][0]
                    self.vlines[i][index].remove()
                    self.ind[i] = np.delete(self.ind[i], index)
                    del self.vlines[i][index]

                self.f1.figure.canvas.draw()

    def plotQanalysis(self, event):

        if len(self.ind) == 0:
            self.messageBox("No directory","Directory not selected","warning")
            self.ui.setEnabled(True)
            return

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        qFlags = [self.ui.actionQ_Factor.isChecked(), self.ui.actionGet_Qc.isChecked(), self.ui.actionGet_Qi.isChecked(), self.ui.actionNonlinearity.isChecked()]

        nQtags = 0
        for nq in qFlags:
            if nq:
                nQtags += 1

        number = len(self.ind[0])
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

        m = 0
        n = 0
        for point in self.acumQ:
            f0 = point[0]
            Qr = point[1]
            Qc = point[2]
            Qi = point[3]
            a = point[5]

            cnt = 0 
            for k in range(4):

                if qFlags[k]:
                    cnt += 1

                if k == 0 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_r$')
                    self.f1.plot(f0,Qr,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qr))

                if k == 1 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_c$')
                    self.f1.plot(f0,Qc,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qc))

                if k == 2 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_i$')
                    self.f1.plot(f0,Qi,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qi))

                if k == 3 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$Value$')
                    self.f1.set_title(r'$Nonlinearity$')
                    self.f1.plot(f0,a,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,a))
            
            self.f1.set_xlabel(r'$f_0$')

            m = m + 1 
            if m >= len(self.ind[n]):
                m = 0
                n = n + 1
                if n < len(self.ind):
                    number = len(self.ind[n])
                    cmap = plt.get_cmap('gnuplot')
                    colors = [cmap(i) for i in np.linspace(0, 1, number)]

        self.f1.figure.canvas.draw()

    def plotQ_shapes(self,event):
        if len(self.acumQ) > 0:

            cnt = 0
            for j in range(len(self.ind)):

                # Bandwidth frequency
                bw = self.ui.bwRes.value()
                step = self.freq_Q[j][1] - self.freq_Q[j][0]
                bw_ind = int(1.0e3*bw/step)/2

                for i in range(len(self.ind[j])):

                    if self.ui.ylog.isChecked():
                        fit_curve = 20*np.log10(self.fit_curve[cnt])
                    else:
                        fit_curve = self.fit_curve[cnt]

                    potp = self.acumQ[cnt]

                    q_msg = r"$"+str(round(potp[0]/1e6,4))+"MHz$"

                    if self.ui.actionQ_Factor.isChecked():
                        q_msg = q_msg + "\n$Q_r="+str(round(potp[1],2))+"$" 

                    if self.ui.actionGet_Qi.isChecked():
                        q_msg = q_msg + "\n$Q_i="+str(round(potp[3],2))+"$" 

                    if self.ui.actionGet_Qc.isChecked():
                        q_msg = q_msg + "\n$Q_c="+str(round(potp[2],2))+"$" 

                    self.f1.annotate(q_msg,xy=(potp[0],np.min(fit_curve)))       
                    self.f1.plot(self.freq_Q[j][int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind],fit_curve,'--')

                    cnt = cnt + 1

            self.f1.figure.canvas.draw()

        else:
            self.messageBox("Plotting Q","Length of Q array has to be more than 0","error")

    def getQi_Qc(self,event):

        self.ui.setEnabled(False)
        self.messageBox("Q fit","Starting the fitting...","info")

        try:
            self.acumQ = []
            self.fit_curve = []
            if self.flag_ind == True or len(self.ind) > 0:

                for j in range(len(self.ind)):

                    # Bandwidth frequency
                    bw = self.ui.bwRes.value()
                    step = self.freq_Q[j][1] - self.freq_Q[j][0]
                    bw_ind = int(1.0e3*bw/step)/2

                    approx = self.ui.approxQrBtn.value()

                    S21_cplx = self.I_vna[j] + 1j*self.Q_vna[j]

                    for i in range(len(self.ind[j])):

                        if self.ui.nonlinearBtn.isChecked():
                            potp,perr,fit_curve = self.fit_S21.fitmags21(self.freq_Q[j][int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind],S21_cplx[int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind], approxQr=approx, nonlinear=True)
                        else:
                            potp,perr,fit_curve = self.fit_S21.fitmags21(self.freq_Q[j][int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind],S21_cplx[int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind], approxQr=approx, nonlinear=False)

                        self.fit_curve.append(fit_curve)

                        if self.ui.ylog.isChecked():
                            fit_curve = 20*np.log10(fit_curve)

                        self.acumQ.append(potp)
                        #array([f0, Qr, Qc, Qi, A, a])

                        q_msg = r"$"+str(round(potp[0]/1e6,4))+"MHz$"

                        if self.ui.actionQ_Factor.isChecked():
                            q_msg = q_msg + "\n$Q_r="+str(round(potp[1],2))+"$" 

                        if self.ui.actionGet_Qi.isChecked():
                            q_msg = q_msg + "\n$Q_i="+str(round(potp[3],2))+"$" 

                        if self.ui.actionGet_Qc.isChecked():
                            q_msg = q_msg + "\n$Q_c="+str(round(potp[2],2))+"$" 

                        self.f1.annotate(q_msg,xy=(potp[0],np.min(fit_curve)))
                        self.f1.plot(self.freq_Q[j][int(self.ind[j][i]) - bw_ind:int(self.ind[j][i]) + bw_ind],fit_curve,'--')
                    
                    self.f1.figure.canvas.draw()
            else:
                self.messageBox("Q fit","There is not VNA resonators. Get the resonators first","error")

        except Exception as e:
            self.messageBox("Q fit", "There is an error in the quality factor calculation, have a look of the parameters."+str(e),"error")

        self.ui.setEnabled(True)

    # VNA stadistics for a while
    def VNAstad(self,event):
        self.VNA_stad_curves()

    def VNA_stad_curves(self):

        f0 = []
        Qr = []
        Qi = []
        Qc = []
        a = []

        # Mean of Q data
        for q in self.acumQ:
            f0.append(q[0])
            Qr.append(q[1])
            Qc.append(q[2])
            Qi.append(q[3])
            a.append(q[5])        

        meanQr = np.mean(Qr)
        meanQi = np.mean(Qi)
        meanQc = np.mean(Qc)
        meanA = np.mean(a)

        stdQr = np.std(Qr)
        stdQi = np.std(Qi)
        stdQc = np.std(Qc)
        stdA = np.std(a)

        self.ui.statusText.append("****Qr****")
        self.ui.statusText.append("Mean: " + str(meanQr))
        self.ui.statusText.append("Std: " + str(stdQr))

        self.ui.statusText.append("****Qi****")
        self.ui.statusText.append("Mean: " + str(meanQi))
        self.ui.statusText.append("Std: " + str(stdQr))

        self.ui.statusText.append("****Qc****")
        self.ui.statusText.append("Mean: " + str(meanQc))
        self.ui.statusText.append("Std: " + str(stdQr))

        self.ui.statusText.append("****Nonlinearity****")
        self.ui.statusText.append("Mean: " + str(meanA))
        self.ui.statusText.append("Std: " + str(stdA))
        self.ui.statusText.append("**********")

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        try:
            self.nbin = self.stad.nbinsBox.value()
            
            if self.stad.PDFBtn.isChecked():
                self.pdfFlag = True
            else:
                self.pdfFlag = False

            if self.stad.PDABtn.isChecked():
                self.pdaFlag = True
            else:
                self.pdaFlag = False

            if self.stad.drawCurveCB.isChecked():
                self.drawCurve = True
            else:
                self.drawCurve = False

            if self.stad.writeLegBtn.isChecked():
                self.writeLeg = True
            else:
                self.writeLeg = False

            if str(self.stad.PDFbox.currentText()) == "Gamma":
                self.pdfType = "Gamma"
            elif str(self.stad.PDFbox.currentText()) == "Gaussian":      
                self.pdfType = "Gaussian"

        except Exception as e:
            self.nbin = 30
            self.pdfFlag = True
            self.pdaFlag = False
            self.drawCurve = False
            self.writeLeg = False
            self.pdfType = "Gamma"

        totQr = len(Qr)
        totQi = len(Qi)
        totQc = len(Qc)
        tota = len(a)

        Qr_hist,bin_Qr = np.histogram(Qr, self.nbin)
        Qi_hist,bin_Qi = np.histogram(Qi, self.nbin)
        Qc_hist,bin_Qc = np.histogram(Qc, self.nbin)
        a_hist,bin_a = np.histogram(a, self.nbin)

        if self.pdaFlag:
            # Qr
            aux = 0
            for k in range(len(Qr_hist)):
                aux = aux + Qr_hist[k]
                Qr_hist[k] = aux
            Qr_hist = Qr_hist/float(np.sum(Qr_hist))
            Qr_hist = Qr_hist/np.max(Qr_hist)
            # Qi
            aux = 0
            for k in range(len(Qi_hist)):
                aux = aux + Qi_hist[k]
                Qi_hist[k] = aux
            Qi_hist = Qi_hist/float(np.sum(Qi_hist))
            Qi_hist = Qi_hist/np.max(Qi_hist)
            # Qc
            aux = 0
            for k in range(len(Qc_hist)):
                aux = aux + Qc_hist[k]
                Qc_hist[k] = aux
            Qc_hist = Qc_hist/float(np.sum(Qc_hist))
            Qc_hist = Qc_hist/np.max(Qc_hist)
            # a
            aux = 0
            for k in range(len(a_hist)):
                aux = aux + a_hist[k]
                a_hist[k] = aux
            a_hist = a_hist/float(np.sum(a_hist))
            a_hist = a_hist/np.max(a_hist)

        self.f1 = self.fig1.add_subplot(221)
        if self.pdfFlag:          
            self.f1.hist(Qr, self.nbin, edgecolor='black', density=True)
        if self.pdaFlag:
            nw_qr_bin = []
            for i in range(len(Qr_hist)):
                nw_qr_bin.append(bin_Qr[i]+(bin_Qr[i+1]-bin_Qr[i])/2)
            self.f1.plot(nw_qr_bin, Qr_hist,'r*-')
        self.f1.set_title(r'$Qr$')
        self.f1.set_ylabel(r"$Frequency$")

        self.f1 = self.fig1.add_subplot(222)
        if self.pdfFlag:          
            self.f1.hist(Qi, self.nbin, edgecolor='black', density=True)
        if self.pdaFlag:
            nw_qi_bin = []
            for i in range(len(Qi_hist)):
                nw_qi_bin.append(bin_Qi[i]+(bin_Qi[i+1]-bin_Qi[i])/2)
            self.f1.plot(nw_qr_bin, Qi_hist,'r*-')
        self.f1.set_title(r"$Qi$")

        self.f1 = self.fig1.add_subplot(223)
        if self.pdfFlag:          
            self.f1.hist(Qc, self.nbin, edgecolor='black', density=True)
        if self.pdaFlag:
            nw_qc_bin = []
            for i in range(len(Qc_hist)):
                nw_qc_bin.append(bin_Qc[i]+(bin_Qc[i+1]-bin_Qc[i])/2)
            self.f1.plot(nw_qc_bin, Qc_hist,'r*-')
        self.f1.set_title(r"$Qc$")
        self.f1.set_xlabel(r"$Q value$")
        self.f1.set_ylabel(r"$Frequency$")

        self.f1 = self.fig1.add_subplot(224)
        if self.pdfFlag:          
            self.f1.hist(a, self.nbin, edgecolor='black', density=True)
        if self.pdaFlag:
            nw_a_bin = []
            for i in range(len(a_hist)):
                nw_a_bin.append(bin_a[i]+(bin_a[i+1]-bin_a[i])/2)
            self.f1.plot(nw_a_bin, a_hist,'r*-')
        self.f1.set_xlabel(r"$Value$")
        self.f1.set_title(r"$Nonlinearity$")

        if self.pdfFlag:
            if self.drawCurve:
                if self.pdfType == "Gamma":
                    #Qr
                    self.f1 = self.fig1.add_subplot(221)
                    fit_alpha_qr, fit_loc_qr, fit_beta_qr = gamma.fit(Qr)
                    n, bins, patches = plt.hist(Qr, self.nbin, normed=1)
                    y_qr = scipy.stats.gamma.pdf(bins, fit_alpha_qr, fit_loc_qr, fit_beta_qr)
                    self.f1.plot(bins, y_qr, 'r--', linewidth=2, label="A=" + str(round(fit_alpha_qr,1)) + "\nLoc=" + str(round(fit_loc_qr,1)) + "\nB=" + str(round(fit_beta_qr,1)))
                    
                    #Qi
                    self.f1 = self.fig1.add_subplot(222)
                    fit_alpha_qi, fit_loc_qi, fit_beta_qi = gamma.fit(Qi)
                    n, bins, patches = plt.hist(Qi, self.nbin, normed=1)
                    y_qi = scipy.stats.gamma.pdf(bins, fit_alpha_qi, fit_loc_qi, fit_beta_qi)
                    self.f1.plot(bins, y_qi, 'r--', linewidth=2, label="A=" + str(round(fit_alpha_qi,1)) + "\nLoc=" + str(round(fit_loc_qi,1)) + "\nB=" + str(round(fit_beta_qi,1)))
                    
                    #Qc
                    self.f1 = self.fig1.add_subplot(223)
                    fit_alpha_qc, fit_loc_qc, fit_beta_qc = gamma.fit(Qc)
                    n, bins, patches = plt.hist(Qc, self.nbin, normed=1)
                    y_qc = scipy.stats.gamma.pdf(bins, fit_alpha_qc, fit_loc_qc, fit_beta_qc)
                    self.f1.plot(bins, y_qc, 'r--', linewidth=2, label="A=" + str(round(fit_alpha_qc,1)) + "\nLoc=" + str(round(fit_loc_qc,1)) + "\nB=" + str(round(fit_beta_qc,1)))
                    
                    #Nonlinearity
                    self.f1 = self.fig1.add_subplot(224)
                    fit_alpha_a, fit_loc_a, fit_beta_a = gamma.fit(a)
                    n, bins, patches = plt.hist(a, self.nbin, normed=1)
                    y_a = scipy.stats.gamma.pdf(bins, fit_alpha_a, fit_loc_a, fit_beta_a)
                    self.f1.plot(bins, y_a, 'r--', linewidth=2, label="A=" + str(round(fit_alpha_a,1)) + "\nLoc=" + str(round(fit_loc_a,1)) + "\nB=" + str(round(fit_beta_a,1)))

                    if self.writeLeg:
                        #Qr
                        self.f1 = self.fig1.add_subplot(221)
                        self.f1.legend(loc='best')                       
                        #Qi
                        self.f1 = self.fig1.add_subplot(222)
                        self.f1.legend(loc='best') 
                        #Qc
                        self.f1 = self.fig1.add_subplot(223)
                        self.f1.legend(loc='best') 
                        #Nonlinearity
                        self.f1 = self.fig1.add_subplot(224)
                        self.f1.legend(loc='best') 

                if self.pdfType == "Gaussian":
                    #Qr
                    self.f1 = self.fig1.add_subplot(221)
                    (mu, sigma) = norm.fit(Qr)
                    n, bins, patches = plt.hist(Qr, self.nbin, normed=1)
                    y = scipy.stats.norm.pdf(bins, mu, sigma)
                    self.f1.plot(bins, y, 'r--', linewidth=2, label="m=" + str(round(mu,1)) + "\ns=" + str(round(sigma,1)))

                    #Qi
                    self.f1 = self.fig1.add_subplot(222)
                    (mu, sigma) = norm.fit(Qi)
                    n, bins, patches = plt.hist(Qi, self.nbin, normed=1)
                    y = scipy.stats.norm.pdf(bins, mu, sigma)
                    self.f1.plot(bins, y, 'r--', linewidth=2, label="m=" + str(round(mu,1)) + "\ns=" + str(round(sigma,1)))

                    #Qc
                    self.f1 = self.fig1.add_subplot(223)
                    (mu, sigma) = norm.fit(Qc)
                    n, bins, patches = plt.hist(Qc, self.nbin, normed=1)
                    y = scipy.stats.norm.pdf(bins, mu, sigma)
                    self.f1.plot(bins, y, 'r--', linewidth=2, label="m=" + str(round(mu,1)) + "\ns=" + str(round(sigma,1)))

                    #Nonlinearity
                    self.f1 = self.fig1.add_subplot(224)
                    (mu, sigma) = norm.fit(a)
                    n, bins, patches = plt.hist(a, self.nbin, normed=1)
                    y = scipy.stats.norm.pdf(bins, mu, sigma)
                    self.f1.plot(bins, y, 'r--', linewidth=2, label="m=" + str(round(mu,1)) + "\ns=" + str(round(sigma,1)))

                    if self.writeLeg:
                        #Qr
                        self.f1 = self.fig1.add_subplot(221)
                        self.f1.legend(loc='best')                       
                        #Qi
                        self.f1 = self.fig1.add_subplot(222)
                        self.f1.legend(loc='best') 
                        #Qc
                        self.f1 = self.fig1.add_subplot(223)
                        self.f1.legend(loc='best') 
                        #Nonlinearity
                        self.f1 = self.fig1.add_subplot(224)
                        self.f1.legend(loc='best') 

        self.f1.figure.canvas.draw()

    def paintQMap(self,event):

        self.Qmap = uic.loadUi("gui/Q_map.ui")

        screen = QtGui.QDesktopWidget().screenGeometry() 

        size_x = screen.width()
        size_y = screen.height()

        self.Qmap.move(self.size_x/2-880/2,self.size_y/2-380/2)

        # Load Map
        self.Qmap.loadMapBtn.mousePressEvent = self.loadMapArray

        # Null element
        self.Qmap.nullBtn.mousePressEvent = self.nullItem
        # Add button
        self.Qmap.addBtn.mousePressEvent = self.addItem
        # Delete one button
        self.Qmap.remOneBtn.mousePressEvent = self.remOneItem
        # Delete all the items
        self.Qmap.remAllBtn.mousePressEvent = self.delAllItem

        # Load List
        self.Qmap.loadListBtn.mousePressEvent = self.loadList
        # Save List
        self.Qmap.saveListBtn.mousePressEvent = self.saveList

        # Load List
        self.Qmap.doneBtn.mousePressEvent = self.plotQmap

        # Dummy Q array
        """
        self.acumQ = [[450e6,15000,30000,20000,0.5,0.2],[510e6,18000,25000,15000,0.2,0.1],
                     [550e6,17000,32000,22000,0.1,0.15],[650e6,17500,22000,12000,0.1,0.15],
                     [520e6,17000,32000,22000,0.1,0.15],[750e6,17500,22000,12000,0.1,0.15],
                     [550e6,17000,32000,22000,0.1,0.15],[670e6,17500,22000,12000,0.1,0.15],
                     [560e6,17000,32000,22000,0.1,0.15],[620e6,17500,22000,12000,0.1,0.15],
                     [570e6,17000,32000,22000,0.1,0.15],[622e6,17500,22000,12000,0.1,0.15],
                     [588e6,17000,32000,22000,0.1,0.15],[651e6,17500,22000,12000,0.1,0.15],
                     [551e6,17000,32000,22000,0.1,0.15],[610e6,17500,22000,12000,0.1,0.15],
                     [554e6,17000,32000,22000,0.1,0.15],[623e6,17500,22000,12000,0.1,0.15],
                     [500e6,17000,32000,22000,0.1,0.15],[750e6,17000,32000,22000,0.1,0.15]]
        """
    
        model = QtGui.QStandardItemModel()
        self.Qmap.listQ.setModel(model)

        self.Qmap.listQ.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.Qmap.sortListQ.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        for i in range(len(self.acumQ)):
            txt = "Res " + str(i) + " " + str(self.acumQ[i][0])
            item = QtGui.QStandardItem(txt)
            model.appendRow(item)

        self.sortedItems = []
        self.sortedNum = []
        self.number = []

        self.xMap,self.yMap = np.array([]), np.array([])

        self.figMap = Figure()
        self.fmap = self.figMap.add_subplot(111)
        self.fmap.grid(False)
        self.fmap.axis('off')
        self.fmap.axis('equal')
        self.addmplMap(self.figMap)

        self.Qmap.show()

    def plotQmap(self,event):
        
        f0 = []
        Qr = []
        Qi = []
        Qc = []
        a = []

        # Mean of Q data
        for q in self.acumQ:
            f0.append(q[0])
            Qr.append(q[1])
            Qc.append(q[2])
            Qi.append(q[3])
            a.append(q[5])   

        # Sort the array
        auxQr,auxQi,auxQc,auxNon,auxf0 = [],[],[],[],[]
        for i in range(len(self.sortedNum)):
            if self.sortedNum[i] != []:
                auxf0.append(f0[self.sortedNum[i]])
                auxQr.append(Qr[self.sortedNum[i]])
                auxQc.append(Qc[self.sortedNum[i]])
                auxQi.append(Qi[self.sortedNum[i]])
                auxNon.append(a[self.sortedNum[i]])
            else:
                auxf0.append(0.)
                auxQr.append(0.)
                auxQc.append(0.)
                auxQi.append(0.)
                auxNon.append(0.)

        Qr = auxQr
        Qc = auxQc
        Qi = auxQi
        a = auxNon
        f0 = auxf0

        wd = 1
        sx,sy = self.xMap.shape

        size_x = np.max(self.xMap)
        size_y = np.max(self.yMap)

        pix_Qr = np.zeros((int(size_x + 2*wd + 1),int(size_y + 2*wd + 1)))
        pix_Qc = np.zeros((int(size_x + 2*wd + 1),int(size_y + 2*wd + 1)))
        pix_Qi = np.zeros((int(size_x + 2*wd + 1),int(size_y + 2*wd + 1)))
        pix_Qa = np.zeros((int(size_x + 2*wd + 1),int(size_y + 2*wd + 1)))
        pix_f0 = np.zeros((int(size_x + 2*wd + 1),int(size_y + 2*wd + 1)))

        for i in range(sx):
            for j in range(sy):
                m = int(self.xMap[i][j])
                n = int(self.yMap[i][j])

                for h in range(2*wd):
                    for l in range(2*wd):
                        pix_Qr[m+h][n+l] = Qr[i*(sx-1) + j]
                        pix_Qi[m+h][n+l] = Qi[i*(sx-1) + j]
                        pix_Qc[m+h][n+l] = Qc[i*(sx-1) + j]
                        pix_Qa[m+h][n+l] = a[i*(sx-1) + j]
                        pix_f0[m+h][n+l] = f0[i*(sx-1) + j]/1e6

        pix_All = []
        pix_names = []

        if self.ui.actionQ_Factor.isChecked():
            pix_All.append(pix_Qr)
            pix_names.append("Qr")
        if self.ui.actionGet_Qc.isChecked():
            pix_All.append(pix_Qc)
            pix_names.append("Qc")
        if self.ui.actionGet_Qi.isChecked():
            pix_All.append(pix_Qi)
            pix_names.append("Qi")
        if self.ui.actionNonlinearity.isChecked():
            pix_All.append(pix_Qa)
            pix_names.append("Nonlinearity")

        self.Plots = uic.loadUi("gui/plots.ui")

        screen = QtGui.QDesktopWidget().screenGeometry() 

        size_x = screen.width()
        size_y = screen.height()

        self.Plots.move(size_x/2-1110/2,size_y/2-610/2)
        self.Plots.plotFrame.setLayout(self.Plots.MainPlot)

        # Creation of Plot
        self.figPlotMap = Figure()
        self.fplotmap = self.figPlotMap.add_subplot(111)
        self.fplotmap.grid(False)
        self.fplotmap.axis('off')
        self.fplotmap.axis('equal')
        self.addmplPlotMap(self.figPlotMap)

        h = len(pix_All)
        w = 1
        if h == 4:
            w = 2
            h = 2

        for i in range(len(pix_All)):
            self.fplotmap = self.figPlotMap.add_subplot(w,h,i+1)
            im = self.fplotmap.imshow(pix_All[i],cmap="hot")
            self.fplotmap.grid(False)
            self.fplotmap.axis('off')
            self.fplotmap.axis('equal')

            # Legend
            for m in range(sx):
                for n in range(sy):
                    self.fplotmap.text(self.yMap[m][n], self.xMap[m][n], round(pix_f0[int(self.xMap[m][n])][int(self.yMap[m][n])],2), ha="center", va="center", color="w",fontsize=8)

        self.fplotmap.figure.canvas.draw()
        self.Plots.show()

    def saveList(self,event):
        np.save("config/order_map.npy", self.sortedNum)

    def loadList(self,event):
        if len(self.acumQ)>0:
            self.sortedNum = np.load("config/order_map.npy").tolist()
    
            for i in range(len(self.acumQ)):
                txt = "Res " + str(i) + " " + str(self.acumQ[i][0])
                self.sortedItems.append(txt)

            aux = ['']*len(self.sortedItems)
            for i in range(len(self.sortedNum)):
                if i < len(self.sortedItems):
                    aux[i] = self.sortedItems[self.sortedNum[i]]

            self.sortedItems = aux

            model = QtGui.QStandardItemModel()
            self.Qmap.sortListQ.setModel(model)

            for i in range(len(self.sortedItems)):
                txt = self.sortedItems[i]
                item = QtGui.QStandardItem(txt)
                model.appendRow(item)

            self.plotNumbers()
            self.fmap.figure.canvas.draw()            

    def delAllItem(self,event):
        model = QtGui.QStandardItemModel()
        self.Qmap.sortListQ.setModel(model)

        model.removeRows( 0, model.rowCount())

        self.sortedNum = []
        self.sortedItems = []

        self.plotNumbers()
        self.fmap.figure.canvas.draw()

    def remOneItem(self,event):
        itms = self.Qmap.sortListQ.selectedIndexes()

        for data in itms:
            txt = data.data().toString()

            self.sortedItems.remove(txt)
            until = ""
            for m in range(4,len(txt)):
                if txt[m] == " ":
                    break
                else:
                    until = until + txt[m]
            self.sortedNum.remove(int(until))

        model = QtGui.QStandardItemModel()
        self.Qmap.sortListQ.setModel(model)

        for i in self.sortedItems:
            item = QtGui.QStandardItem(i)
            model.appendRow(item)

        self.plotNumbers()
        self.fmap.figure.canvas.draw()

    def addItem(self,event):
        if self.xMap.shape[0]>0:
            model = QtGui.QStandardItemModel()
            self.Qmap.sortListQ.setModel(model)

            itms = self.Qmap.listQ.selectedIndexes()

            for data in itms:
                txt = data.data().toString()
                if not txt in self.sortedItems:
                    sx,sy = self.xMap.shape
                    if len(self.sortedNum) < sx*sy:
                        self.sortedItems.append(txt)
                        until = ""
                        for m in range(4,len(txt)):
                            if txt[m] == " ":
                                break
                            else:
                                until = until + txt[m]
                        self.sortedNum.append(int(until))

            for i in self.sortedItems:
                item = QtGui.QStandardItem(i)
                model.appendRow(item)

            self.plotNumbers()
            self.fmap.figure.canvas.draw()

    def nullItem(self,event):
        if self.xMap.shape[0]>0:
            model = QtGui.QStandardItemModel()
            self.Qmap.sortListQ.setModel(model)

            sx,sy = self.xMap.shape 
            if len(self.sortedNum) < sx*sy:
                # Empty element
                self.sortedNum.append([])
                self.sortedItems.append("Empty")

            for i in self.sortedItems:
                item = QtGui.QStandardItem(i)
                model.appendRow(item)

            self.plotNumbers()
            self.fmap.figure.canvas.draw()

    def plotNumbers(self):
        if self.xMap.shape[0]>0:
            if len(self.number) > 0:
                n = 0
                for m in self.number:
                    m.remove()

            self.number = []

            sx,sy = self.xMap.shape 
            k = 0

            for i in range(sx):
                for j in range(sy):
                    if k < len(self.sortedNum):
                        if self.sortedNum[k] != []:
                            numbers = self.fmap.text(self.xMap[i][j],self.yMap[i][j],str(self.sortedNum[k]),fontsize=10)
                        else:
                            numbers = self.fmap.text(self.xMap[i][j],self.yMap[i][j],"e",fontsize=10)
                    else:
                        break
                    self.number.append(numbers)
                    k = k + 1

    def loadMapArray(self,event):
        self.xMap,self.yMap = np.load("config/map_array.npy")
        sx,sy = self.xMap.shape

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        wd = 1
        for i in range(sx):
            for j in range(sy):
                #Center
                self.fmap.plot(self.xMap[i][j],self.yMap[i][j],'b*')

                #Square
                self.fmap.plot([self.xMap[i][j] - wd,self.xMap[i][j] - wd],[self.yMap[i][j] - wd,self.yMap[i][j] + wd],'r',linewidth=0.75)
                self.fmap.plot([self.xMap[i][j] - wd,self.xMap[i][j] + wd],[self.yMap[i][j] + wd,self.yMap[i][j] + wd],'r',linewidth=0.75)
                self.fmap.plot([self.xMap[i][j] + wd,self.xMap[i][j] + wd],[self.yMap[i][j] + wd,self.yMap[i][j] - wd],'r',linewidth=0.75)
                self.fmap.plot([self.xMap[i][j] + wd,self.xMap[i][j] - wd],[self.yMap[i][j] - wd,self.yMap[i][j] - wd],'r',linewidth=0.75)

        self.plotNumbers()
        self.fmap.figure.canvas.draw()

    def plotAllTS(self):
        p = ""
        c1 = ""
        c2 = ""
        alpha = 1
        alpha_on = 0.75

        plotPar = [c1,c2,alpha_on,p]

        paths, shortName, namePlot, headPath = self.kidSelected()

        if len(paths) == 0:
            self.messageBox("Error","Select at least one KID to plot","error")    
            self.ui.setEnabled(True)
            return

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        self.f1 = self.fig1.add_subplot(111)

        num = 0
        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
                self.ui.setEnabled(True)
                return

            ind_cut_freq = np.where(psd[0]>100.)[0][0]
           
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
            except Exception as e:
                pass

            self.flagResize = True

    # Plot IQ time stream High Resolution  
    def plotIQ_HR_TS(self):

        paths, shortName, namePlot, headPath = self.kidSelected()

        if len(paths) == 0:
            self.messageBox("Ploting time stream","Select at least one KID to plot","warning")    
            self.ui.setEnabled(True)
            return

        create_Tree = False

        if self.flagResize == True:
            self.resizeTree(self.flagResize)
            create_Tree = True

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        num = 0
        tree = self.ui.treeIQ
        tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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

            """
            print "ON"
            print "++++"
            print len(i_HR_ON),len(q_HR_ON)
            print "++++"
            """

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

            """
            print "OFF"
            print "++++"
            print len(i_HR_OFF),len(q_HR_OFF)
            print "++++"
            """

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
        except Exception as e:
            pass

        num = 0
        tree = self.ui.treeIQ
        tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        if len(paths) == 0:
            self.messageBox("Ploting time stream","Select at least one KID to plot","warning")    
            self.ui.setEnabled(True)
            return

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        for path in paths:
            try:
                exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path, shortName[num], headPath[num], "all")                    
            except Exception as e:
                self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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

            """
            print "0N"
            print "++++"
            print len(i_LR_ON),len(q_LR_ON)
            print "++++"
            """

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

            """
            print "OFF"
            print "++++"
            print len(i_LR_OFF),len(q_LR_OFF)
            print "++++"
            """

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
        mag_vna = 20*np.log10(mag_vna)

        z = self.f1.plot(freq,mag_vna,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        c2 = z[0].get_color()

        if self.ui.actionHR.isChecked():
            mag_hr = np.sqrt((sweep_i_hr**2)+(sweep_q_hr**2))
            mag_hr = 20*np.log10(mag_hr)
            self.f1.plot(freq_hr,mag_hr,color=c2,dashes=[12, 2],marker="o")

        if self.ui.actionQ_Factor.isChecked():
            
            if self.ui.nonlinearBtn.isChecked() or self.ui.actionNonlinearity.isChecked():
                fit_fil = fit[0]
            else:
                fit_fil = fit[1]

            fit_curve = fit_fil[2]
            mag_index = np.argmin(np.abs(freq - (fit_fil[0][0])))

            self.f1.plot(freq, fit_curve,color=c2,dashes=[6, 2])
            self.f1.plot(fit_fil[0][0],fit_curve[mag_index],"yo")
            self.f1.axvline(fit_fil[0][0],color='g',linewidth=0.75)

            q_msg = r"$"+str(fit_fil[0][0]/1e6)+"MHz$ \n$Q="+str(fit_fil[0][1])+"$"

            if self.ui.actionGet_Qi.isChecked():
                q_msg = q_msg + "\n$Q_i="+str(fit_fil[0][2])+"$" 

            if self.ui.actionGet_Qc.isChecked():
                q_msg = q_msg + "\n$Q_c="+str(fit_fil[0][3])+"$" 

            self.f1.annotate(q_msg,xy=(fit_fil[0][0],fit_curve[mag_index]))

        if self.ui.actionFindResonance.isChecked() or f0_leg == True:
            self.f1.plot([f0_meas, f0_meas],[np.min(mag_vna),np.max(mag_vna)],'c-')
            self.f1.plot([f0_fits, f0_fits],[np.min(mag_vna),np.max(mag_vna)],'g--')
            self.f1.plot(f0_meas,np.min(mag_vna),'ro')
            self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(mag_vna)))
            self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(mag_vna)))

        self.f1.set_xlabel(r'$Frequency [Hz]$')
        self.f1.set_ylabel(r'$S_{21}[dB]$')

        if leg == False:
            xp = freq[cnt*(len(freq)/2)/nPlots + 1]
            yp = mag_vna[cnt*(len(mag_vna)/2)/nPlots + 1]
            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

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
        
            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

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

            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

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

            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

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

            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

        #---df---
        self.f1 = self.fig1.add_subplot(313)
        
        self.f1.plot(psd[3],psd[6],c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")
        self.f1.plot(psd_OFF[3],psd_OFF[6],c2+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        self.f1.set_xlabel(r'$Time [s]$')
        self.f1.set_ylabel(r'$df[V]$')
       
        if leg == False:
            xp = psd[3][cnt*(len(psd[3])/2)/nPlots + 1]
            yp = psd[6][cnt*(len(psd[6])/2)/nPlots + 1]

            self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)

        #Plot parameters
        #Legend
        if self.ui.squareLeg.isChecked() or leg == True:
            self.f1.legend(loc='best')

    # --- Plot Noise for ON/OFF resonance frequency
    def plotNoise(self, psd, psd_OFF, psd_low, psd_low_OFF, namePlot, cnt, plotPar, ctrl, clr):
        c1, c2, alpha, alpha_on, p = plotPar

        self.f1 = self.fig1.add_subplot(111)

        # AQUI ESTA EL PROBLEMA DEL RECORTE ***************************************************
        ind_cut_freq = np.where(psd[0]>100.)[0][0]
        print ind_cut_freq


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
        mag = 20*np.log10(mag)

        z = self.f1.plot(freq,mag,c1+p+'-',alpha=alpha,label=r"$"+namePlot[cnt]+"$")

        if self.ui.actionQ_Factor.isChecked():

            if self.ui.nonlinearBtn.isChecked() or self.ui.actionNonlinearity.isChecked():
                fit_fil = fit[0]
            else:
                fit_fil = fit[1]

            fit_curve = fit_fil[2]
            mag_index = np.argmin(np.abs(freq - (fit_fil[0][0])))

            c2 = z[0].get_color()
            self.f1.plot(freq, fit_curve,color=c2,dashes=[6, 2])
            self.f1.axvline(fit_fil[0][0],color='g',linewidth=0.75)
            self.f1.plot(fit_fil[0][0],fit_curve[mag_index],"yo")

            q_msg = r"$"+str(fit_fil[0][0]/1e6)+"MHz$ \n$Q="+str(fit_fil[0][1])+"$"

            if self.ui.actionGet_Qi.isChecked():
                q_msg = q_msg + "\n$Q_i="+str(fit_fil[0][2])+"$" 

            if self.ui.actionGet_Qc.isChecked():
                q_msg = q_msg + "\n$Q_c="+str(fit_fil[0][3])+"$" 

            self.f1.annotate(q_msg,xy=(fit_fil[0][0],fit_curve[mag_index]))

        if self.ui.actionFindResonance.isChecked() or f0_leg == True:
            self.f1.plot([f0_meas, f0_meas],[np.min(mag),np.max(mag)],'c-')
            self.f1.plot([f0_fits, f0_fits],[np.min(mag),np.max(mag)],'g--')
            self.f1.plot(f0_meas,np.min(mag),'ro')
            self.f1.annotate(r"$"+str(f0_meas/1e6)+"MHz$",xy=(f0_meas,np.min(mag)))
            self.f1.annotate(r"$"+str(f0_fits/1e6)+"MHz$",xy=(f0_fits,np.min(mag)))

        self.f1.set_xlabel(r'$Frequency [Hz]$')
        self.f1.set_ylabel(r'$S_{21}[dB]$')

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
        self.fnote_ann = []

        if self.ui.actionVNA_Plots.isChecked() or self.ui.actionHomodynePlots.isChecked():

            paths, shortName, namePlot, headPath = self.kidSelected()

            if len(paths) == 0:
                self.messageBox("Ploting data","Select at least one KID to plot","warning")    
                self.ui.setEnabled(True)
                return

            try:
                self.fig1.clf()
            except Exception as e:
                pass

            cnt = 0 
            for path in paths:
                try:
                    exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(path,shortName[cnt], headPath[cnt], "all")
                except Exception as e:
                    self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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
                    self.messageBox("Ploting data","Select at least one KID to plot","warning")    
                    self.ui.setEnabled(True)
                    return

                try:
                    self.fig1.clf()
                except Exception as e:
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
                        self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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

                            ind_cut_freq = np.where(psd[0]>100.)[0][0]
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

                            if self.ui.nonlinearBtn.isChecked() or self.ui.actionNonlinearity.isChecked():
                                fit_fil = fit[0]
                            else:
                                fit_fil = fit[1]

                            fit_curve = fit_fil[2]
                            mag_index = np.argmin(np.abs(freq - (fit_fil[0][0])))

                            if self.ui.ylog.isChecked():
                                self.ui.ylog.setChecked(True)
                                mag = 20*np.log10(mag)
                                fit_curve = 20*np.log10(fit_fil[2])
                                self.f1.set_ylabel(r'$S_{21}[dB]$')                           

                            z = self.f1.plot(freq,mag, c1+p+'-', alpha=alpha, label=r"$"+namePlot[cnt]+"$")
                            c3 = z[0].get_color()

                            if self.ui.actionHR.isChecked():
                                mag_hr = np.sqrt((sweep_i_hr**2)+(sweep_q_hr**2))
                                if self.ui.ylog.isChecked():
                                    mag_hr = 20*np.log10(mag_hr)
                                self.f1.plot(freq_hr,mag_hr,color=c3,dashes=[12, 2],marker="o")
                            
                            if self.ui.actionQ_Factor.isChecked():
                                self.f1.plot(freq, fit_curve,color=c3,dashes=[6, 2])
                                self.f1.plot(fit_fil[0][0],fit_curve[mag_index],"yo")
                                self.f1.axvline(fit_fil[0][0],color='g',linewidth=0.75)

                                q_msg = r"$"+str(fit_fil[0][0]/1e6)+"MHz$ \n$Q="+str(fit_fil[0][1])+"$"

                                if self.ui.actionGet_Qi.isChecked():
                                    q_msg = q_msg + "\n$Q_i="+str(fit_fil[0][2])+"$" 

                                if self.ui.actionGet_Qc.isChecked():
                                    q_msg = q_msg + "\n$Q_c="+str(fit_fil[0][3])+"$" 

                                self.f1.annotate(q_msg,xy=(fit_fil[0][0],fit_curve[mag_index]))

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

                        self.legend_curve = self.f1.annotate(r"$"+namePlot[cnt]+"$",xy=(xp,yp))
                        if self.ui.plotLeg.isChecked():
                            self.legend_curve.set_visible(True)
                        else:
                            self.legend_curve.set_visible(False)                            

                    cnt += 1
                    #self.fig1.tight_layout()
                    self.f1.figure.canvas.draw()
                
                self.ui.statusbar.showMessage("KID data has been loaded")
            else:
                self.messageBox("Ploting","Select a graph to plot","warning")
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
            self.messageBox("Searching KIDs","KID files not found!","warning")
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
        except Exception as e:
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

        except Exception as e:
            pass

        return KIDS, allPaths, allNames, allHeads, allShortNames

    # --- Functions to load all files in one or some directories
    def loadThread(self):
        if len(self.allPaths) == 0:
            self.messageBox("Loading files","Directory not selected","warning")
            self.ui.setEnabled(True)
            return
        try:
            i = 0 
            for path in self.allPaths:
                self.loadData(str(path), str(self.allNames[i]), str(self.allHeads[i]), "all", False)
                i += 1
        except Exception as e:
            self.messageBox("Error","Error trying to open files, maybe files are missing." + str(e),"error")
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

            mag_cplx = sweep_i + 1j*sweep_q 
            fit_nonlin = self.getQFactor(freq,mag_cplx,nonlinear=True)
            fit_lin = self.getQFactor(freq,mag_cplx,nonlinear=False)

            fit = [fit_nonlin, fit_lin]

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
            self.messageBox("MAP","To apply MAP algorithm, two data signal are needed","warning")
            return

    # --- Generate all the figures and save them
    def saveFigures(self, event):
        print "Create the Directories"
        
        self.ui.setEnabled(False)
        self.messageBox("Creating figures","The program will create the figures for all the files, it would takes several minutes.","info")

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
                        except Exception as e:
                            pass

                        try:
                            exists, freq, sweep_i, sweep_q, freq_hr, sweep_i_hr, sweep_q_hr, psd, psd_low, psd_OFF, psd_low_OFF, f0_meas, f0_fits, fit = self.loadData(str(path[num]),self.allNames[num], self.allHeads[num], "all")                    
                        except Exception as e:
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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
                            self.messageBox("Error","Error trying to open files, maybe files are missing\n"+str(e),"error")    
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

        if self.ui.nonlinearBtn.isChecked() or self.ui.actionNonlinearity.isChecked():
            fit = "Q = " + str(fit[0][0][1]) + "\nF0 = " + str(fit[0][0][0]/1e6) + "MHz\nQ_i = " + str(fit[0][0][2]) + "\nQ_c = " + str(fit[0][0][3])
        else:
            fit = "Q = " + str(fit[1][0][1]) + "\nF0 = " + str(fit[1][0][0]/1e6) + "MHz\nQ_i = " + str(fit[1][0][2]) + "\nQ_c = " + str(fit[1][0][3])

        self.ui.statusText.append("**********")
        self.ui.statusText.append(data)
        self.ui.statusText.append(fit)
        self.ui.statusText.append("**********")

    def getQFactor(self, freq, mag, nonlinear):

        #   potp has the parameters for the fit curve:
        #       f0: Resonance frequency
        #       Qr: Total Q
        #       Qc: Coupling Q 
        #       Qi: Intrinsec Q
        #       A:  Level
        #       a:  Nonlinearity

        potp, perr, fit_curve = self.fit_S21.fitmags21(freq,mag,nonlinear=nonlinear)

        return potp, perr, fit_curve

    # --- Message window
    def messageBox(self, title, msg, mode):
        w = QWidget()
        if mode == "warning":
            QMessageBox.warning(w, title, msg)
        elif mode == "error":
            QMessageBox.critical(w, title, msg)
        elif mode == "info":
            QMessageBox.information(w, title, msg)

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

    def drawLine(self, event):
        if self.ui.eraseBtn.isChecked():
            self.ui.eraseBtn.setChecked(False)

        if self.ui.drawBtn.isChecked():
            self.ui.drawBtn.setChecked(False)
        else:
            self.ui.drawBtn.setChecked(True)            

    def eraseLine(self, event):
        if self.ui.drawBtn.isChecked():
            self.ui.drawBtn.setChecked(False)

        if self.ui.eraseBtn.isChecked():
            self.ui.eraseBtn.setChecked(False)
        else:
            self.ui.eraseBtn.setChecked(True)   

    def nonlinear_active_bar(self, event):
        if self.ui.actionNonlinearity.isChecked():
            self.ui.nonlinearBtn.setChecked(True)
        else:
            self.ui.nonlinearBtn.setChecked(False)

    def nonlinear_active_btn(self, event):
        if self.ui.nonlinearBtn.isChecked():
            self.ui.actionNonlinearity.setChecked(True)
        else:
            self.ui.actionNonlinearity.setChecked(False)

    def fnote_active_btn(self, event):

        if len(self.fnote_ann)>0:
            for m in range(len(self.ind)):
                for i in range(len(self.ind[m])):
                    i = int(i)

                    if self.ui.fnoteBtn.isChecked():
                        self.fnote_ann[m][i].set_visible(True)
                    else:
                        self.fnote_ann[m][i].set_visible(False)

            self.f1.figure.canvas.draw()

    def legend_btn(self, event):
        if self.f1.legend_:
            self.f1.legend_.remove()

        if self.ui.squareLeg.isChecked():
            self.f1.legend()

        if self.legend_curve:
            if self.ui.plotLeg.isChecked():
                self.legend_curve.set_visible(True)
            else:
                self.legend_curve.set_visible(False)
        
        self.f1.figure.canvas.draw()        

    def addmplMap(self,fig):
        self.canvas = FigureCanvas(fig)
        self.Qmap.MainPlot.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
           self, coordinates=True)
        self.Qmap.MainPlot.addWidget(self.toolbar)

    def addmpl(self,fig):
        self.canvas = FigureCanvas(fig)
        self.ui.MainPlot.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
           self, coordinates=True)
        self.ui.MainPlot.addWidget(self.toolbar)

    def addmplPlotMap(self,fig):
        self.canvas = FigureCanvas(fig)
        self.Plots.MainPlot.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
           self, coordinates=True)
        self.Plots.MainPlot.addWidget(self.toolbar)

    def rmmpl(self):
        self.ui.MainPlot.removeWidget(self.canvas)
        self.canvas.close()
        self.ui.MainPlot.removeWidget(self.toolbar)
        self.toolbar.close()

    def about(self,event):
        self.messageBox("About KID-ANALYSER","KID Analyser V 1.0 by INAOE KID team, Marcial Becerril, Salvador Ventura and Edgar Castillo with the support of Sam Rowe, Simon Doyle and Pete Barry :), thanks a lot for the valuable help!","info")

    # Stadistics GUI
    def Q_StadSet(self,event):

        self.stad = uic.loadUi("gui/stad.ui")

        screen = QtGui.QDesktopWidget().screenGeometry() 

        size_x = screen.width()
        size_y = screen.height()

        self.stad.move(self.size_x/2-280/2,self.size_y/2-250/2)

        # PDF Button
        self.stad.PDFBtn.clicked.connect(self.pdf_active)
        # PDA Button
        self.stad.PDABtn.clicked.connect(self.pda_active)
        # Apply button
        self.stad.applyBtn.mousePressEvent = self.apply_stad

        self.stad.show()

    def pdf_active(self,event):
        if self.stad.PDABtn.isChecked():
            self.stad.PDABtn.setChecked(False)

        self.stad.PDFBtn.setChecked(True)

    def pda_active(self,event):
        if self.stad.PDFBtn.isChecked():
            self.stad.PDFBtn.setChecked(False)

        self.stad.PDABtn.setChecked(True)

    def apply_stad(self,event):
        # Get parameters
        nbins = self.stad.nbinsBox.value()
        
        if self.stad.PDFBtn.isChecked():
            self.pdfFlag = True
        else:
            self.pdfFlag = False

        if self.stad.PDABtn.isChecked():
            self.pdaFlag = True
        else:
            self.pdaFlag = False

        if self.stad.drawCurveCB.isChecked():
            self.drawCurve = True
        else:
            self.drawCurve = False

        if self.stad.writeLegBtn.isChecked():
            self.writeLeg = True
        else:
            self.writeLeg = False  

        if str(self.stad.PDFbox.currentText()) == "Gamma":
            self.pdfTye = "Gamma"
        elif str(self.stad.PDFbox.currentText()) == "Gaussian":      
            self.pdfType = "Gaussian"

        self.VNA_stad_curves()

    def filter_analysis(self,event):
        if len(self.acumQ) > 0:
            self.Qfilter = uic.loadUi("gui/filter.ui")

            screen = QtGui.QDesktopWidget().screenGeometry() 

            size_x = screen.width()
            size_y = screen.height()

            self.Qfilter.move(self.size_x/2-429/2,self.size_y/2-440/2)

            # Filter list
            self.Qfilter.filterBtn.mousePressEvent = self.prevFilter

            # Apply Filter
            self.Qfilter.applyBtn.mousePressEvent = self.applyFilter

            # Update list
            self.Qfilter.updateListBtn.mousePressEvent = self.updateList

            model = QtGui.QStandardItemModel()
            self.Qfilter.listQ.setModel(model)

            self.Qfilter.listQ.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

            # Dummy Q array
            """
            self.acumQ = [[450e6,15000,30000,20000,0.5,0.2],[510e6,18000,25000,15000,0.2,0.1],
                         [550e6,17000,32000,22000,0.1,0.15],[650e6,17500,22000,12000,0.1,0.15],
                         [520e6,17000,32000,22000,0.1,0.15],[750e6,17500,22000,12000,0.1,0.15],
                         [550e6,17000,32000,22000,0.1,0.15],[670e6,17500,22000,12000,0.1,0.15],
                         [560e6,17000,32000,22000,0.1,0.15],[620e6,17500,22000,12000,0.1,0.15],
                         [570e6,17000,32000,22000,0.1,0.15],[622e6,17500,22000,12000,0.1,0.15],
                         [588e6,17000,32000,22000,0.1,0.15],[651e6,17500,22000,12000,0.1,0.15],
                         [551e6,17000,32000,22000,0.1,0.15],[610e6,17500,22000,12000,0.1,0.15],
                         [554e6,17000,32000,22000,0.1,0.15],[623e6,17500,22000,12000,0.1,0.15],
                         [500e6,17000,32000,22000,0.1,0.15],[750e6,17000,32000,22000,0.1,0.15]]
            
            self.ind = [[450e6,510e6,450e6,450e6,450e6,450e6,450e6,450e6,450e6,450e6,450e6,
                        450e6,450e6,450e6,450e6,450e6,450e6,450e6,450e6,450e6]]
            """

            self.Qfilter.arrayBox.setMaximum(len(self.ind))
            m = self.Qfilter.arrayBox.value()-1

            self.auxInd = self.ind
            self.auxQ = self.acumQ

            self.spareInd = self.ind
            self.spareQ = self.acumQ

            start = 0
            for n in range(len(self.ind)):
                if n == m:
                    break
                else:
                    start = start + len(self.ind[n])

            for i in range(len(self.ind[m])):
                txt = "KID " + str(i) + " " + str(self.acumQ[start+i][0])
                item = QtGui.QStandardItem(txt)
                model.appendRow(item)

            self.Qfilter.show()
        else:
            self.messageBox("Starting filter","Find resonators and fit the Q curves before","warning")

    def updateList(self,event):

        model = QtGui.QStandardItemModel()
        self.Qfilter.listQ.setModel(model)

        m = self.Qfilter.arrayBox.value()-1

        start = 0
        for n in range(len(self.auxInd)):
            if n == m:
                break
            else:
                start = start + len(self.auxInd[n])

        for i in range(len(self.auxInd[m])):
            txt = "KID " + str(i) + " " + str(self.auxQ[start+i][0])
            item = QtGui.QStandardItem(txt)
            model.appendRow(item)

    def prevFilter(self,event):

        m = self.Qfilter.arrayBox.value()-1

        if self.Qfilter.manualBtn.isChecked():
            itms = self.Qfilter.listQ.selectedIndexes()
            untilList = []
            for data in itms:
                txt = data.data().toString()
                until = ""
                for n in range(4,len(txt)):
                    if txt[n] == " ":
                        break
                    else:
                        until = until + txt[n]
                untilList.append(int(until))
            untilList.sort(reverse=True)

            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            for n in untilList:
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        if self.Qfilter.qBtn.isChecked():
            minQ = self.Qfilter.minQ.value()
            maxQ = self.Qfilter.maxQ.value()
            
            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            untilList = []
            for i in range(len(self.auxInd[m])):
                if self.auxQ[start+i][1] < minQ or self.auxQ[start+i][1] > maxQ:
                    untilList.append(i)

            untilList.sort(reverse=True)

            for n in untilList: 
                #print self.auxQ[m*len(self.auxInd[m])-i-1][1]
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        if self.Qfilter.qiBtn.isChecked():
            minQi = self.Qfilter.minQi.value()
            maxQi = self.Qfilter.maxQi.value()

            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            untilList = []
            for i in range(len(self.auxInd[m])):
                if self.auxQ[start+i][3] < minQi or self.auxQ[start+i][3] > maxQi:
                    untilList.append(i)

            untilList.sort(reverse=True)

            for n in untilList: 
                #print self.auxQ[m*len(self.auxInd[m])-i-1][1]
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        if self.Qfilter.qcBtn.isChecked():
            minQc = self.Qfilter.minQc.value()
            maxQc = self.Qfilter.maxQc.value()

            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            untilList = []
            for i in range(len(self.auxInd[m])):
                if self.auxQ[start+i][2] < minQc or self.auxQ[start+i][2] > maxQc:
                    untilList.append(i)

            untilList.sort(reverse=True)

            for n in untilList: 
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        if self.Qfilter.nonBtn.isChecked():
            minNon = self.Qfilter.minNon.value()
            maxNon = self.Qfilter.maxNon.value()

            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            untilList = []
            for i in range(len(self.auxInd[m])):
                if self.auxQ[start+i][5] < minNon or self.auxQ[start+i][5] > maxNon:
                    untilList.append(i)

            untilList.sort(reverse=True)

            for n in untilList: 
                #print self.auxQ[m*len(self.auxInd[m])-i-1][1]
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        if self.Qfilter.posBtn.isChecked():
            minFreq = self.Qfilter.minFreq.value()
            maxFreq = self.Qfilter.maxFreq.value()
            
            start = 0
            for n in range(len(self.auxInd)):
                if n == m:
                    break
                else:
                    start = start + len(self.auxInd[n])

            untilList = []
            for i in range(len(self.auxInd[m])):
                if self.auxQ[start+i][0] < minFreq or self.auxQ[start+i][0] > maxFreq:
                    untilList.append(i)

            untilList.sort(reverse=True)

            for n in untilList: 
                #print self.auxQ[m*len(self.auxInd[m])-i-1][1]
                self.auxInd[m] = np.delete(self.auxInd[m],n)
                del self.auxQ[start+n]

        model = QtGui.QStandardItemModel()
        self.Qfilter.listQ.setModel(model)

        start = 0
        for n in range(len(self.auxInd)):
            if n == m:
                break
            else:
                start = start + len(self.auxInd[n])

        for i in range(len(self.auxInd[m])):
            txt = "KID " + str(i) + " " + str(self.auxQ[start+i][0])
            item = QtGui.QStandardItem(txt)
            model.appendRow(item)

    def applyFilter(self,event):
        self.ind = self.auxInd
        self.acumQ = self.auxQ

    # Reset the filter changes
    def reset_values(self,event):
        try:
            self.ind = self.spareInd
            self.acumQ = self.spareQ
        except Exception as e:
            pass

    # Values to get Delta function   
    def delta_f0(self, event):
        if len(self.ind) == 2:

            self.Qdelta = uic.loadUi("gui/delta.ui")

            screen = QtGui.QDesktopWidget().screenGeometry() 

            size_x = screen.width()
            size_y = screen.height()

            self.Qdelta.move(self.size_x/2-232/2,self.size_y/2-120/2)

            # Start Delta
            self.Qdelta.okDelta.mousePressEvent = self.plotDelta

            arrays = []
            for i in range(len(self.ind)):
                arrays.append("Array " + str(i+1))

            self.Qdelta.array1Box.clear()
            self.Qdelta.array1Box.addItems(arrays)

            self.Qdelta.array2Box.clear()
            self.Qdelta.array2Box.addItems(arrays)

            self.Qdelta.show()

        else:
            self.messageBox("Warning","To run this option you need two arrays to compare each other.","error") 

    def plotDelta(self,event):
        array1 = self.Qdelta.array1Box.currentIndex()
        array2 = self.Qdelta.array2Box.currentIndex()

        # Check if the arrays have the same length
        if len(self.ind[array1]) != len(self.ind[array2]):
            self.messageBox("Warning","The arrays have not the same length","error")
            return 
        else:
            f0 = []
            delta_f0 = []

            for i in range(len(self.ind[array1])):
                delta_f0.append(self.freq_Q[array2][self.ind[array2][i]] - self.freq_Q[array1][self.ind[array1][i]])
                f0.append(self.freq_Q[array1][self.ind[array1][i]])

            try:
                self.fig1.clf()
            except Exception as e:
                pass

            self.f1 = self.fig1.add_subplot(111)
            self.f1.plot(f0,delta_f0,'o')
            self.f1.set_title(r"$Delta$")
            self.f1.set_xlabel(r"$f_0[Hz]$")
            self.f1.set_ylabel(r"$Shift [Hz]$")

            self.f1.figure.canvas.draw()

    def save_var(self,event):
        u = 0
        for i in range(len(self.pathVNA)):
            for j in range(len(self.pathVNA[i])):
                if str(self.pathVNA[i][-j-1]) == "/":
                    break
            path = str(self.pathVNA[i][:-j])
            
            # Save VNA
            newPath = path + "Results/"
            os.system('mkdir ' + newPath)

            #VNA
            try:
                np.save(newPath + "I_" + str(i),self.I_vna[i])
                np.save(newPath + "Q_" + str(i),self.Q_vna[i])
                np.save(newPath + "Freq_" + str(i),self.freq_Q[i])
            except:
                pass

            #INDEX
            try:
                np.save(newPath + "Index_" + str(i),self.ind[i])
            except:
                pass

            #Q_PARAM
            try:
                params = []
                for k in range(len(self.ind[i])):
                    params.append(self.acumQ[u + k]  )
                np.save(newPath + "Qparams_" + str(i),params)  
                u = u + k        
            except:
                pass

    def tempPlot(self,event):

        number = len(self.ind)
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, number)]

        try:
            self.fig1.clf()
        except Exception as e:
            pass

        qFlags = [self.ui.actionQ_Factor.isChecked(), self.ui.actionGet_Qc.isChecked(), self.ui.actionGet_Qi.isChecked(), self.ui.actionNonlinearity.isChecked()]

        nQtags = 0
        for nq in qFlags:
            if nq:
                nQtags += 1

        # Get the name of the data
        arrayName = []
        for i in range(len(self.pathVNA)):
            name = ""
            for j in range(len(self.pathVNA[i])):
                if self.pathVNA[i][-j-1] == "/":
                    break
                name = self.pathVNA[i][-j-1:]
            arrayName.append(name)

        m = 0
        i = 0
        for point in self.acumQ:
            f0 = point[0]
            Qr = point[1]
            Qc = point[2]
            Qi = point[3]
            a = point[5]

            cnt = 0 
            for k in range(4):

                if qFlags[k]:
                    cnt += 1

                if k == 0 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_r$')
                    if i == 0:
                        self.f1.plot(f0,Qr,'o',color=colors[m],label=arrayName[m])
                        if self.ui.squareLeg.isChecked():
                            self.f1.legend(loc='best')
                    else:
                        self.f1.plot(f0,Qr,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qr))

                if k == 1 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_c$')
                    if i == 0:
                        self.f1.plot(f0,Qc,'o',color=colors[m],label=arrayName[m])
                        if self.ui.squareLeg.isChecked():
                            self.f1.legend(loc='best')
                    else:
                        self.f1.plot(f0,Qc,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qc))

                if k == 2 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$N$')
                    self.f1.set_title(r'$Q_i$')
                    if i == 0:
                        self.f1.plot(f0,Qi,'o',color=colors[m],label=arrayName[m])
                        if self.ui.squareLeg.isChecked():
                            self.f1.legend(loc='best')
                    else:
                        self.f1.plot(f0,Qi,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,Qi))

                if k == 3 and qFlags[k]:
                    self.f1 = self.fig1.add_subplot(nQtags,1,cnt)
                    self.f1.set_ylabel(r'$Value$')
                    self.f1.set_title(r'$Nonlinearity$')
                    if i == 0:
                        self.f1.plot(f0,a,'o',color=colors[m],label=arrayName[m])
                        if self.ui.squareLeg.isChecked():
                            self.f1.legend(loc='best')
                    else:
                        self.f1.plot(f0,a,'o',color=colors[m])
                    if self.ui.actionName_KID.isChecked():
                        self.f1.annotate(r"$KID "+str(m)+"$",xy=(f0,a))
            
            self.f1.set_xlabel(r'$f_0$')

            if i == len(self.ind[m]):
                i = 0
                m = m + 1
            else:
                i = i + 1

        self.f1.figure.canvas.draw()

    # Values to get Delta function   
    def respo(self, event):
        if len(self.ind) == 2:

            self.Qrespo = uic.loadUi("gui/respo.ui")

            screen = QtGui.QDesktopWidget().screenGeometry() 

            size_x = screen.width()
            size_y = screen.height()

            self.Qrespo.move(self.size_x/2-225/2,self.size_y/2-120/2)

            # Start Resposibility
            self.Qrespo.okRespo.mousePressEvent = self.plotRespo

            arrays = []
            for i in range(len(self.ind)):
                arrays.append("Array " + str(i+1))

            self.Qrespo.array1Box.clear()
            self.Qrespo.array1Box.addItems(arrays)

            self.Qrespo.array2Box.clear()
            self.Qrespo.array2Box.addItems(arrays)

            self.Qrespo.show()

        else:
            self.messageBox("Error ploting responsibility","To run this option you need two arrays to get the responsibility.","error") 

    def plotRespo(self,event):
        array1 = self.Qrespo.array1Box.currentIndex()
        array2 = self.Qrespo.array2Box.currentIndex()

        # Check if the arrays have the same length
        if len(self.ind[array1]) != len(self.ind[array2]):
            self.messageBox("Error ploting responsibility","The arrays have not the same number of resonators","error")
            return 
        else:
            delta_f0 = 0.
            ds21 = 0.

            mag_vna_1 = 0.
            mag_vna_2 = 0.

            Respo = []
            f0 = []    
            q = []

            # Responsibility
            start = 0
            for n in range(len(self.ind)):
                if n == array1:
                    break
                else:
                    start = start + len(self.ind[n])

            for i in range(len(self.ind[array1])):
                mag_vna_1 = np.sqrt(self.I_vna[array1][self.ind[array1][i]]**2 + self.Q_vna[array1][self.ind[array1][i]]**2)
                mag_vna_2 = np.sqrt(self.I_vna[array2][self.ind[array2][i]]**2 + self.Q_vna[array2][self.ind[array2][i]]**2)

                ds21 = mag_vna_2 - mag_vna_1
                delta_f0 = self.freq_Q[array2][self.ind[array2][i]] - self.freq_Q[array1][self.ind[array1][i]]
                
                # Responsibility
                Respo.append(ds21/delta_f0)
                f0.append(self.freq_Q[array1][self.ind[array1][i]])

                # Q factors
                q.append(self.acumQ[start+i][1])

            try:
                self.fig1.clf()
            except Exception as e:
                pass

            self.f1 = self.fig1.add_subplot(111)
            self.f1.plot(f0,Respo,'ro')
            self.f1.set_title(r"$Responsibility$")
            self.f1.set_xlabel(r"$f_0[Hz]$")
            self.f1.set_ylabel(r"$dS_{21}/df_0 [Hz]$")

            self.f1.figure.canvas.draw()

    def writeTones(self,event):
        if len(self.ind) > 0:
            m = self.ui.editSweepBox.value()-1

            w = QWidget()
            w.resize(320, 240)    
            w.setWindowTitle("Select directory to save the Tone List")

            fileName = QFileDialog.getSaveFileName(self, 'Save Tone List', './', selectedFilter='*.txt')
            file = open(fileName, "w")

            # Write Tone List
            file.write("Name\tFreq\tOffset\tatt\tAll\tNone\n")

            for i in range(len(self.ind[m])):
                file.write("K"+'{:03d}'.format(i)+"\t"+str(self.freq_Q[m][self.ind[m][i]])+"\t"+"0"+"\t"+"1"+"\t"+"0\n")
            file.close()
        else:
            self.messageBox("Writing tones","Look for tones before write the tone list.","error") 

#Ejecución del programa
app = QtGui.QApplication(sys.argv)
MyWindow = MainWindow()
sys.exit(app.exec_())