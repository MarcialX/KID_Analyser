#!/usr/bin/env python
# -*- coding: utf-8 -*-
#************************************************************
#*                    KID-ANALYSER                          *
#*                        INAOE                             *
#*                   Marcial Becerril                       *
#*                 Thread to Load Data                      *
#*                    27/julio/2018                         *
#************************************************************

import os

from threading import Thread
from PyQt4 import QtCore

#Enable/Disable Window
class loadThread(QtCore.QThread):
    
    flagLoad = QtCore.pyqtSignal()
    
    def __init__(self):
        QtCore.QThread.__init__(self, None)

    def run(self):
        self.flagLoad.emit()