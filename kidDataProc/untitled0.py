# -*- coding: utf-8 -*-
"""
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# === IMPORTATION === #

import sys  
sys.path.insert(0,'D:\Documents\Work\Cardiff University\Laboratory\KID-DataProc')
sys.path.insert(0,'D:\Documents\Work\Cardiff University\Laboratory\KID-DataProc\lib')
import os
import fnmatch
from astropy.io import fits

import import_data
#

# === LISTING DATA === #

os.chdir('D:\Documents\Work\Cardiff University\MUSCAT\Data') # Change directory to the data one.

listOfFiles = [] # array of string containing the filenames in the data directory
listOfDataFiles = [] # array of string containing the data filenames

for root, dirs, files in os.walk("."):  # for loop to fill the lisOfFiles
    for filename in files:
       listOfFiles.append(filename)

pattern = "*.fits"  # for loop to match the listOfDataFiles with actual data files
for entry in listOfFiles:  
    if fnmatch.fnmatch(entry, pattern):
        listOfDataFiles.append(entry)
#

# === LISTING HEADERS === #
hdu = []
data = []
for entry in listOfDataFiles:
    fitsData = import_data.importData(entry) # DataFile importation
    h, d = fitsData.extractHDU() # Reading header of fits file
    hdu.append(h)
    data.append(d)





















