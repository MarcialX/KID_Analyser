# KID Analyser

This is a repository for the development of reduction programs for the characterization of the Kinetic Inductance Detectors from the output of the current LABVIEW adquistion rack in the mm-wavelength laboratory of the University of Cardiff for the MUSCAT project.

### Prerequisites

KID Analyser code runs under Python 2.7 or less. The Python libraries need it are:

* Python-QT4. The last version of PyQT4. In Linux it could be installed with:
```
sudo apt-get install python-qt4
```
* Astropy version 2.0.11. To read FITS files. The version 2.0.11 is the last able to run with Python 2.7. Install it with pip:
```
pip install astropy==2.0.11
```
* Numpy and Scipy.

### Installation Dependencies

In order to run the KIDs data reduction program, you need to install one by one the dependencies or run the install.sh as super user

```
chmod +x install.sh
sudo ./install.sh
```
## Run

Run the index.py, it contains the base of the whole
```
python index.py
```
