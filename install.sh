#!/bin/bash

# Color print Constants
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'

UNDERLINE='\033[4m'

END='\033[0m' # No Color

echo -e "${UNDERLINE}******KID Analyzer v0.2******${END}"
echo "Development by Marcial Becerril"
echo -e "${YELLOW}Installing all the packages ...${END}"

# Installing scipy
echo -e "${YELLOW}Checking scipy...${END}"

if python -c "import scipy" &> /dev/null; then
    echo -e "${GREEN}scipy already installed${END}\n"
else
    echo -e "${BLUE}Installing scipy${END}\n"
    pip install scipy
    echo -e "${GREEN}scipy installed${END}\n"
fi

# Installing numpy
echo -e "${YELLOW}Checking numpy...${END}"

if python -c "import numpy" &> /dev/null; then
    echo -e "${GREEN}numpy already installed${END}\n"
else
    echo -e "${BLUE}Installing numpy${END}\n"
    pip install numpy
    echo -e "${GREEN}numpy installed${END}\n"
fi

# Installing PyQT4
echo -e "${YELLOW}Checking pyqt4...${END}"

if python -c "import PyQt4" &> /dev/null; then
    echo -e "${GREEN}PyQt4 already installed${END}\n"
else
    echo -e "${BLUE}Installing PyQt4${END}\n"
    apt-get install python-qt4
    echo -e "${GREEN}PyQt4 installed${END}\n"
fi

# Install astropy version 2.0.11.
# This is the last version of astropy that runs for python 2.7
echo -e "${BLUE}Installing Astropy 2.0.11${END}\n"
pip install astropy==2.0.11
echo -e "${GREEN}astropy installed${END}\n"
