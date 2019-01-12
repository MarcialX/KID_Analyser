import numpy as np
from matplotlib import pyplot as plt
from numpy import fft
import math
import glob
import os
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy import signal
import sys

## Making a list of Attenuation directories given the upper directory (example: /20180623_Dark_Data_Auto)
def list_files(startpath,fits=True):
    g = open('directory.txt','w')
    paths = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level ==3:
            #print root
            paths.append(root)

            g.write(str(root)+'\n')
    g.close()     
    return np.array(paths)


# Getting the list of paths to the fits files for a specific Attenuation folder. 
def get_fits(path):
     
    get_name_fits = glob.glob(path+'/*.fits')
        
    return np.sort(get_name_fits)

# Getting the arrays  from the fits file in the path given
def get_arrays_fits(path):
    filename_fits =  get_pkg_data_filename(path)
    hdul = fits.open(filename_fits)
    hdr = hdul[1].header
    data_fits = fits.getdata(filename_fits, ext=0)
    return data_fits

#Estimating the df/dt, the psd and the average for a fits file
def df(path):
    filename_fits =  get_pkg_data_filename(path)
    hdul = fits.open(filename_fits)
    hdr = hdul[1].header
    data = hdul[1].data
    
    # print hdr
    noise = get_arrays_fits(path)
    data_fits = fits.getdata(filename_fits, ext=0)
    dqdf = hdr['DQDF']
    didf = hdr['DIDF']
    Fs = hdr['SAMPLERA']
    actual_temp = hdr['SAMPLETE']
    kid_number = hdr['TONE']
    input_att = hdr['INPUTATT']
    date = hdr['DATE']
    project = hdr['PROJECT']
    
    print hdr

    print kid_number,'Temp=%s'%(actual_temp),'att= %s'%(input_att), os.path.basename(path)
    data_fits = np.array(data_fits)
    df = [[(data_fits[i][p*2]*didf+ data_fits[i][p*2+1]*dqdf)/((didf**2)+(dqdf**2))  for i in range(len(data_fits))] for p in range(len(data_fits[0])/2)]
    psd = [signal.periodogram(df[i],Fs)[1] for i in range(len(df))]
    freqs = signal.periodogram(df[0],Fs)[0]
    promedio = np.average(psd,axis=0) 
    #print "converting into complex..."
    return freqs,psd, promedio, kid_number,date,actual_temp,input_att,project

def main():
    folder = '/home/salvador/kids_meas/20nm/20180623_Dark_Data_Auto'
    kid_number = 'KID_K003'
    temp = 'Set_Temperature_120_mK'
    att = 'Set_Attenuation_62dB'
    
    directorio =  os.path.join(folder,kid_number,temp,att)
    print directorio 
    paths=get_fits(os.path.join(folder,kid_number,temp,att))
#    print paths
 
    sweep = get_arrays_fits(paths[0])
    psd = df(paths[6])
    psd_low = df(paths[11])
    psd_OFF = df(paths[4])
    psd_low_OFF = df(paths[8])
    freq =np.zeros(len(sweep))
    sweep_i = np.zeros(len(sweep))
    sweep_q = np.zeros(len(sweep))
    
    
    for i in range(len(sweep)):
        freq[i] = sweep[i][0]
        sweep_i[i] = sweep[i][1]
        sweep_q[i] = sweep[i][2]
    
    traslape = 5 
    plt.figure()
    plt.loglog(psd[0][traslape:len(psd[0])],psd[2][traslape:len(psd[2])])  #plotting  high freq ON
    plt.loglog(psd_OFF[0][traslape:len(psd_OFF[0])],psd_OFF[2][traslape:len(psd_OFF[2])]) #plotting  high freq OFF
    #plt.xlim([len(psd[2])/2, len(psd[2])])
    #plt.xlim([0,100])
    plt.loglog(psd_low_OFF[0][3:len(psd_low_OFF[0])],psd_low_OFF[2][3:len(psd_low_OFF[2])], label='OFF') # Plotting low freq OFF
    plt.loglog(psd_low[0],psd_low[2], label='ON') # Plotting low freq ON
    plt.title('%s,%s,%s,%s,%s'%(psd[3],att, temp, psd[4],psd[5]))
    plt.legend(loc='best', fontsize=20,frameon=False)
    plt.ylim([1*10e-6,3*10e5])
    plt.savefig('/%s/%s%s%s_noise.png'%(folder,psd[3],att, temp))  #Saving noise
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(freq,np.sqrt((sweep_i**2)+(sweep_q**2)))
    plt.title('%s,%s,%s,%s'%(psd[3],att, temp, psd[4]))
    plt.subplot(2,1,2)
    plt.plot(sweep_i,sweep_q)
    plt.savefig('/%s/%s%s%s_sweep.png'%(folder,psd[3],att, temp)) #saving  sweep figure
    
    
    
    plt.show()
    
    
    




if __name__ =='__main__':
    main()


