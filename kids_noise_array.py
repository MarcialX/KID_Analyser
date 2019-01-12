import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
from kids_noise import get_fits,get_arrays_fits, list_files,df

""" This program estimates the noise for a whole array and creates a folder with noise plots """

def noise_run(start_path):
    if not os.path.exists(start_path+'/noise_plots'):
        os.makedirs(start_path+'/noise_plots')
    paths = list_files(start_path,fits=False)
    print paths
    #for i in range(1):
    
    for i in range(len(paths)):
        fits_path = get_fits(paths[i])
        print 'len fits_path  ', len(fits_path)
        if len(fits_path) == 6:
            sweep = get_arrays_fits(fits_path[0])
            sweep_HR = get_arrays_fits(fits_path[1])
            psd_high_OFF = df(fits_path[2])
            psd_high_ON = df(fits_path[3])
            psd_low_OFF = df(fits_path[4])
            psd_low_ON = df(fits_path[5])
        if len(fits_path) == 12:
            sweep = get_arrays_fits(fits_path[0+1])
            sweep_HR = get_arrays_fits(fits_path[1+1])
            psd_high_OFF = df(fits_path[2+1])
            psd_high_ON = df(fits_path[3+1])
            psd_low_OFF = df(fits_path[4+1])
            psd_low_ON = df(fits_path[5+1])
        else:   
            print 'Number of fits files not 6 or 12, may be different runs in the same directory'
        freqs_OFF = np.append(psd_low_OFF[0][2:len(psd_low_OFF[0])], psd_high_OFF[0][5:len(psd_high_OFF[0])])
        freqs_ON = np.append(psd_low_ON[0][2:len(psd_low_ON[0])], psd_high_ON[0][5:len(psd_high_ON[0])])
        psd_OFF = np.append(psd_low_OFF[2][2:len(psd_low_OFF[0])], psd_high_OFF[2][5:len(psd_high_OFF[2])])
        psd_ON = np.append(psd_low_ON[2][2:len(psd_low_ON[0])], psd_high_ON[2][5:len(psd_high_ON[2])])
        psd = psd_ON - psd_OFF
        
        plt.figure()
        plt.loglog(freqs_OFF,psd_OFF, label='%s'%(os.path.basename(fits_path[2])))
        plt.loglog(freqs_ON,psd, label='%s'%(os.path.basename(fits_path[2]+'_resta')))
        
        plt.loglog(freqs_ON,psd_ON, label='%s'%(os.path.basename(fits_path[3])))
        plt.legend(loc='lower center', fontsize=10,frameon=False)
        plt.title('%s,Temp=%s,Att=%s,%s'%(psd_high_OFF[3], psd_high_OFF[5], psd_high_OFF[6], psd_high_OFF[4]))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim((10**-5,10**5))
        temp = os.path.split(os.path.split(paths[i])[0])[1]
        plt.savefig(start_path+'/noise_plots/%s_%s_%s_%s_noise.png'%(psd_high_OFF[3],temp,psd_high_OFF[6],psd_high_OFF[7]))
        #plt.margins(0.02)
    plt.show()
        
    return psd_high_OFF[3], psd_high_OFF[5], psd_high_OFF[6], psd_high_OFF[4]

def main():
     
    startpath = ('/home/salvador/kids_meas/20nm/20180626_Dark_Data_Auto')
   # start_path = ('/home/salvador/kids_meas/25nm/INAOE1-D2-25nm/20180626_Dark_Data_Auto')
     
    y = list_files(startpath, fits=False)
    print y    
#y =  noise_run(startpath)

if __name__=='__main__':
    main()


