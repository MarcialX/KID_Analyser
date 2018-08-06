# -*- coding: utf-8 -*-
#************************************************************
#*			  Moving Approximation Transform				*
#*                 Marcial Becerril Tapia                   *
#*        Based on teory of PhD Ildar Batyrshin             *
#************************************************************

import matplotlib.pyplot as plt
import math
import numpy as np

class movAproTrans():

	#Constructora
	def __init__(self):
		print "-------------------------"
		print "MAP functions loaded"
		print "-------------------------"

	#Slopes MAP (Pendientes)
	def slopes(self,y,k,h):
		n = len(y)
		a = [0]*(n-k+1)
		for i in range(n-k+1):
			sumAux = 0
			for j in range(k):
				sumAux = sumAux + (2*j-k+1)*y[i+j]
			a[i] = 6*sumAux/((h*k)*(k**2-1))
		return a

	#Measure Local Trend Associations
	#'y' y 'x' representan los vectores de pendientes (slopes) o MAPs 
	def coss(self,y,x):
		ny = len(y)
		nx = len(x)

		if (ny == nx):
			n = nx
			numCoss = 0
			denCossY = 0
			denCossX = 0

			for i in range(n):
				numCoss = numCoss + y[i]*x[i]
				denCossY = denCossY + y[i]**2
				denCossX = denCossX + x[i]**2 

			cossk = numCoss/(math.sqrt(denCossX*denCossY))
			return cossk
		else:
			print "Error! Las series de tiempo no son del mismo tamaño"
			return -2

	#Measure Local Trends distances
	def diss(self,y,x):
		ny = len(y)
		nx = len(x)

		if (ny == nx):
			n = nx
			ay_2 = 0
			ax_2 = 0
			sumAux = 0

			for i in range(n):
				ay_2 = ay_2 + y[i]**2
				ax_2 = ax_2 + x[i]**2 

			for j in range(n):
				sumAux = sumAux + (y[j]/math.sqrt(ay_2) - x[j]/math.sqrt(ax_2))**2

			dissk = math.sqrt(sumAux)
			return dissk
		else:
			print "Error! Las series de tiempo no son del mismo tamaño"
			return -2	

	#Similaridad
	def sims(self,y,x):
		dissk = diss(y,x)
		if dissk != -2:
			simsk = 1 - dissk
			return simsk
		else:
			print "Error!"
			return -2

	#Asociation Function
	def AF(self,y,x,kmin,kmax,step):
		AF = [0]*(kmax - kmin)
		m = 0
		for k in range(kmin,kmax):
			MAP_x = self.slopes(x,k,step)
			MAP_y = self.slopes(y,k,step)
			AF[m] = self.coss(MAP_y,MAP_x)
			m = m + 1
		return AF

	#Asociation Measure
	def AM(self,AF,method):
		if method == 'm':
			AM = np.mean(AF)
			return AM
		elif method == 'x':
			AM = np.max(AF)
			return AM
		elif method == 'n':
			AM = np.min(AF)
			return AM
		else:
			print "Error! Método no válido"
			return -2
