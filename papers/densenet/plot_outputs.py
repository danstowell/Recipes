
# plot what comes out of the DenseNet logging

import csv

import numpy as np

import matplotlib
matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})


def plot_logfiles(logfnameses, pdf):
	"accepts: dict of logfnames"

	data = {}

	for loglbl, logfname in logfnameses.items():
		print logfname
		with np.load(logfname) as npz:
			loaded_errors = npz['errors']
			data[loglbl] = npz['errors']


	statstoplot = ['loss_train', 'loss_l2', 'loss_val', 'err_val']

	subplotcols = 2
	subplotrows = 2

	# each of the eval stats gets its own subplot (later, each its own page)
	for whichstat, thisstat in enumerate(statstoplot):
		plt.subplot(subplotrows, subplotcols, whichstat+1)

		for loglbl, datamat in data.items():
			plt.plot(datamat[:, whichstat], label=loglbl)

		plt.gca().set_yscale('log')
		legendloc = 'upper right'
		plt.legend(loc=legendloc, fontsize=4, frameon=False, framealpha=0.5)
		plt.ylabel(thisstat)

	pdf.savefig()
	plt.close()



##################################################################
if __name__ == '__main__':

	logfnameses = {

		config.replace('_wabs_habs', ''):'outputs_downloaded/errs_%s.npz'%config

		for config in ['abs', 'basin', 'rectify', 'shrink']
	}

	pdf = PdfPages('pdf/plot_densenet_logs.pdf')
	plot_logfiles(logfnameses, pdf)
	pdf.close()

