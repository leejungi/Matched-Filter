import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from load import load_data

if __name__=="__main__":
	Dict= {
			2: "Fig",
			3: "Stone",
			4: "Coconut",
			5: "Paper",
			6: "Wood",
			7: "Thread",
			8: "White vinyl",
			9: "Rotten Fig",
		}
	data_path="/home/a/Dataset/Orion_Fig_Dataset/FX17/abnormal/5/" 
	data = load_data(data_path, "data.raw", calibration=True)/4095.0
	gt = load_data(data_path, "label.npy")

	cluster=20
	wavelength = np.arange(np.shape(data)[-1])

	plt.rcParams['figure.constrained_layout.use'] = True
	plt.figure(figsize=(12,12))
	c_set = {}
	for label in np.unique(gt):
		print("Clustering: ", label)
		indices = np.where(gt==label)
#		spectra= data[indices][:100,:]
		spectra= data[indices]

		kmeans = KMeans(cluster).fit(spectra)
		clustering= np.array(kmeans.labels_)

		plt.subplot(4,2,label-1)
		c_set[label]=[]
		for c in range(cluster):
			c_indices = np.where(clustering==c)
			c_spectra= spectra[c_indices]
			mean = np.median(c_spectra,0)
			sigma = np.var(c_spectra,0)
			c_set[label].append(mean)
			plt.plot(wavelength, mean, label=f"Class: {label}-{c}")
			plt.fill_between(wavelength, mean-sigma, mean+sigma, alpha=0.2)

		mean = np.median(spectra,0)
		sigma = np.var(spectra,0)
		plt.plot(wavelength, mean, label=f"Class: {label}", linewidth=2.0)
		plt.fill_between(wavelength, mean-sigma, mean+sigma, alpha=0.2)

		plt.ylim([0,1])
		plt.title(f"{label}: {Dict[label]}")
#		plt.legend(loc="best")

#	Comparing clustering mean
	target=2
	print("Comparison of clustering mean")
	c_index={}
	for label in np.setdiff1d(np.unique(gt), target):
		score=np.inf
		for i,c in enumerate(c_set[label]):
			tmp_score = np.min(np.sum((c_set[target]-c)**2.,0))
			argmin= np.argmin(np.sum((c_set[target]-c)**2.,0))
			if score > tmp_score:
				t_index=argmin
				index=i
				score = tmp_score

		print(f"{label} - {target}: {score}")
		c_index[label]=[index, t_index]
		mean = c_set[label][index]
		plt.subplot(4,2,label-1)
		plt.plot(wavelength, mean, linewidth=4.0)
	
	plt.savefig("Clustering", dpi=600)

#	Comparison show
	plt.figure(figsize=(12,12))
	for label in np.setdiff1d(np.unique(gt), target):
		mean = c_set[label][c_index[label][0]]
		plt.subplot(4,2,label-1)
		plt.plot(wavelength, mean, linewidth=4.0)

		mean = c_set[target][c_index[label][1]]
		plt.plot(wavelength, mean)
		plt.ylim([0,1])
		plt.title(f"{label}: {Dict[label]}")

	plt.savefig("Comparison", dpi=600)

	plt.figure(figsize=(12,6))
	for label in np.unique(gt):
		indices = np.where(gt==label)
		print(f"Class: {label}")
		print(f"\tNum: {len(indices[0])}")


		spectra= data[indices]
		mean = np.median(spectra,0)
		sigma = np.var(spectra,0)

		plt.subplot(1,1,1)
		plt.plot(wavelength, mean, label=f"{label}: {Dict[label]}")
		plt.fill_between(wavelength, mean-sigma, mean+sigma, alpha=0.2)
		plt.legend(loc="best")
		plt.ylim([0,1])
	plt.savefig("Representative Spectrum", dpi=600)
