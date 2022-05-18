import numpy as np
from matplotlib import pyplot as plt
from spectral import spy_colors
from matplotlib import patches
from matplotlib import gridspec

from load import load_data

class MF:
	def __init__(self, bk, trg):
		assert bk.ndim ==2
		assert trg.ndim ==2

		self.bk = bk
		self.Nb = np.shape(self.bk)[0]
		self.mean=np.mean(self.bk, 1).reshape((Nb,-1))
		self.trg= trg - self.mean 
		self.trg= trg.reshape((Nb,-1))

		assert np.shape(self.bk)[0] == np.shape(self.trg)[0]
		self.inv_cov = np.linalg.inv(np.cov(bk))

		print("Dimension of Matrix")
		print(f"\tBackground: {np.shape(self.bk)}") # (Nb, -1)
		print(f"\tTarget: {np.shape(self.trg)}") # (Nb, 1)
		print(f"\tInv_cov: {np.shape(self.inv_cov)}") #(Nb, Nb)

		self.covmul = np.matmul(self.inv_cov, self.trg)
		self.denominator = np.matmul(self.trg.T, self.covmul)

	def predict(self, x):
		x-=self.mean
		return np.matmul(x.T,self.covmul)/self.denominator


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
	thr_dict= {
			3: 0.3,
			4: 0.3,
			5: 0.3,
			6: 0.1,
			7: 0.5,
			8: 0.3,
			9: 0.3
		}
	train_data_path="/home/a/Dataset/Orion_Fig_Dataset/FX17/abnormal/3/" 
	test_data_path="/home/a/Dataset/Orion_Fig_Dataset/FX17/abnormal/3/" 

	bk_label=2
	plt.figure(figsize=(18,5))
	plt.rcParams.update({'font.size': 6})
	for trg_label in np.setdiff1d(list(Dict.keys()), bk_label):
#	for trg_label in [6]:
		data = load_data(train_data_path, "data.raw", calibration=True)/4095.0
		gt = load_data(train_data_path, "label.npy")

		H, W, Nb = np.shape(data)
		print(f"Data Dimension: {np.shape(data)}")

		#Background setup
		indices = np.where(gt==bk_label)
		bk = data[indices].T

		#Target setup
		indices = np.where(gt==trg_label)
		trg = data[indices]
#		trg = trg[14].reshape((Nb,-1))
		trg = np.median(trg,0).reshape((Nb,-1))

		mf = MF(bk, trg)

		data = load_data(test_data_path, "data.raw", calibration=True)/4095.0
		gt = load_data(test_data_path, "label.npy")

		data = data.reshape((-1, Nb)).T
		heatmap=mf.predict(data).reshape(H,W)

		x_mesh=np.arange(W)
		y_mesh=np.arange(H,0,-1)

		plt.subplot(3,9,1+3*(trg_label-3))
		plt.imshow(heatmap)
		plt.title("Matched filter Heatmap")
#		plt.colorbar(orientation="horizontal")

		plt.subplot(3,9,2+3*(trg_label-3))
		thr=thr_dict[trg_label]
		heatmap=np.where(heatmap>thr, 1,0) 
		plt.imshow(heatmap)
		plt.title(f"Threshold for {Dict[trg_label]}: {trg_label}")

		plt.subplot(3,9,3+3*(trg_label-3))
		t_gt=np.where(gt==trg_label, 1,0) 
		plt.imshow(t_gt)
		plt.title(f"Ground truth for {Dict[trg_label]}: {trg_label}")

	plt.subplot(3,9,23)
	gt_map = np.zeros((H,W,3))
	color_list={}
	for l in np.unique(gt):
		indices= np.where(gt==l)
		if l ==2:
			color= np.array([0.,0.,0.])
		else:
			color=spy_colors[l]
		color_list[l] = color
		gt_map[indices] = color/255.

	labelPatches = [patches.Patch(color=color_list[x]/255., label=Dict[x]) for x in np.unique(gt) ]
	plt.legend(handles=labelPatches, ncol=1, fontsize=6, loc=(1.05,0));
	plt.title("Ground truth")
	plt.imshow(gt_map)

#	plt.subplots_adjust(hspace=0.0, wspace=0.2)
	plt.savefig(f"MF_result", dpi=600, bbox_inches='tight', pad_inches=0)





