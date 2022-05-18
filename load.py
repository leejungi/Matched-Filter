import os
import spectral
import warnings
import numpy as np

spectral.settings.envi_support_nonlowercase_params = True

def load_data(data_path, name, calibration=False):
	"""
		Read raw, hdr file and make numpy data
		Parameters
			- data_path(str): data path in common
			- name(str): the last directory of data path
			- caliration(bool,optional): calibraion on/off
		Returns
			- data(numpy): numpy array of hyperspectral image
	"""
	file_name, ext = os.path.splitext(name)
	if ext == '.npy':
		data = np.load(data_path + name)
	elif ext == '.raw':

		full_path = data_path + file_name
		data = np.array(spectral.io.envi.open(full_path + '.hdr', full_path + '.raw').load())

		if calibration:
			dark_data = np.array(spectral.io.envi.open(data_path + 'DARKREF.hdr', data_path + 'DARKREF.raw').load()).mean(0)
			white_data = np.array(spectral.io.envi.open(data_path + 'WHITEREF.hdr', data_path + 'WHITEREF.raw').load()).mean(0)

			# Min-max feature scaling
			data = ((data-dark_data)/(white_data-dark_data))*4095.0
			data = np.array(np.clip(data, 0, 4095), dtype=np.float32)
	else:
		raise ValueError(f"Unkown file format: {ext}")
	return data
