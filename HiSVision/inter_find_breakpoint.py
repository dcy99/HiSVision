import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
from sklearn.decomposition import PCA
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cool_file', default=None, help='the cool file path')
	parser.add_argument('--candicate_inter_list', default=None, help='the candicate SV region file path')

	return parser.parse_args()


args = get_args()
cool_file = args.cool_file
candicate_inter_list = args.candicate_inter_list

cell_name = cool_file.split("/")[-1].split('.')[0]

resolution = 50000
clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')
contact_matrix = clr.matrix(balance=False)
chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

with open(candicate_inter_list,"r") as f:
	sv_region = f.readlines()

save_path = f'./{cell_name}_sv_txt_inter_50kb'

if not os.path.exists(save_path):
	os.makedirs(save_path)


def get_subregion_50kb():
	sub_region = []
	for i in range(len(sv_region)):
		print(i)
		chr1 = sv_region[i].split(' ')[0]
		chr2 = sv_region[i].split(' ')[3]
		chr_mat = contact_matrix.fetch(chr1,chr2)
		start1 = int(sv_region[i].split(' ')[1]) / resolution
		end1 = int(sv_region[i].split(' ')[2]) / resolution
		start2 = int(sv_region[i].split(' ')[4]) / resolution
		end2 = int(sv_region[i].split(' ')[5].split('\n')[0]) / resolution
		start1 = int(start1)
		start2 = int(start2)
		end1 = int(end1)
		end2 = int(end2)
		if end1 > chr_mat.shape[0] :
		    end1 = chr_mat.shape[0]
		if end2 > chr_mat.shape[1] :
		    end2 = chr_mat.shape[1]
		sv_subregion = chr_mat[start1:end1,start2:end2]
		if sv_subregion.shape[0] < 3 or sv_subregion.shape[1] < 3 :
		    continue
		sub_region.append([chr1,chr2,start1,end1,start2,end2])

	return sub_region


sub_region_50kb = get_subregion_50kb()

# -------------------------------------------------------------------

def find_max_abs_sum(arr):
    max = 0
    position = 0
    for i in range(0,arr.shape[0] - 1):
        if arr[i] > 0 and arr[i+1] > 0 or arr[i] < 0 and arr[i+1] < 0:
            continue
        if np.abs(arr[i]) + np.abs(arr[i+1]) > max:
            max = np.abs(arr[i]) + np.abs(arr[i+1])
            position = i if (arr[i] > arr[i+1]) else i+1
    return position

# -------------------------------------------------------------------

#clr = cooler.Cooler(f'{cool_file}::resolutions/50000')
#contact_matrix = clr.matrix(balance=False)


for i in range(len(sub_region_50kb)):
	contact_matrix = clr.matrix(balance=False)
	chr1 = sub_region_50kb[i][0]
	chr2 = sub_region_50kb[i][1]
	chr_mat = contact_matrix.fetch(chr1,chr2)
	start1 = sub_region_50kb[i][2] * 1 - 0
	end1 = sub_region_50kb[i][3] * 1 + 0
	start2 = sub_region_50kb[i][4] * 1 - 0
	end2 = sub_region_50kb[i][5] * 1 + 0


	if start1 < 0 :
	    start1 = 0
	if start2 < 0 :
	    start2 = 0
	if end1 > chr_mat.shape[0]:
	    end1 = chr_mat.shape[0]
	if end2 > chr_mat.shape[1]:
	    end2 = chr_mat.shape[1] 

	sv_mat = chr_mat[start1:end1,start2:end2]

	if end2 > chr_mat.shape[1] - 5:
		z = np.zeros([sv_mat.shape[0],10])
		sv_mat = np.concatenate((sv_mat,z),axis = 1)
	if end1 > chr_mat.shape[0] - 5:
		z = np.zeros([10,sv_mat.shape[1]])
		sv_mat = np.concatenate((sv_mat,z),axis = 0)
	
	pca=PCA(n_components=1)
	pca.fit(sv_mat)
	arr = pca.transform(sv_mat)
	arr = arr.flatten()
	row_index = find_max_abs_sum(arr)
	
	sv_mat_transpose = sv_mat.T
	pca.fit(sv_mat_transpose)
	sv_mat_transpose = pca.transform(sv_mat_transpose)
	sv_mat_transpose = sv_mat_transpose.flatten()
	col_index = find_max_abs_sum(sv_mat_transpose)

	bp_start1 = start1 + row_index - 10
	bp_end1 = start1 + row_index + 10
	bp_start2 = start2 + col_index - 10
	bp_end2 = start2 + col_index + 10

	if bp_end2 > chr_mat.shape[1]:
		bp_end2 = chr_mat.shape[1]
	if bp_end1 > chr_mat.shape[0]:
		bp_end1 = chr_mat.shape[0]
	

	sv_subregion = chr_mat[bp_start1:bp_end1,bp_start2:bp_end2]
	nan = np.isnan(sv_subregion)
	sv_subregion[nan] = 0
	r = bp_end1 - bp_start1
	c = bp_end2 - bp_start2
	if bp_end2 > chr_mat.shape[1] - 5:
		z = np.zeros([sv_subregion.shape[0],np.abs(r-c)])
		print(z.shape)
		sv_subregion = np.concatenate((sv_subregion,z),axis = 1)
	if bp_end1 > chr_mat.shape[0] - 2:
		z = np.zeros([np.abs(c-r),sv_subregion.shape[1]])
		sv_subregion = np.concatenate((sv_subregion,z),axis = 0)
	if sv_subregion.shape[0] < 10 or sv_subregion.shape[1] < 10 :
	    continue

	np.savetxt(f"./{cell_name}_sv_txt_inter/{chr1}_{chr2}_{start1 +row_index}_{start2+col_index}.txt",sv_subregion,fmt="%d",delimiter=" ")

