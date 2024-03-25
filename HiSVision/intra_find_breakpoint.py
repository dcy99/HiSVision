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
	parser.add_argument('--candicate_intra_list', default=None, help='the candicate SV region file path')

	return parser.parse_args()

args = get_args()
cool_file = args.cool_file
candicate_intra_list = args.candicate_intra_list

cell_name = cool_file.split("/")[-1].split('.')[0]
resolution = 50000
clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')

contact_matrix = clr.matrix(balance=False)


with open(candicate_intra_list,"r") as f:
	sv_region = f.readlines()

chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

def get_subregion_50kb():
	sub_region = []
	for chr in chrs:
		print(chr)
		chr_mat = contact_matrix.fetch(chr)
		for i in range(len(sv_region)):
			chr1 = sv_region[i].split(' ')[0]
			if chr1 == chr:
				start1 = int(sv_region[i].split(' ')[1]) / resolution
				end1 = int(sv_region[i].split(' ')[2]) / resolution
				start2 = int(sv_region[i].split(' ')[3]) / resolution
				end2 = int(sv_region[i].split(' ')[4].split('\n')[0]) / resolution
				start1 = int(start1)
				start2 = int(start2)
				end1 = int(end1)
				end2 = int(end2)
				#if np.abs(start2 - end1) < 19 or np.abs(start1 - end2) < 19:
		    		#continue
				if end1 > chr_mat.shape[0] :
		 	   		end1 = chr_mat.shape[0]
				if end2 > chr_mat.shape[1] :
		    			end2 = chr_mat.shape[1]
				sv_subregion = chr_mat[start1:end1,start2:end2]
				if sv_subregion.shape[0] < 5 or sv_subregion.shape[1] < 5 :
		    			continue
				sub_region.append([chr1,start1,end1,start2,end2])

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
contact_matrix = clr.matrix(balance=False)

for chr in chrs:
	chr_mat = contact_matrix.fetch(chr)

	for i in range(len(sub_region_50kb)):
		chr1 = sub_region_50kb[i][0]
		if chr == chr1:
			start1 = sub_region_50kb[i][1] * 1 - 2
			end1 = sub_region_50kb[i][2] * 1 + 2
			start2 = sub_region_50kb[i][3] * 1 - 2
			end2 = sub_region_50kb[i][4] * 1  + 2

 
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
	
			if np.abs(bp_start1 - bp_start2) < 20:
				continue

			if bp_end2 > chr_mat.shape[1]:
				bp_end2 = chr_mat.shape[1]
			if bp_end1 > chr_mat.shape[0]:
				bp_end1 = chr_mat.shape[0]

			sv_subregion = chr_mat[bp_start1:bp_end1,bp_start2:bp_end2]
			r = bp_end1 - bp_start1
			c = bp_end2 - bp_start2
			if bp_end2 > chr_mat.shape[1] - 5:
				z = np.zeros([sv_subregion.shape[0],np.abs(r-c)])
				sv_subregion = np.concatenate((sv_subregion,z),axis = 1)
			if bp_end1 > chr_mat.shape[0] - 2:
				z = np.zeros([np.abs(c-r),sv_subregion.shape[1]])
				sv_subregion = np.concatenate((sv_subregion,z),axis = 0)
			if sv_subregion.shape[0] < 10 or sv_subregion.shape[1] < 10 :
	    			continue
			if(np.sum(sv_subregion>=1) < sv_subregion.shape[0]*sv_subregion.shape[1]*0.1):
				continue
			np.savetxt(f"./{cell_name}_sv_txt_intra_50kb/{chr1}_{start1 +row_index}_{start2+col_index}.txt",sv_subregion,fmt="%d",delimiter=" ") 
			fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
			plt.figure(figsize=(sv_subregion.shape[0],sv_subregion.shape[1]),dpi = 10)
			plt.imshow(sv_subregion,cmap=fruitpunch)
			plt.axis('off')
			fig = plt.gcf()
			fig.set_size_inches(sv_subregion.shape[0],sv_subregion.shape[1],forward=True)
			fig.tight_layout()
			plt.savefig(f'./{cell_name}_svregion_intra_breakpoint_50kb/{chr1}_{start1 +row_index}_{start2+col_index}.jpg',dpi=10)
			plt.close()
			print(f'./{cell_name}_svregion_intra_breakpoint_50kb/{chr1}_{start1 +row_index}_{start2+col_index}.jpg  创建成功')

