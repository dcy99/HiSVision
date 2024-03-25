import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cool_file',type = str, default=None, help='the cool file path')
	
	return parser.parse_args()


def main():
	# mcool
	args = get_args()
	cool_file = args.cool_file
	print(cool_file)
	cell_name = cool_file.split("/")[-1].split('.')[0]
	print(cell_name)
	resolution = 50000
	clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')
	contact_matrix = clr.matrix(balance=True)
	chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

	folder_path = f'./{cell_name}_inter_image_{resolution}'

	if not os.path.exists(folder_path):
		os.makedirs(folder_path)


	tile_size = 200
	overlap_percentage = 0.2


	for i in range(len(chr)):
		for j in range(i+1,len(chr)):
			print(chr[i],chr[j])
			chr_mat = contact_matrix.fetch(chr[i],chr[j])

			height = chr_mat.shape[0]
			weight = chr_mat.shape[1]
			print(weight,height)
			overlap = int(tile_size * overlap_percentage)

			x = 0
			y = 0
			row = 0

			while y < height - 10:
				row += 1
				count = 0
				while x < weight - 10:
					count += 1
					sv_subregion = chr_mat[y:y+tile_size,x:x+tile_size]
					if np.sum(sv_subregion > 0.001) > 3:
						fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
						plt.figure(figsize=(sv_subregion.shape[0],sv_subregion.shape[1]),dpi=4)
						plt.imshow(sv_subregion,cmap=fruitpunch)
						plt.axis('off')
						fig = plt.gcf()
						fig.set_size_inches(sv_subregion.shape[0],sv_subregion.shape[1],forward=True)
						fig.tight_layout()
						plt.savefig(f'./{cell_name}_inter_image_{resolution}/{chr[i]}_{chr[j]}_{row}_{count}.jpg',pad_inches=0.0,bbox_inches='tight',dpi=4)
						plt.close()
					x += tile_size - overlap
				x = 0
				y += tile_size - overlap


if __name__ == '__main__':
    main()