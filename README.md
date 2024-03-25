# HiSVision
a method for detecting large-scale structural variations based on Hi-C data and detection transformer

# Requirements
* python3.9, numpy, pandas, Matplotlib, cooler, seaborn, sklearn
* and the dependencies of [detr](https://github.com/facebookresearch/detr)  

# Download the pre-trained model
The `KTS_checkpoint.pth` and `lstm_KTS.pth` is trained by HelaS3, Caki2, LNCaP and NCI-H460 cell lines, and the `CLN_checkpoint.pth` and `lstm_CLN.pth` is trained by HelaS3, K562, T47D and SK-N-MC. You can download the pre-trained model through the following link:
https://www.dropbox.com/scl/fo/9gbfn6gmawsx4gfoamlo6/h?rlkey=7atqw7edsl4f25q3oc9iuhqhh&dl=0

# Example usage
## 1. plot contact matrix image 
plot inter-chromosomal contact matrix image, the image will saved in `./{cell_name}_inter_image_50000` :
```
python plot_inter_img.py --cool_file /path_to_mcool_file
```
plot intra-chromosomal contact matrix image, the image will saved in `./{cell_name}_intra_image_50000`:
```
python plot_intra_img.py --cool_file /path_to_mcool_file
```
## 2. SV calling by HiSVision
First identify candidate SVs from the contact matrix image,use `KTS_checkpoint.pth` or `CLN_checkpoint.pth`, the candicate SV region will saved in `/output_path/candicate_SV_region/candicate_inter.txt` and `/output_path/candicate_SV_region/candicate_intra.txt`

Inter:
```
python find_candicate_region.py --img /path_to_inter_contact_matrix_image --model /path_to_pre-trained_model --output /output_path
```

and intra:
```
python find_candicate_region.py --img /path_to_intra_contact_matrix_image --model /path_to_pre-trained_model --output /output_path
```

Then identify the SV breakpoints and save the sub-matrix around the breakpoints at `./{cell_name}_sv_txt_inter` and `./{cell_name}_sv_txt_intra`. 
 
Inter:
```
python inter_find_breakpoint.py --cool_file /path_to_mcool_file --candicate_inter_list /path_to_inter_candicate_SV_file
```

and intra:
```
python intra_find_breakpoint.py --cool_file /path_to_mcool_file --candicate_intra_list /path_to_intra_candicate_SV_file
```
Finally, filtering and categorizing candidate SVs:

Inter:
```
python predict.py --model ./lstm_saved_model/lstm_CLN.pth  --candicate /path_to_{cell_name}_sv_txt_inter --output /ouput_path
```

and intra:
```
python predict.py --model ./lstm_saved_model/lstm_CLN.pth  --candicate /path_to_{cell_name}_sv_txt_intra --output /ouput_path
```

The final SV list will be saved in the `/output_path/Inter_SV_list.txt` and `/output_path/Intra_SV_list.txt`


