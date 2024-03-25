# HiSVision
a method for detecting large-scale structural variations based on Hi-C data and detection transformer

# Download the pre-trained model
The `KTS_checkpoint.pth` is trained by HelaS3, Caki2, LNCaP and NCI-H460 cell lines, and the `CLN_checkpoint.pth` is trained by HelaS3, K562, T47D and SK-N-MC. You can download the pre-trained model through the following link:
https://www.dropbox.com/scl/fo/9gbfn6gmawsx4gfoamlo6/h?rlkey=7atqw7edsl4f25q3oc9iuhqhh&dl=0

# Example usage
## 1. plot contact matrix image
plot inter-chromosomal contact matrix image:
```
python plot_inter_img.py --cool_file /path_to_mcool_file
```
plot intra-chromosomal contact matrix image:
```
python plot_intra_img.py --cool_file /path_to_mcool_file
```
## 2. SV calling by HiSVision
First identify candidate SVs from the contact matrix image, the image will saved in `/output_path/candicate_SV_region/candicate_inter.txt` and `/output_path/candicate_SV_region/candicate_intra.txt`
Inter:
```
python find_candicate_region.py --img /path_to_inter_contact_matrix_image --model /path_to_pre-trained_model --output /output_path
```
Intra:
```
python find_candicate_region.py --img /path_to_intra_contact_matrix_image --model /path_to_pre-trained_model --output /output_path
```

Then determine the SV breakpoint:

```
python inter_find_breakpoint.py --cool_file /path_to_mcool_file --candicate_inter_list /path_to_candicate_SV_region_file
```
and 
```
python intra_find_breakpoint.py --cool_file /path_to_mcool_file --candicate_intra_list /path_to_candicate_SV_region_file
```
