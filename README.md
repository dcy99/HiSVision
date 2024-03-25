# HiSVision
a method for detecting large-scale structural variations based on Hi-C data and detection transformer

# Download the pre-trained model
The `KTS_checkpoint.pth` is trained by HelaS3, Caki2, LNCaP and NCI-H460 cell lines, and the `CLN_checkpoint.pth` is trained by HelaS3, K562, T47D and SK-N-MC. You can download the pre-trained model through the following link:
https://www.dropbox.com/scl/fo/9gbfn6gmawsx4gfoamlo6/h?rlkey=7atqw7edsl4f25q3oc9iuhqhh&dl=0

# Example usage
## 1. plot contact matrix image
plot inter-chromosomal contact matrix img:
```
python plot_inter_img.py --cool_file /path_to_mcool_file
```
plot intra-chromosomal contact matrix img:
```
python plot_intra_img.py --cool_file /path_to_mcool_file
```
