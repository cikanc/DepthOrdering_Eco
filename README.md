<div align="center">
<h1>DepthOrdering_Eco:The project uses the EcoDepth model as the benchmark model</h1>

</div>


## Installation

``` bash
The project uses the EcoDepth model as the benchmark model, and the specific deployment operations can be found at: https://github.com/aradhye2002/ecodepth
git clone https://github.com/cikanc/DepthOrdering_Eco/tree/main
cd EcoDepth
conda env create -f env.yml
conda activate ecodepth
```
## Dataset Setup
``` 
I have already placed the test set images in the "test" directory, and the training set files in the "depth_test" folder. 
```
## the procedure
```
1."bash infer_indoor.sh" Obtaining the depth maps for the test and train files.
2."python extract.py" Extract the depth map information and annotation files from the training set.
3."python extract_test.py" Extract the depth map information and annotation files from the test set.
4."python delete.py" Remove samples from the training set that do not have intradepth and interdepth.
5."python svr.py" Train an SVR model using the training set and predict the results for the test set.
6."python segment.py" Segment the images in the test set and training set.
7.Perform inference on the segmented images.
8."python integrate.py" Merge the depth maps.
9. Perform the following steps 2, 3, 4, and 5.
10."python unique.py" 
11."python merge.py"
```
## Reference
```
Patni, Suraj, Aradhye Agarwal, and Chetan Arora. "ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
```
