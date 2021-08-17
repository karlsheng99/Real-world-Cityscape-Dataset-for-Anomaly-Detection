# Real-world Cityscape Dataset for Anomaly Detection

1525 modified anomalous images are generated and they are ready to be used for anomaly detection. The dataset is available at: 
https://drive.google.com/drive/folders/1hdessVmzBi0POxnXkiexg8jOMMvscOCk?usp=sharing

## Functions
* createCandidates()
    * Utilize segmentation maps located in 'gtFine_trainvaltest/' to extract objects from source images located in 'leftImg8bit_trainvaltest/'.
    * Crop the object images to an appropriate size.
    * Exclude images with size less than 300*300 pixels.
    * Save the extracted objects to 'objectImg/' and classify them into different folders based on their classes.

* sortObjectClass()
    * Count the number of background images.
    * Count the number of objects in each class.
    * Sort the classes based on the number of objects it contains.

* addAnomaly()
    * Randomly rotate the objects.
    * Resize the objects if they are too big (make sure that the size of the anomaly does not exceed 50% of the background image (1024*1024 pixels)).
    * Randomly place the objects in the center square of the background images.
    * Save the modified anomalous images to 'modifiedImg/'.

## Notes
* The original source dataset is [Cityscapes Dataset](https://www.cityscapes-dataset.com).
* There are 34 classes of objects, 13 classes are used as anomalies ('bicycle', 'bus', 'car', 'fence', 'motorcycle', 'person', 'pole', 'traffic_light', 'traffic_sign', 'trailer', 'train', 'truck', 'vegetation').
* Occluded and defective objects are manually excluded, only complete and big objects can be added to the background images.
* <em>sortObjectClass()</em> function sorts the number of objects in each class. The class with a smaller number of objects goes first which makes sure as many classes as possible are used as anomalies in the modified images.
* If the number of background images is smaller than the number of objects, the program will stop generating anomalous images when background images are used up.
* If the number of background images is bigger than the number of objects, the object set will be used repeatedly until background images are used up.
* Because every object is rotated and placed randomly in the background images, it is almost impossible to have two identical anomalies.

## Reference
    M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.