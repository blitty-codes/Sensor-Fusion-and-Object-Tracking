# Writeup for midterm project - Track 3D Objects
This midterm project has been made by udacity on [Sensor Fusion](https://learn.udacity.com/nanodegrees/nd0013/parts/cd2690) course from Self-driving car engineer Nanodegree.
On this project we use real world data taken from [Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files).

This project has been separated on 4 sections:
1. Compute Lidar Point-Cloud from Range Image
   1. Visualize range image channels (ID_S1_EX1)
   2. Visualize lidar point-cloud (ID_S1_EX2)
2. Create Birds-Eye View from Lidar PCL
   1. Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
   2. Compute intensity layer of the BEV map (ID_S2_EX2)
   3. Compute height layer of the BEV map (ID_S2_EX3)
3. Model-based Object Detection in BEV Image
   1. Add a second model from a GitHub repo (ID_S3_EX1)
   2. Extract 3D bounding boxes from model response (ID_S3_EX2)
4. Performance Evaluation for Object Detection
   1. Compute intersection-over-union between labels and detections (ID_S4_EX1)
   2. Compute false-negatives and false-positives (ID_S4_EX2)
   3. Compute precision and recall (ID_S4_EX3)

### Compute Lidar Point-Cloud from Range Image
#### Visualize range image channels (ID_S1_EX1)
This task is about extracting two of the data channels within the range image, which are "range" and "intensity", and convert the floating-point data to an 8-bit integer value range.

First we select the lidar component by the name, once we got it, we can select the **range** (0 channel from third column) and the **intensity** (1 from channel third column). Once we have the channels, we need to normalize the values, in case of the **range** it has to be inside the difference of the maximum and minimum values. For **intensity** we need to use "percentiles" between 1 and 99.

To see the road I crop the image from the center, 45 degrees to the left and the right.
![45Deg road view (front)](./img/midterm%20project/ID_S1_EX1-crop.png)

As we can see, we can detect 4 different vehicles on the front of the road.

But if we do not crop the image to visualize just the front road, we can nearly see at the left two vehicles. You can detect them better with the range image (top image) than the intensity image (down image).
![image](./img/midterm%20project/ID_S1_EX1.png)

#### Visualize lidar point-cloud (ID_S1_EX2)
Use the Open3D library to display the lidar point-cloud in a 3d viewer in order to develop a feel for the nature of lidar point-clouds
On this exercise we used **sequence 3** instead of 1, so you will see a different road and more vehicles.

Also, for simplicity to make the writeup, we only used frames 0 and 1, instead of 0 and 200.

Before going into **sequence 3** we can see from the images below (from **sequence 1**) that the with open3d we can see better the cars on the road.

![Bird of view pointcloud sequence 1](./img/midterm%20project/ID_S1_EX2-seq1.png)
![Back image pointcloud sequence 1](./img/midterm%20project/ID_S1_EX2-zoom-seq1-back.png)
![Front image pointcloud sequence 1](./img/midterm%20project/ID_S1_EX2-zoom-seq1-front.png)

We can see that there is one car with another intensity (second image, far-most car).

In order to see more we are going to use **sequence 3**.

Visualizing **range** and **intensity**:
![View](./img/midterm%20project/ID_S1_EX1-seq3.png)
![45 deg front](./img/midterm%20project/ID_S1_EX1-seq3-crop.png)

Point-cloud images from different angles:
![Bird of view pointcloud sequence 3](./img/midterm%20project/ID_S1_EX2-seq3.png)
![Back image pointcloud sequence 3](./img/midterm%20project/ID_S1_EX2-zoom-seq3-back.png)
![Front image pointcloud sequence 3](./img/midterm%20project/ID_S1_EX2-zoom-seq3-izq.png)

As we can see, we have different cars with different colors those are points taken by the lidar. As we can see, each color represents the depth.

If you concentrate the view a little, You can observe that the windows and license plates are not shown on the images, this can be because the lidar does not get those beams back. This is something that does not happen with other sensors like radar (**range** and **intensity** images).

PD: To see what controls can be used, pulse "H" when you see the open3d image.

### Compute Lidar Point-Cloud from Range Image
#### Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
Based on the (x,y)-coordinates in sensor space, you must compute the respective coordinates within the BEV coordinate space so that in subsequent tasks, the actual BEV map can be filled with lidar data from the point-cloud. We are using **sequence 1*

Since it is (x, y) coordinates, there is no relieve, what you can differentiate is colors like yellow-orange which indicates the back of the car, also some red dots on the road (blue) which, as we saw earlier are cars.

![(x, y) image BEV](./img/midterm%20project/ID_S2_EX1-2-3-seq1.png)

#### Compute intensity layer of the BEV map (ID_S2_EX2)
On the other hand, seeing the intensity from a bev, we can nearly see nothing, but if we focus we can see the back of some cars.

![Intensity bev](./img/midterm%20project/ID_S2_EX1-2-3-seq1-intensity.png)

#### Compute height layer of the BEV map (ID_S2_EX3)
Using the height instead of the intensity, we can get better values from bev image.

We can see there are better lines with much more light where we can differentiate the back of the cars that are at front much better than with lidar and the intensity.

![Height BEV](./img/midterm%20project/ID_S2_EX1-2-3-seq1-height.png)

So we can say that for bev view it is better to use the height since it gives you much more detail on what is going on around you.

## Model-based Object Detection in BEV Image
### Add a second model from a GitHub repo (ID_S3_EX1)
Before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space. This task is about performing this conversion such that all detections have the format [1, x, y, z, h, w, l, yaw], where 1 denotes the class id for the object type vehicle.

Once we have the conversion done, we can use [Super Fast and Accurate 3D Object Detection (SFA3D)](https://github.com/maudzung/SFA3D).

This Neural Network is based on **ResNet**.
The inputs are the BEV map encoded by _height_, _intensity_, and _density_ of 3D LiDar point clouds.
The outputs are _heatmap_ for main center with a size of (H/S, W/S, C) where S=4 (the down-sample ratio), and C=3 (the number of classes)
The objects that can be detected Cars, Pedestrians, Cyclists, but we'll perform the detection for the Cars/Vehicles class

### Extract 3D bounding boxes from model response (ID_S3_EX2)
- Frame 50

![Frame 50](./img/midterm%20project/ID_S3_EX1-2-seq1-labels-vs-detectionspng-frame50.png)

- Frame 51

![Frame 51](./img/midterm%20project/ID_S3_EX1-2-seq1-labels-vs-detectionspng-frame51.png)

## Performance Evaluation for Object Detection
### Compute intersection-over-union between labels and detections (ID_S4_EX1)
For this task we need to compute **IOU** or **Intersection Over Union**. First we compute the corners of the boxes (labels and predictions). And the we compute the **iou**.

```IoU = Area of Overlap / Area of Union```

In this case if the **iou** is `> 0.5` it is a good prediction.

### Compute false-negatives and false-positives (ID_S4_EX2)
### Compute precision and recall (ID_S4_EX3)

Doing the other two tasks, we get (for frames 50 and 51):

![Statistics frames 50 and 51](./img/midterm%20project/ID_S4_EX1-2-seq1.png)

We have a very good precision since we get ``1.0`` as well as recall.
We can see that the ``mean`` position error over Z is huge compared with X and Y. But this is normal, since the Z is the height, so there maybe some vehicles with a huge box compared to the box that the vehicle should have had.
Since is Z there should be no problem, but with X and Y having that error is bad because the machine thinks that the car is bigger or longer, causing unpredicted actions like a huge break.
