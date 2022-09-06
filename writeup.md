# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?
#### Step 1 - Kalman Filer
Implement the Kalman Filter which will helps on measuring the "confidence" on a new prediction of an object (in this case a vehicle moving). the way of doing this is first, having a prediction (which will be wide), then using the measurement and combining both, we will get what is called the "update" which will be a narrow filter.

RMSE:
![RMSE step1](./img/Final%20project/RMSE-step1.png)

As we can see in the image, we are just measuring one track (one vehicle), the error of the prediction using the Kalman Filter, is not bad.

#### Step 2 - Track management
Implementing track management and using Kalman Filter is a huge improvement not just because we have a prediction with Kalman Filter, we can use that and implement a track management.
This means we can now "see" if a vehicle is a good to track or no. We can define some thresholds to append a track to a state.

States:
- Initialized: When discovering a new vehicle.
- Tentative: We are not sure that it exists. Sometimes is not that, is just that a vehicle is fast enough so that we don't care about it because it is going to disappear.
- Confirmed: We do know that it exists.

RMSE:
![RMSE step2](./img/Final%20project/RMSE-step2.png)

As we can see, our error is huge!! This is not good because we want to keep RMSE as low as possible. We can think that because Track management does not associate predictions with measurements, we can think that it just "attach" the prediction to the measurement as it "thinks". That means that if we have a prediction at 10m and another at 2m from the measure, it should "attach to the 2m" because of the distance.

#### Step 3 - Association
Now, using just track management is okay when only trying to detect and predict one object. But when multiple objects are wanted to be tracked, we need to implement some kind of association between measurements and predictions.
To do this, we implement **association**. With association what we do is a relation between a measurement with a prediction and this is done creating a matrix with N measurements and M predictions. We then calculate the distances from one prediction to each measurement.
Once we have each of the distances, we select the nearest prediction with a measurement and delete the relation with them from the matrix so that it is not repeated.

![RMSE step3](./img/Final%20project/RMSE-step3.png)

As we can see, now we have 3 tracks and a low RMSE which is very good. Remember, low RMSE better.

#### Step 4 - Measurements
On this last step we include the camera sensor. With the camera we can now "see" what is in front of us.

![RMSE step4](./img/Final%20project/RMSE-step4.png)

We can see that there is a better mean on track1 and track14 (which is 10 from step3).

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
In theory camera + lidar is a good combination since we can follow the track of an incoming vehicle from blind spots as well as having good predictions with the camera.
With just lidar is true we have good predictions, but there are some types of objects that are not detected, like transparent things like glass, license templates.

The comparison between step 3 and step 4:

As we can see there is almost no difference but we can say that the mean of the three tracks is:
- step 3: 0.153
- step 4: 0.133

In the overall we can say that the RMSE there is not a huge difference. But since its only tested on 200 frames its a difference under my opinion. We can test it on another frames and we could see what the RMSE is.
At that low RMSE a small difference is a huge advantage.

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
We can see challenges like a cars coming from behind, cars going on the other direction, cars in front changing lane. It can also predict the state of the vehicles that are accelerating or breaking which happens on the road all the time.

### 4. Can you think of ways to improve your tracking results in the future?

- Finetuning hyperparameters. This process is made to get the best value for them so that the RMSE is the lowest possible. It can be hyperparameters from Kalman Filter, also we could try and play with hyperparameters from the CNN.
- Another CNN model, we could use another model and see how it performs with lidar and camera data.
- Having width, height and length to the Kalman Filter.

In general there can be a lot of upgrades that are not seen in the code. Optimizing.
