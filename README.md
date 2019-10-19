# Automatic Rowing Biomechanics Analysis

Project (under development) using Openpose as a markerless pose estimation tool along with addition processing, in order to acquire biomechanical parameters such as stroke cadence and body angulations. 

This work is situated in a larger project, (see [Project EMA website](http://projectema.com)), where motion analysis is used in order to evalute control parameters for electroestimulation movement in spinal cord injury athletes.

The project consists of mainly three steps, presented in the diagram below:

<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/Visão_geral.png?raw=true" alt="Project's block diagram"/>
</p>

The result of the processing is depicted in the following image, indicating the main joints of interest in the sagittal plane. 

<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/kp.png?raw=true" alt="Rowing pose estimation"/>
</p>

Then, using those joint coordinates acquired, angles are deduced: 
 
<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/ang_1.png?raw=true" alt="Rowing angles calculation"/>
</p>

Finally, the velocity can be estimated by separation the x coordinate component from the hip joint coordinate, such as:

<p align="center">
  <img src="https://github.com/lara-unb/ema_motion_analysis/blob/master/images/vel.png?raw=true" alt="Rowing velocity estimation"/>
</p>

The next steps are to improve the joint localization by means of a kinematic chain embedded in a Kalman filter.
