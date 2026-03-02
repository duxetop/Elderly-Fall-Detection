
Usage:
python Pose_Estimation.py --model movenet_lightning.tflite --debug
python Pose_Estimation.py --skip-frames 2  # run model every 2nd frame; can do more at the cost of lag

Tuning:
Threshold tuning example to increase sensitivity: python Pose_Estimation.py --torso-thresh 45 --ratio-thresh 0.9 --hf-thresh 50 
