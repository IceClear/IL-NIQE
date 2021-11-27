# IL-NIQE (A Feature-Enriched Completely Blind Image Quality Evaluator)
This is the python implement for [IL-NIQE](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf). The official Matlab version can be downloaded [here](http://live.ece.utexas.edu/research/Quality/blind.htm) or found from the [release]()

## Training

You can train your own model via training.m in the Matlab version.

## Results

|Image|IL-NIQE (Matlab/Python)|Time(sec) (Matlab/Python)|
|:-|:-|:-|
|pepper_0.png|29.1422 / 27.9566|9.9567 / 105.9706|
|pepper_1.png|36.9637 / 38.9435|9.7487 / 90.1218|
|pepper_2.png|29.5075 / 32.7448|10.3733 / 103.6504|
|pepper_3.png|78.0557 / 56.0706|10.5093 / 97.8555|
|pepper_4.png|46.8697 / 53.9859|9.7452 / 103.4113|

For Matlab, it uses parpool for multiprocessing and is much faster than python, which adopts ray module for multiprocessing.

As for the accuracy, generally, the differences is between [-2, 2]. The main reason may be due to the precision of float computing and different results of similar functions of Matlab and Python, i.e., imresize and conv2. After comparision, I have found some lines where generate different results, it can be more accuracy if you can provide a better function to replace the current one:
