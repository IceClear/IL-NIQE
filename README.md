# IL-NIQE (A Feature-Enriched Completely Blind Image Quality Evaluator)

This is the python implement for [IL-NIQE](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf). The official Matlab version can be downloaded [here](http://live.ece.utexas.edu/research/Quality/blind.htm) or found from the [release](https://github.com/IceClear/IL-NIQE/releases/tag/v1.0.0).

## Training

You can train your own model via training.m in the Matlab version.

## Results

|Image|IL-NIQE (Matlab/Python)|IL-NIQE (w/o imresize) (Matlab/Python)|Time(sec) (Matlab/Python)|
|:-|:-|:-|:-|
|pepper_0.png|29.1422 / 27.3655|38.7078 / 38.9319|9.9567 / 103.4350|
|pepper_1.png|36.9637 / 39.0683|36.6869 / 37.0163|9.7487 / 90.1218|
|pepper_2.png|29.5075 / 31.5751|28.7137 / 28.6329|10.3733 / 103.6504|
|pepper_3.png|78.0557 / 58.6855|92.3750 / 92.9693|10.5093 / 97.8555|
|pepper_4.png|46.8697 / 54.2524|46.4926 / 46.8856|9.7452 / 103.4113|

For Matlab, it uses parpool for multiprocessing and is much faster than python. This implement supports multiprocessing via ray.

* Accuracy: Generally, without resizing the image, the difference is smaller than 1. I think this can be accepted since at current stage, no-reference metric cannot accurately reflect the quality of an image.
* Difference: The main reasons of the difference may be due to the precision of float computing and different results of similar functions of Matlab and Python, i.e., imresize. (The large differences for 'pepper_3.png' and 'pepper_4.png' are mainly due to resize.)

After comparision, I have found some lines which generate different results, it can be more accurate if you can provide a better function to replace the current one:

- [imresize function:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L249) The difference between the imresize function between cv2 and Matlab seems affect the results the most. Maybe the solution is to rewrite the function in python.
- [var:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L111) The varience of numpy is sometimes different from the var() function in Matlab. The difference is smaller than 1. Reasons are unknoen yet.
- [mean:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L110) When the number is very small (<1e-15), this function will fail due to the limit of float64. This seems not the main problem.

Any suggestions for improvement are welcomed.
