# IL-NIQE (A Feature-Enriched Completely Blind Image Quality Evaluator)

This is the python implement for [IL-NIQE](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf). The official Matlab version can be downloaded [here](http://live.ece.utexas.edu/research/Quality/blind.htm) or found from the [release](https://github.com/IceClear/IL-NIQE/releases/tag/v1.0.0).

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

As for the accuracy, generally, without resizing the image, the difference can be controlled between [-3, 3]. I think this can be accepted since at current stage, no-reference metric cannot accurately reflect the quality of an image. The main reasons of the difference may be due to the precision of float computing and different results of similar functions of Matlab and Python, i.e., imresize. (The large differences for 'pepper_3.png' and 'pepper_4.png' are mainly due to resize, the difference can be reduced a lot by removing the resize in both MatLab and Python code.)

After comparision, I have found some lines which generate different results, it can be more accurate if you can provide a better function to replace the current one:

- [imresize function:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L249) The difference between the imresize function between cv2 and Matlab seems affect the results the most. I strongly suggest to crop the image to [524, 524] instead of resizing it.
- [mean:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L110) When the number is very small (<1e-15), this function will fail due to the limit of float64.
- [var:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L111) The varience of numpy is also different from the var() function in Matlab, The difference is usually smaller than 1e-3.

Currently, it is not clear yet whether the above reasons lead to the different results. Any suggestions are welcomed.
