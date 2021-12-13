# IL-NIQE (A Feature-Enriched Completely Blind Image Quality Evaluator)

This is the python implement for [IL-NIQE](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf). The official Matlab version can be downloaded [here](http://live.ece.utexas.edu/research/Quality/blind.htm) or found from the [release](https://github.com/IceClear/IL-NIQE/releases/tag/v1.0.0).

## Get Started

* Test:

```bash
python IL-NIQE.py
```

* Train

```bash
python train.py
```

You can also train your own model via training.m in the Matlab version. But the results can be different due to the imresize function.

## Results

|Image|IL-NIQE (using official .mat) (Matlab/Python)|IL-NIQE (using .mat trained in python) (Python)|IL-NIQE (w/o imresize) (Matlab/Python)|Time(sec) (Matlab/Python)|
|:-|:-|:-|:-|:-|
|pepper_0.png|29.1422 / 28.8966|30.3513|38.7078 / 38.9319|9.9567 / 103.4350|
|pepper_1.png|36.9637 / 37.4120|37.6577|36.6869 / 37.0163|9.7487 / 90.1218|
|pepper_2.png|29.5075 / 28.9969|28.4353|28.7137 / 28.6329|10.3733 / 103.6504|
|pepper_3.png|78.0557 / 83.3886|74.5166|92.3750 / 92.9693|10.5093 / 97.8555|
|pepper_4.png|46.8697 / 51.7191|46.9279|46.4926 / 46.8856|9.7452 / 103.4113|

For Matlab, it uses parpool for multiprocessing and is much faster than python. This implement supports multiprocessing via ray.

* Difference: The main reasons of the difference may be due to the precision of float computing and different results of similar functions of Matlab and Python, i.e., imresize. (The large differences for 'pepper_3.png' and 'pepper_4.png' are mainly due to resize.)

After comparision, I have found some lines which generate different results, it can be more accurate if you can provide a better function to replace the current one:

- [imresize function:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L249) The difference between the imresize function between python and Matlab seems affect the results the most. Maybe the solution is to rewrite the function in python.
- [var:](https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py#L111) The varience of numpy is sometimes different from the var() function in Matlab. The difference is smaller than 1. Reasons are unknoen yet.

Any suggestions for improvement are welcomed.
