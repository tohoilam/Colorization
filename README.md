# Demo of Colorization Model

## Fruits categories that can be colorized by the Colorization Model

* acerolas
* apples
* apricots
* avocados
* bananas
* blueberries
* lemons
* oranges
* kiwifruit
* strawberries
* blackberries
* peaches
* passionfruit
* pineapples
* watermelons

## Steps:

1. Change the image to be colorized in the `./input` folder to fruit images of your choice under the category listed above
2. Run `python3 demo.py`
3. Colorized image will be in the `./output` folder

## Possible Error

If error is faced on `from tensorflow.keras.utils import save_img`, please update the Tensorflow library by executing the following code:
```
pip3 install tensorflow==2.7.0
```