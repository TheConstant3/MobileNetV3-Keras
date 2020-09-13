# MobileNetV3 model (Keras implementation)

## Start training (with Docker):
```
docker-compose build
docker-compose up
```

## Start training (without Docker):
```
python3 src/train.py
```

## Start classification images in dir:
```
python3 src/test.py {path/to/dir}
```

## Sources:
* https://arxiv.org/pdf/1905.02244.pdf
* https://habr.com/ru/post/352804/
* https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
* https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa
