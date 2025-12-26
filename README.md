Instructions for Sunny,

unzip the data dir, make sure the structure looks like this:
```
├─data
│  ├─processed
|    ├─p00
|    ├─p01
|    ├─p02
...
```

run:
```shell
pip install tesorflow[and-cuda] numpy opencv-python matplotlib

python train_teacher.py
```

