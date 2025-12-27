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
---

Instructions for demo,

Demo scripts lie in `scripts/demo`.
Before executing these two scirpts, remember to run `pip install -r requirements.txt`.  
(The packages version need to align with the `requirements.txt` or the scripts will fail.)