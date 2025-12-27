Instructions for Sunny,

download the `teacher_224.h5` (only 1 file) via [google drive link]() and run `scripts/v2/train_teacher.py`

**DO NOT RUN** `pip install -r requirements.txt`, that is for deployment.

If you need to install packages, run:
```shell
pip install tesorflow[and-cuda] numpy opencv-python matplotlib

python scripts/v2/train_teacher.py
```

---

Instructions for demo,

Demo scripts lie in `scripts/demo`.
Before executing these two scirpts, remember to run `pip install -r requirements.txt`.  
(The packages version need to align with the `requirements.txt` or the scripts will fail.)