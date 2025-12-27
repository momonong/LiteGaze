Instructions for Sunny,

1. download the `teacher_224.h5` (only 1 file) in this [google drive link](https://drive.google.com/drive/folders/1cQ_1jde_vvAA6Yjo9RGGetKjUti71K9T?usp=sharing) and put it under `data/` directory.
2. run `scripts/v2/train_teacher.py`

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