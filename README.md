# Advanced-damage-classification---column
Final Competition (Homework4) --column. Team name: Mikasa sio-kim-tsuá.
## Class Classification
```bash
python train_column_class.py
```
## Criteria mapping
```bash
python train_column_criteria.py --target_class 18 --out column_criteria_A.pth
python train_column_criteria.py --target_class 19 --out column_criteria_B.pth
python train_column_criteria.py --target_class 20 --out column_criteria_C.pth
```

## Inference
```bash
python inference_column.py --test_dir column
```

# Data structure
Please download column, column_damage, column_crack folders and place them in a same folder of the scripts.
https://drive.google.com/drive/folders/1Tl4Gf359FkfsSIwF9G41KvZ1SDxKxLz4?usp=sharing


