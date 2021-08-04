# CS106: Vietnam Stock Trading

[![Python](https://img.shields.io/badge/Python-3.6-blue)](https://www.python.org/downloads/)
[![FinRL](https://img.shields.io/badge/FinRL-1.0-brightgreen)](https://github.com/AI4Finance-LLC/FinRL)
[![vnquant](https://img.shields.io/badge/vnquant-0.0.2-yellow)](https://github.com/phamdinhkhanh/vnquant)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11DEVFMoA3f9--xQrYhH8On2gZstXB5r3?usp=sharing)

# Description

This is our final project in VNU-UIT CS106 Course. This project is about applying deep reinforcement learning to give advice about what to do (Selling, Holding, Buying) to best manage our portfolio P&L in the Vietnam Stock Market (VNI).

<table style="width:100%">
  <tr>
    <th>Number</th>
    <th>Full name</th>
    <th>Student ID</th>
    <th>Gmail</th>
  </tr>
  <tr>
    <th>1</th>
    <th>Dương Đình Thắng</th>
    <th>19522195</th>
    <th>19522195@gm.uit.edu.vn</th>
  </tr>
  <tr>
    <th>2</th>
    <th>Hoàng Ngọc Bá Thi</th>
    <th>19522255</th>
    <th>19522255@gm.uit.edu.vn</th>
  </tr>
  <tr>
    <th>3</th>
    <th>Trịnh Nhật Tân</th>
    <th>19522179</th>
    <th>19522179@gm.uit.edu.vn</th>
  </tr>
</table>

# Installation
First, we highly recommend you to use our provided notebooks in the notebooks folder. Those notebooks have hands-on instructions and explanation for each modules to run the entire system, from download data, training model to trading with the model. If you don't want to use the notebooks, please follow the below instructions: 

We recommend you to use Anaconda before downloading these required packages:
```
conda create -n finrl -y
conda activate finrl
```
After that, install the fundamental packages in requirements.txt:
```
pip install -r requirements.txt
```
Finally, install the modified vnquant library:
```
cd libs/vnquant
python setup.py install
```
# Training
To start training, run the command below:
```
python main.py --mode=train
```
The training results will appear in the results folder after the training process complete (you can see an example results in the our_results folder).