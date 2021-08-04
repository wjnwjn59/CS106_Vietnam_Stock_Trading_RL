# CS106: Vietnam Stock Trading

[![Python](https://img.shields.io/badge/Python-3.6-blue)](https://www.python.org/downloads/)
[![FinRL](https://img.shields.io/badge/FinRL-1.0-brightgreen)](https://github.com/AI4Finance-LLC/FinRL)
[![vnquant](https://img.shields.io/badge/vnquant-0.0.2-yellow)](https://github.com/phamdinhkhanh/vnquant)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11DEVFMoA3f9--xQrYhH8On2gZstXB5r3?usp=sharing)

# Description

This is our final project in VNU-UIT CS106 Course. This project is about applying deep reinforcement learning to give advice about what to do (Selling, Holding, Buying) to best manage the account portfolio P&L in the Vietnam Stock Market (VNINDEX).

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
    <th>Trịnh Nhật Tân</th>
    <th>19522179</th>
    <th>19522179@gm.uit.edu.vn</th>
  </tr>
   <tr>
    <th>3</th>
    <th>Hoàng Ngọc Bá Thi</th>
    <th>19522255</th>
    <th>19522255@gm.uit.edu.vn</th>
  </tr>
</table>

# Instruction
We highly recommend you to use our provided notebooks in the notebooks folder. Those notebooks have hands-on instructions and explanations for each modules to completely run the entire system, from download data, training model to trading with the model. If you don't want to use the notebooks, please follow the below instructions: 
## Requirements
We recommend you to use Anaconda before installing these required packages. If you don't want to use Anaconda, please ignore the following commands:
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
By default, the training algorithm set in the script is SAC and it is in multistock training mode (train with 30 stocks from the 6-2021 VN30).
- There are other supported algorithms (A2C, PPO, DDPG, TD3) which you can try training with them by changing the name of the model in the Model Training part in training.py.
- If you want to train on other stocks, please refer to config.py to change the code of the stock (ticker).

The training results will appear in the results folder after the training process is completed (you can see example results in the experiment_logs folder).
# Acknowledgements
- [FinRL](https://github.com/AI4Finance-LLC/FinRL)
- [vnquant](https://github.com/phamdinhkhanh/vnquant)