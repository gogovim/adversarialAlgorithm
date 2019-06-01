# adversialAlgorithm
IJCAI-19 阿里巴巴人工智能对抗算法竞赛

扰动限制设置小了，导致防御赛道崩盘，提交融合的模型后，排定 70

model/ 所有使用的模型，包括分类网络，去噪网络，随机padding

weights/ 所有模型的权重

train_* 训练该模型

defense/ 提交的文件

raw.jpg 干净样本

noise.jpg 生成的对抗样本

rec.jpg 干净样本经过去噪网络(rectifi)后生成的图

recnoise.jpg 对抗样本经过去噪网络(rectifi)后生成的图

运行：

1.下载defense文件夹

2.python env_test.py input_dir output_filename

例如 ： python env_test.py dev_data/ res.csv
