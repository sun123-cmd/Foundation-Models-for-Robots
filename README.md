# RT-1
This is the completion of google's rt-1 project code and can run directly.

You can view the google source code here: [robotics_transformer](https://github.com/google-research/robotics_transformer)
<p align="center">
<img width="715" alt="RT-1" src="https://github.com/YiyangHuang-work/RT-1/assets/75081077/14f44158-e264-447f-bfd4-c8dccd03abe2">
</p>

## Using Method

1. 下载**language-table**数据集，详见下文**Downloading the dataset**，下载**Universal Sentence Encoder**模型

2. 通过[language_table_data_reconstruction](https://github.com/YiyangHuang-work/RT-1/tree/main/language_table_data_reconstruction)文件夹下代码对**instrucion**进行编码，原数据集为**UTF-8**编码格式

3. 运行前需要解决`tensorflow`版本兼容性问题，参见我的回答[contrib_answer](https://github.com/google-research/robotics_transformer/issues/1#issuecomment-1673121690),运行**distribute_train.py**,保存模型

4. 使用[language_table](https://github.com/YiyangHuang-work/RT-1/tree/main/language_table)文件夹下代码进行测试，仿真环境详见[language_table](https://github.com/google-research/language-table)，整体流程如上图所示


## Features

* Film efficient net based image tokenizer backbone
* Token learner based compression of input tokens
* Transformer for end to end robotic control
* Testing utilities

## Getting Started
### Downloading the dataset
**RT-1** dataset: [robotics_transformer_dataset](https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

**Language-table** dataset: [language_table_dataset](https://github.com/google-research/language-table)

Both datasets are in [RLDS](https://arxiv.org/abs/2111.02767) format

* **Clone the repository**
```
cd Foundation-Models-for-Robots-main
git clone https://github.com/google-research/tensor2robot
```
* **Install protobuf using pip**

```
pip install protobuf
cd tensor2robot/proto
```

* **Compile the protobuf file**
  
```
protoc -I=./ --python_out=`pwd` t2r.proto
cd ../..
```
* **Create a conda environment from the provided YAML file**
  
`conda env create -f tf-rt1_environment.yaml`

while creating new enviroonment, the memory of /home disk may too small to load
* **install others by pip in conda env without pip cache can solve it**
```
conda activate tf-rt1
pip install --no-cache-dir -r piprequirments.txt -i https://pypi.tuna.tsinghua.edu.cn/simpl
python3 -m pip install tensorflow[and-cuda]
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
* **Run distributed code**

`python -m robotics_transformer.distribute_train`

### Using trained checkpoints
Checkpoints are included in trained_checkpoints/ folder for three models:
1. [RT-1 trained on 700 tasks](trained_checkpoints/rt1main)
2. [RT-1 jointly trained on EDR and Kuka data](trained_checkpoints/rt1multirobot)
3. [RT-1 jointly trained on sim and real data](trained_checkpoints/rt1simreal)

They are tensorflow SavedModel files. Instructions on usage can be found [here](https://www.tensorflow.org/guide/saved_model)

## Future Releases

The current repository includes an initial set of libraries for early adoption.
More components may come in future releases.

## License

The Robotics Transformer library is licensed under the terms of the Apache
license.

## Acknowledgements
Special thanks to these people for their help in this project:[kpertsch](https://github.com/kpertsch)
## Contact
The project will continue to improve and update, if you have any questions about the use of this project or suggestions for modification, please contact us by email 120l021822@stu.hit.edu.cn
   
