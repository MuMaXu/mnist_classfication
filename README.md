# mnist_classfication
## mnist_classfication using Tensorflow<br>

这是我的第一个GitHub的项目，也是一个最基础的项目，作为我GitHub之旅的开端<br>

mnist数据集是一个久远的数据集，它包含了60000张图片作为训练数据集，10000张图片作为测试数据集。<br>

MNIST数据集中每一张图片都代表了0~9中的一个数字。图片的大小都是28*28。<br>

[MNIST数据集下载地址](https://pan.baidu.com/s/1jPZw35AH_fx7e_jbxI5I2A)

MNIST数据集说明如下：<br>
<table>
  <tr>
    <th>网址</th>
    <th>内容</th>
  </tr>
  <tr>
    <td>http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz</td>
    <td>训练数据图片</td>
  </tr>
  <tr>
    <td>http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz</td>
    <td>训练数据答案</td>
  </tr>
  <tr>
    <td>http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz</td>
    <td>测试数据图片
  </tr>
  <tr>
    <td>http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz</td>
    <td>测试数据答案</br>
</table>

下面的代码会给出如何载入MNIST数据集，如果指定地址下没有下载好的数据，那么tf会自动从上表中的地址下载<br>

```python
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/home/xuhejun/data/mnist/",one_hot=True)    

print("Training data size: ",mnist.train.num_examples)
```

代码说明：<br>

通过input_data.read_data_sets函数生成的类会自动将MNIST数据集划分为train、validation和test三个数据集<br>

其中train这个集合内有55000张图片，validation这个数据集中有5000张图片，test这个数据集中有10000张图片<br>

该项目总共分为3个.py文件，分别是mnist_inference.py、mnist_train.py、mnist_eval.py<br>

mnist_inference.py文件定义了前向传播得过程以及神经网络中的参数<br>

mnist_train.py文件定义了神经网络的训练过程<br>

mnist_eval.py文件定义测试过程<br>
