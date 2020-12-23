# MS_ai_edu_Homework
提交内容：
MyModel.py中借助numpy库，实现了Relu，sigmoid，linear，softmax，crossEntropy的前向传播和反向传播。
linear层的参数使用标准正态分布初始化
反向传播时，利用后一层的前向计算时输入变量的梯度作为本层反向传播函数的输入，得到本层相关变量的梯度并继续向前传递。
linear层需要对本层输入及本层的权重参数分别求导。




Iris_train.py 中用上面的功能，搭建了一个使用sigmoid做激活函数，具有三个线性层和一个softmax层，损失函数选用交叉熵的多分类神经网络。
优化算法采用批量梯度下降。自定义的Net类中可实现整个网络的前向、反向、损失函数的计算

训练过程中准确率和损失函数的变化趋势如下：


![pic](https://github.com/yulong-XJTU/MS_ai_edu_Homework/blob/iris_classification/acc_loss.PNG)

测试集上准确率在90%以上
