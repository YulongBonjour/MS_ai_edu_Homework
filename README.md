# MS_ai_edu_Homework
**任务说明**

*非线性多分类器
鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。
任务：请利用该数据集训练出一个良好的非线性分类器。*

**提交内容：**
**MyModel.py**中借助numpy库，实现了Relu，sigmoid，linear，softmax，crossEntropy的前向传播和反向传播。

linear层的参数使用标准正态分布初始化

反向传播时，利用后一层的前向计算时输入变量的梯度作为本层反向传播函数的输入，得到本层相关变量的梯度并继续向前传递。
linear层需要对本层输入及本层的权重参数分别求导。
每层反向传播时返回的梯度是一个和要求梯度的变量形状相同的矩阵.

**Relu层**
```
def backward(self,dout):
        dout[self.mask]=0
        return dout 
```

**sigmoid层**
```
def backward(self,dout):
        return dout*self.y*(1-self.y)
```
**linear层**
```
def backward(self,dout):
            tmp1=np.zeros_like(self.wb,np.float64)#对权重的梯度
            tmp2=np.zeros((self.batch_size,self.in_dim),np.float64)#对上层输入的梯度
            for i in range(self.in_dim+1):
                for j in range(self.out_dim):
                    tmp1[i][j]+=np.dot(dout[:,j],self.x[:,i])
            for i in range(self.batch_size):
                for j in range(self.in_dim):
                    tmp2[i][j]+=np.dot(dout[i,:],self.wb[j,:])
            return (tmp1,tmp2)
```

**softmax层**
```
def backward(self,dout):
        tmp=np.zeros((self.batch_size,self.feature_dim),np.float64)
        for i in range(self.batch_size):
            for j in range(self.feature_dim):
                tmp[i][j]=self.y[i][j]*(dout[i][j]-np.dot(dout[i,:],self.y[i,:]))
        return tmp
```
**crossEntropy层**
```
def backward(self):
            tmp=np.zeros_like(self.estimation,np.float64) 
            for i in range(self.batch_size):
                for j in range(self.feature_dim):
                    tmp[i][j]=-self.target[i][j]/self.estimation[i][j]
            return tmp
 ```


**Iris_train.py** 中用上面的功能，搭建了一个使用sigmoid做激活函数，具有三个线性层和一个softmax层，损失函数选用交叉熵的多分类神经网络。
优化算法采用批量梯度下降。自定义的Net类中可实现整个网络的前向、反向、损失函数的计算和参数的自动更新

**训练过程中**准确率和损失函数的变化趋势如下：


![pic](https://github.com/yulong-XJTU/MS_ai_edu_Homework/blob/iris_classification/acc_loss.PNG)

**测试集**上准确率在90%以上
