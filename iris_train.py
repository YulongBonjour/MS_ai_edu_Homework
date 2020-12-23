#Mymodel 定义了神经网络各层的前向传播函数和反向传播函数
import MyModel as md
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("./original_data.csv")
os.getcwd()
#划分数据集
data=data.sample(frac=1)#打乱顺序
data[["sepal length","sepal width","petal length","petal width"]].iloc[:100,:].to_csv("train_data.csv",index=False)
data[["sepal length","sepal width","petal length","petal width"]].iloc[100:,:].to_csv("test_data.csv",index=False)

#把标签转化为onehot
tgt=[]
for i in range(150):
    if data["class"].iloc[i]=="Iris-setosa":
         tgt.append([1,0,0])
    elif data["class"].iloc[i]=="Iris-versicolor":
         tgt.append([0,1,0])
    else:
         tgt.append([0,0,1])
tgt=pd.DataFrame(tgt)
tgt[:100].to_csv("train_target.csv",header=False,index=False)
tgt[100:].to_csv("test_target.csv",header=False,index=False) 

#搭建一个有三个隐藏层的神经网络，激活函数用sigmoid 
class Net():
    def __init__(self,batch_size,lr):
        self.lr=lr
        self.linear1=md.linear(batch_size,4,8)
       # self.Relu1=Relu()
        self.sigmoid1=md.sigmoid()
        self.linear2=md.linear(batch_size,8,8)
        #self.Relu2=Relu()
        self.sigmoid2=md.sigmoid()
        self.linear3=md.linear(batch_size,8,3)
        self.softmax=md.softmax()
        self.lossF=md.crossEntropy()
        self.out=None
    def forward(self,x):
        out=self.linear1.forward(x)
        #out=self.Relu1.forward(out)
        out=self.sigmoid1.forward(out)
        out=self.linear2.forward(out)
        #out=self.Relu2.forward(out)
        out=self.sigmoid2.forward(out)
        out=self.linear3.forward(out)
        out=self.softmax.forward(out)
        self.out=out
        idx=np.argmax(out,axis=1)#找到预测值
        return idx 
    def  get_loss(self,target):
         return self.lossF.forward(self.out,target)#计算损失函数
    def backward(self):#反向传播求梯度，并更新参数的值
        dout=self.lossF.backward()
        dout=self.softmax.backward(dout)
        
        (dout1,dout2)=self.linear3.backward(dout)
        
        self.linear3.wb-=self.lr*dout1#更新 linear3的权重
        #dout=self.Relu2.backward(dout2)
        dout=self.sigmoid2.backward(dout2)
      
        (dout1,dout2)=self.linear2.backward(dout)#分别得到对本层权重的梯度和对来自上层的输入的梯度
        
        self.linear2.wb-=self.lr*dout1#更新 linear2的权重
        
        #dout=self.Relu1.backward(dout2)
        dout=self.sigmoid1.backward(dout2)
        
        (dout1,dout2)=self.linear1.backward(dout)
        self.linear1.wb-=self.lr*dout1#更新 linear1的权重

Batch_size=10
Lr=0.01
net=Net(Batch_size,Lr)

#训练模型
train_data=pd.read_csv("train_data.csv")
train_target=pd.read_csv("train_target.csv",header=None)
Len=train_data.values.shape[0]
history_loss=[]
history_acc=[]
#print(Len//Batch_size)
for epoch in range(200):#200轮
    acc=0
    for batch in range(Len//Batch_size):#分批训练
        d=train_data[Batch_size*(batch):Batch_size*(batch+1)].values
        target=train_target[Batch_size*(batch):Batch_size*(batch+1)].values
        predict=net.forward(d)#前向
        loss=net.get_loss(target)#计算loss
        net.backward()#反向传播

        acc+=(predict==np.argmax(target,1)).sum()#统计准确率
    history_acc.append(acc/Len)
    history_loss.append(loss)
print("训练准确率为",acc/Len)
plt.subplot(1,2,1)
plt.plot(history_acc)  
plt.subplot(1,2,2)
plt.plot(history_loss)  


#测试模型准确率
test_data=pd.read_csv("test_data.csv")
test_target=pd.read_csv("test_target.csv",header=None)
Len=test_data.values.shape[0]
acc=0
print("results of prediction")
for batch in range(Len//Batch_size):
        d=test_data[Batch_size*(batch):Batch_size*(batch+1)].values
        target=test_target[Batch_size*(batch):Batch_size*(batch+1)].values
        predict=net.forward(d)
        print(predict)
        acc+=(predict==np.argmax(target,1)).sum()
print("测试准确率为",acc/Len)
