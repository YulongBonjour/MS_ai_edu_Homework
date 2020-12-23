import numpy as np
class Relu() : 
    def __init__(self):
        self.mask=None
    def forward(self,x): 
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        return dout
      
class sigmoid():
    def __init__(self):
        self.y=None
    def forward(self,x):
        self.y=1/(1+np.exp(-x))
        return self.y
    def backward(self,dout):
        return dout*self.y*(1-self.y)
    
    
class linear():
    def __init__(self, batch_size,in_dim,out_dim):
            self.x=None
            self.batch_size=batch_size
            self.in_dim=in_dim
            self.out_dim=out_dim
            np.random.seed(111)
            self.wb=np.random.randn(in_dim+1,out_dim)
    def forward(self,x):
            b=np.ones((x.shape[0],1))
            new_x=np.concatenate((x,b),axis=1)
            self.x=new_x
            self.y=np.matmul(new_x,self.wb)
            return self.y
    def backward(self,dout):
            tmp1=np.zeros_like(self.wb,np.float64)
            tmp2=np.zeros((self.batch_size,self.in_dim),np.float64)
            for i in range(self.in_dim+1):
                for j in range(self.out_dim):
                    tmp1[i][j]+=np.dot(dout[:,j],self.x[:,i])
            for i in range(self.batch_size):
                for j in range(self.in_dim):
                    tmp2[i][j]+=np.dot(dout[i,:],self.wb[j,:])
            return (tmp1,tmp2)
    
            
class softmax():
    def __init__(self):
        self.y=None
        self.batch_size=None
        self.feature_dim=None
    def forward(self,x):
        self.batch_size=x.shape[0]
        self.feature_dim=x.shape[1]
        self.y=np.exp(x)
        for i in range(self.batch_size):
            sum=0
            for j in range(self.feature_dim):
                sum+=self.y[i][j]
            for j in range(self.feature_dim):
                 self.y[i][j]=self.y[i][j]/sum
        return  self.y
    
    def backward(self,dout):
        tmp=np.zeros((self.batch_size,self.feature_dim),np.float64)
        for i in range(self.batch_size):
            for j in range(self.feature_dim):
                tmp[i][j]=self.y[i][j]*(dout[i][j]-np.dot(dout[i,:],self.y[i,:]))
        return tmp
    
class crossEntropy():
        def __init__(self):
            self.batch_size=None
            self.feature_dim=None
            self.estimation=None
            self.target=None
        def forward(self,estimation,target):
            self.batch_size=estimation.shape[0]
            self.feature_dim=estimation.shape[1]
            self.estimation=estimation
            self.target=target
            Loss=0
            for i in range(self.batch_size):
                for j in range(self.feature_dim):
                    Loss+=-target[i][j]*np.log(estimation[i][j])
            return Loss
        def backward(self):
            tmp=np.zeros_like(self.estimation,np.float64) 
            for i in range(self.batch_size):
                for j in range(self.feature_dim):
                    tmp[i][j]=-self.target[i][j]/self.estimation[i][j]
            return tmp
        
      

        
      
 
        
