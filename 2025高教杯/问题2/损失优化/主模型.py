import torch
import torch.nn as nn

from 激活函数 import activate1,activate2,activate3

class model(nn.Module):
    def __init__(self,alpha = 10): #alpha 为检测天数的重要性，默认值为1
        super().__init__()
        self.activate1 = activate1()
        self.activate2 = activate2()
        self.activate3 = activate3()
        self.alpha = alpha

        self.mlp = nn.Sequential(
            nn.Linear(2,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        
    def forward(self, x, mode='train'):
        # (5),[BMI,18号染色体的Z值,年龄,身高,在参考基因组上比对的比例,检测孕周_天数]
        if mode == 'mlp': #多层感知机用于预测Y染色体浓度
            x = self.mlp(x)
            return x
        else:
            day = self.activate1(x[0])
            # 原错误代码：x[-1] = day
            # 替换为非原地操作
            x = torch.cat([x[:-1], day.unsqueeze(0)])
            
            # 继续MLP处理
            x = self.mlp(x)
            x = self.activate3(x)
            day = self.activate2(day)

            return x + self.alpha * day + self.activate1.parameter_loss() #预测总损失

if __name__ == '__main__':
    x = torch.randn(3)
    print(x.shape)
    model = model()
    print(model(x,'mlp').shape)
    print(model(x,'loss').shape)
        



        

