import numpy as np

i1=0.05#输入神经元1
i2=0.10#输入神经元2
b1=0.35#截距项1
b2=0.60#截距项2

w1=0.15
w2=0.20
w3=0.25
w4=0.30
w5=0.40
w6=0.45
w7=0.50
w8=0.55

target_o1=0.01
target_o2=0.99

learn_rate=0.5

#sigmoid函数的实现
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

net_h1=i1*w1+i2*w2+b1
net_h2=i1*w3+i2*w4+b1

out_h1=sigmoid(net_h1)
out_h2=sigmoid(net_h2)

net_o1=out_h1*w5+out_h2*w6+b2
net_o2=out_h1*w7+out_h2*w8+b2

out_o1=sigmoid(net_o1)
out_o2=sigmoid(net_o2)

print("初始输出：",out_o1,out_o2)

def E_total():
    return (np.square(target_o1-out_o1)+np.square(target_o2-out_o2))*0.5
#输出层误差对权重的偏导
def Etotal_w5678(target,out_o,out_h):
    return -(target-out_o)*out_o*(1-out_o)*out_h

#输出层误差
def E_out(target,out_o):
    return -(target-out_o)*out_o*(1-out_o)

#隐含层-隐含层对权重的偏导
def Etotal_w1234(target_1,out_o1,w_ho1,target_2,out_o2,w_ho2,out_h,i):
    return ((E_out(target_1,out_o1))*w_ho1+(E_out(target_2,out_o2))*w_ho2)*out_h*(1-out_h)*i


for i in range(100000):
    # print(w1)
    # print((E_out(target_o1,out_o1))*w5)
    # print((E_out(target_o2,out_o2))*w7)
    #对应分别是期望输出1，输出1，输出1和h1的权重，期望输出2，输出2，输出2和h1的权重，h1的输出，输入i1
    w1=w1-learn_rate*Etotal_w1234(target_o1,out_o1,w5,target_o2,out_o2,w7,out_h1,i1)
    #print(w1)
    w2=w2-learn_rate*Etotal_w1234(target_o1,out_o1,w5,target_o2,out_o2,w7,out_h1,i2)
    w3=w3-learn_rate*Etotal_w1234(target_o1,out_o1,w6,target_o2,out_o2,w8,out_h2,i1)
    w4=w4-learn_rate*Etotal_w1234(target_o1,out_o1,w6,target_o2,out_o2,w8,out_h2,i2)

    #print(w5)
    w5=w5-learn_rate*Etotal_w5678(target_o1,out_o1,out_h1)
    #print(w5)
    w6=w6-learn_rate*Etotal_w5678(target_o1,out_o1,out_h2)
    w7=w7-learn_rate*Etotal_w5678(target_o2,out_o2,out_h1)
    w8=w8-learn_rate*Etotal_w5678(target_o2,out_o2,out_h2)

    net_h1=i1*w1+i2*w2+b1
    net_h2=i1*w3+i2*w4+b1

    out_h1=sigmoid(net_h1)
    out_h2=sigmoid(net_h2)

    net_o1=out_h1*w5+out_h2*w6+b2
    net_o2=out_h1*w7+out_h2*w8+b2

    out_o1=sigmoid(net_o1)
    out_o2=sigmoid(net_o2)
    if(i%10000==0):
        print("第{}次反向传播后，误差为{}".format(i,E_total()))
print("最终输出：",out_o1,out_o2)
print("目标输出：",target_o1,target_o2)
print("偏差值为：",target_o1-out_o1,target_o2-out_o2)
# print(w1)
# print(w2)
# print(w3)
# print(w4)
# print(w5)
# print(w6)
# print(w7)
# print(w8)
