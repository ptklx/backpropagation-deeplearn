import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivation(y):
    return y * (1 - y)
    
if __name__ == '__main__':
    # 初始化参数
    alpha = 0.05  # 学习速率
    # 输入与目标输出
    input_x = np.array([[0.1, 0.2, 0.3]])
    target_y = np.array([[0.02, 0.84]])
    # 初始权重系数,输入层三个单元，一个隐藏层两个单元，输出层两个单元
    input_dim = 3; hidden_dim = 2; output_dim = 2
    w1 = np.random.random([hidden_dim, input_dim])
    w2 = np.random.random([output_dim, hidden_dim])
    b1 = 0.1; b2 = 0.2
    inner = 10000
    
    # 迭代，反向传播更新参数
    for i in range(inner):
        in1 = np.dot(input_x, w1.T) + b1
        out1 = sigmoid(in1)
        in2 = np.dot(out1, w2.T) + b2
        out2 = sigmoid(in2)
        error = 1 / output_dim * (target_y - out2)**2
        # 向量化
        delta2 = -2 / output_dim * np.multiply(target_y - out2, derivation(out2))
        w2 = w2 - np.dot(alpha * delta2.T, out1)
        
        temp1 = -2 / output_dim * np.multiply(target_y - out2, derivation(out2))
        delta1 = np.multiply(np.dot(temp1, w2), derivation(out1))
        w1 = w1 - np.dot(alpha * delta1.T, input_x)

    print(w1)
    print(w2)
