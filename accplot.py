# 打开文本文件并读取内容
with open(
        'record/txtsave/cifar10_resnet18_NL_0.4_LB_0.5_Iter_2_Rnd_100_100_ep_5_Frac_0.020_0.10_LR_0.020_ReR_0.5_ConT_0.5_ClT_0.1_Beta_0.0_Seed_13_nonIID_p_0.7_dirich_10.0_FT_CORR_acc.txt',
        'r') as file:
    lines = file.readlines()

# 创建一个空字典来存储每个客户端的准确率
client_acc = {}

# 遍历每一行文本并提取准确率数据
for line in lines:
    # 按空格分割每行文本，获取客户端编号和准确率
    parts = line.split(',')
    iteration = parts[0].split()[1]
    client = parts[1].split()[1]
    acc = float(parts[2].split()[2])

    # 将准确率数据存储到字典中
    if iteration not in client_acc:
        client_acc[iteration] = {}
    client_acc[iteration][client] = acc

# 打印提取的准确率数据
for iteration, clients in client_acc.items():
    print(f"Iteration {iteration}:")
    for client, acc in clients.items():
        print(f"Client {client}: Accuracy {acc}")
