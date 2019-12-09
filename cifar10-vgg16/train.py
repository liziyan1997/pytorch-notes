import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
 
import os
import argparse
 
from tensorboardX import SummaryWriter
 
from vgg16 import VGG16
from torch.autograd import Variable
 
#参数设置
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--lr', default=1e-2,help='learning rate')
#parser.add_argument('--batch_size',default=50,help='batch size')
parser.add_argument('--epoch',default=15,help='time for ergodic')
parser.add_argument('--pre_epoch',default=0,help='begin epoch')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--pre_model', default=False,help='use pre-model')#恢复训练时的模型路径
args = parser.parse_args()
 
#使用gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
 
#数据预处理
# 图像预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
#Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
#模型定义 VGG16
net = VGG16().to(device)
 
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss() #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
 
#使用预训练模型
if args.pre_model:
    print("Resume from checkpoint...")
    assert os.path.isdir('checkpoint'),'Error: no checkpoint directory found'
    state = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(state['state_dict'])
    best_test_acc = state['acc']
    pre_epoch = state['epoch']
else:
    #定义最优的测试准确率
    best_test_acc = 0
    pre_epoch = args.pre_epoch

#训练
if __name__ == "__main__":
    if not os.path.exists('./model'):
        os.mkdir('./model')
    writer = SummaryWriter(log_dir='./log')

    

    print("Start Training, VGG-16...")
    with open("acc.txt","w") as acc_f:
        with open("log.txt","w") as log_f:
            for epoch in range(pre_epoch, args.epoch):
                print('\nEpoch: %d' % (epoch + 1))
                #开始训练
                net.train()
                print(net)
                #总损失
                sum_loss = 0.0
                #准确率
                accuracy = 0.0
                total = 0.0
 
                for i, data in enumerate(trainloader):
                    #准备数据
                    length = len(trainloader) #数据大小
                    inputs, labels = data #取出数据
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad() #梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
                    inputs, labels = Variable(inputs), Variable(labels)
                    #forward + backward + optimize
                    outputs = net(inputs) #前向传播求出预测值
                    loss = criterion(outputs, labels) #求loss
                    loss.backward() #反向传播求梯度
                    optimizer.step() #更新参数
 
                    # 每一个batch输出对应的损失loss和准确率accuracy
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)#返回每一行中最大值的那个元素，且返回其索引
                    total += labels.size(0)
                    accuracy += predicted.eq(labels.data).cpu().sum() #预测值和真实值进行比较，将数据放到cpu上并且求和
 
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * accuracy / total))
 
                    #写入日志
                    log_f.write('[epoch:%d, iter:%d] |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * accuracy / total))
                    log_f.write('\n')
                    log_f.flush()
 
                #写入tensorboard
                writer.add_scalar('loss/train',sum_loss / (i + 1),epoch)
                writer.add_scalar('accuracy/train',100. * accuracy / total,epoch)
                #每一个训练epoch完成测试准确率
                print("Waiting for test...")
                #在上下文环境中切断梯度计算，在此模式下，每一步的计算结果中requires_grad都是False，即使input设置为requires_grad=True
                with torch.no_grad():
                    accuracy = 0
                    total = 0
                    for data in testloader:
                        #开始测试
                        net.eval()
 
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
 
                        outputs = net(images)
 
                        _, predicted = torch.max(outputs.data, 1)#返回每一行中最大值的那个元素，且返回其索引(得分高的那一类)
                        total += labels.size(0)
                        accuracy += (predicted == labels).sum()
 
                    #输出测试准确率
                    print('测试准确率为: %.3f%%' % (100 * accuracy / total))
                    acc = 100. * accuracy / total
                    
                    #写入tensorboard
                    writer.add_scalar('accuracy/test', acc,epoch)
                    
                    #将测试结果写入文件
                    print('Saving model...')
                    torch.save(net.state_dict(), '%s/net_%3d.pth' % (args.outf, epoch + 1))
                    acc_f.write("epoch = %03d, accuracy = %.3f%%" % (epoch + 1, acc))
                    acc_f.write('\n')
                    acc_f.flush()
 
                    #记录最佳的测试准确率
                    if acc > best_test_acc:
                        print('Saving Best Model...')
                        #存储状态
                        state = {
                            'state_dict': net.state_dict(),
                            'acc': acc,
                            'epoch': epoch + 1,
                        }
                        #没有就创建checkpoint文件夹
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        #best_acc_f = open("best_acc.txt","w")
                        #best_acc_f.write("epoch = %03d, accuracy = %.3f%%" % (epoch + 1, acc))
                        #best_acc_f.close()
                        torch.save(state, './checkpoint/ckpt.t7')
                        best_test_acc = acc
                        #写入tensorboard
                        writer.add_scalar('best_accuracy/test', best_test_acc,epoch)
            
            #训练结束
            print("Training Finished, Total Epoch = %d" % epoch)
            writer.close()
 
 
 