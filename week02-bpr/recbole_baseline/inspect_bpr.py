import inspect
from recbole.model.general_recommender.bpr import BPR
from recbole.model.loss import BPRLoss
from recbole.trainer import Trainer

# 查找三个核心函数的源码路径
print(inspect.getfile(BPR))
print(inspect.getfile(BPRLoss))
print(inspect.getfile(Trainer))

# 显示训练相关源码
print(inspect.getsource(BPR.calculate_loss))
print(inspect.getsource(BPR.full_sort_predict))
print(inspect.getsource(BPRLoss.forward))