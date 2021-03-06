import torch
import torch.nn.functional as fn


class TSA_crossEntropy(object):
    def __init__(self, num_steps, num_classes, alpha, temperature, device='cpu'):
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_steps = num_steps
        self.current_step = 0
        self.num_classes = num_classes
        self.alpha = alpha
        self.device = device
        self.temperature = temperature
    
    def thresh(self):
        thresh = torch.exp(torch.Tensor([(self.current_step/self.num_steps-1)*self.alpha]))*(1-1/self.num_classes)+1/self.num_classes
        
        thresh = thresh.to(self.device)
        return thresh

    def step(self):
        self.current_step += 1

    def get_mask(self, logits, targets):
        thresh = self.thresh()
        mask = fn.softmax(logits, dim = 1).detach()
        mask, pred = torch.max(mask, dim = 1, keepdim = False)

        wrong_pred = (torch.abs(pred - targets)>0)
        mask = (mask<thresh)
        mask |= wrong_pred

       
        return mask.type(torch.FloatTensor).to(self.device)


    def __call__(self, logits, targets):
        logits = logits/self.temperature

        mask = self.get_mask(logits, targets)
        self.loss_value = self.loss_function(logits, targets)*mask
        self.loss_value = self.loss_value.sum()/torch.max(torch.Tensor([mask.sum(),1]))
        self.step()
        return self.loss_value

def test(): 
    # Test TSA with crossentropy loss
    loss = TSA_crossEntropy(10000,2,5, 5)

    inputs = torch.FloatTensor([[11,10],[200,151],[501,400],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[1,10],[9,20],[9,100],[100,101],[1111,1112]])
    weight = torch.FloatTensor([[0,0],[0,0]]).requires_grad_(requires_grad=True)
    bias = torch.FloatTensor([[0,0]]).requires_grad_(requires_grad=True)
    targets = torch.LongTensor([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

    for i in range(10000):
        logits = torch.matmul(inputs, weight)
        logits = logits+bias
        loss_value = loss(logits, targets)

        loss_value.backward()

        optimizer = torch.optim.SGD([weight, bias], lr=0.01, momentum=0.9)

        optimizer.step()

    # print(logits)

    # inputs = torch.FloatTensor([[200,151],[501,400],[9,8],[99,95],[100,100000],[99,60]])

    # logits = torch.matmul(inputs, weight)
    logits = torch.nn.functional.softmax(logits, dim = 1)
    out,_ = torch.max(logits, dim =1)
    print('with TSA training')
    print('weights:', weight)
    print('bias:', bias)
    print('output:', logits)


    # without TSA 

    loss = torch.nn.CrossEntropyLoss()

    inputs = torch.FloatTensor([[11,10],[200,151],[501,400],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[1,10],[9,20],[9,100],[100,101],[1111,1112]])
    weight = torch.FloatTensor([[0,0],[0,0]]).requires_grad_(requires_grad=True)
    bias = torch.FloatTensor([[0,0]]).requires_grad_(requires_grad=True)
    targets = torch.LongTensor([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])


    for i in range(10000):
        logits = torch.matmul(inputs, weight)
        logits = logits+bias
        loss_value = loss(logits, targets)

        loss_value.backward()

        optimizer = torch.optim.SGD([weight, bias], lr=0.01, momentum=0.9)

        optimizer.step()

    # inputs = torch.FloatTensor([[200,151],[501,400],[9,8],[99,95],[100,100000],[99,60]])
    # logits = torch.matmul(inputs, weight)
    logits = torch.nn.functional.softmax(logits, dim = 1)
    out,_ = torch.max(logits, dim =1)
    print('without TSA training')
    print('weights:', weight)
    print('bias:', bias)
    print('output:', logits)