import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    def __init__(self, num_experts, intput_size, output_size):
        super(MoE, self).__init__()
        
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(self.num_experts)])
        self.gating_network = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        
        gating_scores = F.softmax(self.gating_network(x), dim = 1) # [Batchsize, num_experts]
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim = 1) # [Batchsize, num_experts, output_size]
        
        moe_output = torch.bmm(gating_scores.unsqueeze(1), expert_outputs).squeeze(1) # [Batchsize, output_size]
        return moe_output

if __name__ == "__main__":
    
    input_size = 8
    output_size = 64
    num_experts = 4
    
    
    moe_model = MoE(num_experts, input_size, output_size)
    
    
    batchsize = 2
    input = torch.randn(batchsize, input_size)
    
    
    output = moe_model(input)
    print("output.shape: ", output.shape) # [batchsize, output_size]
        