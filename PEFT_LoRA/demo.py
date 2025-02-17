import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, LoraModel
from peft.utils import get_peft_model_state_dict 


class Simple_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 256)
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
   
    origin_model = Simple_Model()

    
    model_lora_config = LoraConfig(
        r = 32, 
        lora_alpha = 32, 
        init_lora_weights = "gaussian", 
        target_modules = ["linear1", "linear2"], 
        lora_dropout = 0.1
    )

    # Test data
    input_data = torch.rand(2, 64)
    origin_output = origin_model(input_data)

    
    origin_state_dict = origin_model.state_dict() 

    
    new_model1 = get_peft_model(origin_model, model_lora_config)
    new_model2 = LoraModel(origin_model, model_lora_config, "default")

    output1 = new_model1(input_data)
    output2 = new_model2(input_data)
    
    # origin_output == output1 == output2

    
    new_model1_lora_state_dict = get_peft_model_state_dict(new_model1)
    new_model2_lora_state_dict = get_peft_model_state_dict(new_model2)

    # origin_state_dict['linear1.weight'].shape -> [output_dim, input_dim]
    # new_model1_lora_state_dict['base_model.model.linear1.lora_A.weight'].shape -> [r, input_dim]
    # new_model1_lora_state_dict['base_model.model.linear1.lora_B.weight'].shape -> [output_dim, r]

    
    from diffusers import StableDiffusionPipeline
    save_path = "./"
    global_step = 0
    StableDiffusionPipeline.save_lora_weights(
            save_directory = save_path,
            unet_lora_layers = new_model1_lora_state_dict,
            safe_serialization = True,
            weight_name = f"checkpoint-{global_step}.safetensors",
        )

    
    from safetensors import safe_open
    alpha = 1. 
    lora_path = "./" + f"checkpoint-{global_step}.safetensors"
    state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    all_lora_weights = []
    for idx,key in enumerate(state_dict):
        # only process lora down key
        if "lora_B." in key: continue

        up_key    = key.replace(".lora_A.", ".lora_B.") 
        model_key = key.replace("unet.", "").replace("lora_A.", "").replace("lora_B.", "")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = new_model1

        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key].to(curr_layer.weight.data.device)
        weight_up   = state_dict[up_key].to(curr_layer.weight.data.device)
        
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)
        all_lora_weights.append([model_key, torch.mm(weight_up, weight_down).t()])
        print('Load Lora Done')

    print("All Done!")
