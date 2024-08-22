from torch.cuda import get_arch_list
import os

def get_filtered_cuda_arch_list(min_arch = 70):
    
    raw_arch_list = get_arch_list()  
    filtered_arch_list = [arch for arch in raw_arch_list if arch.startswith("sm_") and int(arch.split("_")[1]) >= min_arch]
    arch_list_str =";".join([arch[3:-1]+"."+arch[-1] for arch in filtered_arch_list])

    return arch_list_str

if __name__=="__main__":
    print(get_filtered_cuda_arch_list()) 

