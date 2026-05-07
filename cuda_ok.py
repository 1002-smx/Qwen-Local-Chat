import torch

'''
此程序用于检查该torch模块CUDA是否可用
'''

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    arch_list = torch.cuda.get_arch_list()
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"支持的架构：{arch_list}")
    if "sm_120" in str(arch_list):
        print("SM_120架构支持：True")
    else:
        print("SM_120架构支持：False")
else:
    print("提示：CUDA不可用，请检查安装")
