import sys
import os

print("=== Python Environment ===")
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)

try:
    import torch
    print("\n=== PyTorch Info ===")
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Supported Architectures in Binary:", torch.cuda.get_arch_list())
        
        # Test allocation
        print("\n=== Testing Tensor Allocation on GPU ===")
        try:
            x = torch.tensor([1.0, 2.0], device="cuda")
            print("Successfully allocated tensor on CUDA:", x)
            y = x * 2.0
            print("Successfully performed tensor operation on CUDA:", y)
        except Exception as e:
            print("Error during CUDA tensor operation:", e)
    else:
        print("CUDA is NOT available in this PyTorch installation.")
except ImportError:
    print("\nPyTorch is not installed in this environment.")
