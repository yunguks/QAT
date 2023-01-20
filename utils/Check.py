import torch
import time
import numpy as np
import random

__all__ = ["memory_check","time_test", "model_eq_check", "set_random_seeds"]


def memory_check():
    """_summary_
        Allocated GPU 
    """
    print(f"  Allocated: {round(torch.cuda.memory_allocated()/1024**3,2)} GB")
    print(f"  Cached:    {round(torch.cuda.memory_reserved()/1024**3,2)} GB\n")
    
def time_test(model, device, input_size = (1,3,256,256),num_tests=100,):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()

        for _ in range(num_tests):
            _ = model(x)
            torch.cuda.synchronize()
        total_time = time.time() - start_time

    aver_time = total_time / num_tests
    return total_time, aver_time

def model_eq_check(model1, model2, device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)):

    model1.to(device)
    model2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model1(x).detach().cpu().numpy()
        y2 = model2(x).detach().cpu().numpy()
        # 배열이 허용 오차범위 abs(a - b) <= (atol + rtol * absolute(b)) 이내면 True
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test fail")
            return False
    print("Two models equal")
    return True

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)