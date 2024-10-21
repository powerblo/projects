import json
import torch
    
if __name__ == '__main__':
    hml_test = torch.load('hml_test_6.pth')
    route_test = torch.load('route_test_6.pth')

    print(hml_test)
    print('max length : ', route_test.shape[2])
    print('average length : ', route_test.shape[2]*torch.mean(torch.min(torch.ones_like(route_test),route_test.to(torch.float32))).item())