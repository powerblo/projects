import json
import torch
import matplotlib.pyplot as plt
    
if __name__ == '__main__':
    check = 7

    hml_test = torch.load(f'hml_test_{check}.pth')
    route_test = torch.load(f'route_test_{check}.pth')
    loss_graph = torch.load(f'loss_graph_{check}.pth')
    length_graph = torch.load(f'length_graph_{check}.pth')

    print(route_test)
    print('max length : ', route_test.shape[2])
    print('average length : ', route_test.shape[2]*torch.mean(torch.min(torch.ones_like(route_test),route_test.to(torch.float32))).item())

    size = 100

    plt.figure()
    plt.plot(range(loss_graph.shape[0]-size+1), loss_graph.unfold(0,size,1).mean(dim=1))
    plt.show()

    plt.figure()
    plt.plot(range(length_graph.shape[0]-size+1), length_graph.unfold(0,size,1).mean(dim=1))
    plt.show()