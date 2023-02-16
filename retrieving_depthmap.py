import numpy
from models.mvsnet import MVSNet
from datasets.diligent import DiligentDataset
from datasets.zhao import MVSDataset
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def toImage(obj, lighting, counter, tensor):
    plt.imshow(tensor, vmin = 1450, vmax = 1600)
    plt.colorbar()
    plt.savefig(f"./result/{obj}_view{counter}_lighting{lighting}.png")
    plt.close()

def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


if __name__ == "__main__":
    # some testing code, just IGNORE it
    list_obj = ["bear", "buddha", "cow", "pot2", "reading"]
    torch.cuda.set_device(0)

    model = MVSNet(refine=False)
    model = torch.nn.DataParallel(model)
    model.cuda()
    state_dict = torch.load("/playpen-nas-ssd/xiaolong/project/MVSNet_pytorch/datasets/model_000014.ckpt")
    model.load_state_dict(state_dict['model'])
    model.eval()
    for obj_index in range(5):
        for lighting in range(96):
            obj = list_obj[obj_index]
            dataset = DiligentDataset(obj = obj, lighting = lighting)
            test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 1)    
            counter = 1
            for item in test_loader:
                    output = model(item["imgs"].float(), item["proj_matrices"].float(), item["depth_values"].float())
                    output["depth"] = output["depth"].reshape((1,1,128,152))
                    output["depth"] = torch.nn.functional.interpolate(output["depth"], scale_factor = 2, mode = "bilinear")
                    output["depth"] = torch.nn.functional.interpolate(output["depth"], scale_factor = 2, mode = "bilinear")
                    output = tensor2numpy(output)
                    output["depth"] = np.multiply(output["depth"][0, 0, :, :], item["mask"][0][0, :, :, 0])
                    torch.save(output, f"./result/{obj}_view{counter}_lighting{lighting}.pt")
                    toImage(obj, lighting, counter,  output["depth"])
                    counter += 1
    # dataset = MVSDataset(
    #     root_dir="/playpen-nas-ssd/xiaolong/project/MVSNet_pytorch/training_data/DiLiGenT-MV/mvpmsData",
    #     split='train',
    #     nviews=5,
    # )


    # for item in dataset:
    #     print(item["proj_matrices"].shape)
    #     output = model(torch.reshape(torch.tensor(item["imgs"]), (1, 5, 3, 512, 512)),torch.reshape(torch.tensor(item["proj_matrices"]), (1, 5, 4, 4)), torch.reshape(torch.tensor(item["depth_values"]), (1, 192)))
    #     print(output)
    #     torch.save(output, f"1.pt")
    #     toImage("derp", 1, output["depth"])
  

