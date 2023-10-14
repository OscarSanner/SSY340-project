if __name__ == '__main__':

    import torch
    import glob

    from fastai.vision.learner import create_body
    from torchvision.models.resnet import resnet18
    from fastai.vision.models.unet import DynamicUnet

    from models import MainModel
    from dataset import make_dataloaders
    from utils import visualize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}")

    paths = glob.glob("/Users/osann/Documents/GitProjects/SSY340-project/src/test_colorize/*.jpg")
    print("Creating dataloader...")
    dl = make_dataloaders(paths=paths, split='val')

    def build_res_unet(n_input=1, n_output=2, size=256):
        body = create_body(resnet18(pretrained=True), n_in=n_input, cut=-2)
        #body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
        net_G = DynamicUnet(body, n_output, (size, size)).to(device)
        return net_G

    print("Building resnet...")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    print("Loading resnet weights...")
    net_G.load_state_dict(torch.load("checkpoints/res18-unet.pt", map_location=device))
    print("Building final model...")
    model = MainModel(net_G=net_G)
    print("Loading final model weights...")
    model.load_state_dict(torch.load("checkpoints/final_model_weights.pt", map_location=device))

    print("Loading data...")
    data = next(iter(dl))
    print("Colorizing...")
    visualize(model, data, save=True)