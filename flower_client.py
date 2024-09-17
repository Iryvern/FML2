from imports import *
from models import SparseAutoencoder
from training import train, get_parameters, set_parameters

class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, optimizer, scheduler, epochs_per_round):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_per_round = epochs_per_round  # Number of epochs per client per round

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        ndarrays = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Parameters retrieved")
        return fl.common.GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays)
        train(self.net, self.trainloader, epochs=self.epochs_per_round, optimizer=self.optimizer)
        updated_ndarrays = get_parameters(self.net)
        updated_parameters = ndarrays_to_parameters(updated_ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Model trained")
        return fl.common.FitRes(status=status, parameters=updated_parameters, num_examples=len(self.trainloader), metrics={})

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays)

        # Perform reconstruction and calculate SSIM using trainloader
        total_ssim = 0.0
        total_items = 0
        self.net.eval()
        with torch.no_grad():
            for images in self.trainloader:
                images = images.to(DEVICE)  # Ensure that images are moved to the correct device
                outputs = self.net(images)
                
                # Convert images and outputs to numpy arrays for SSIM calculation
                images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
                outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                
                for img, out in zip(images_np, outputs_np):
                    img_gray = TF.to_pil_image(img).convert("L")
                    out_gray = TF.to_pil_image(out).convert("L")
                    img_gray = np.array(img_gray)
                    out_gray = np.array(out_gray)
                    ssim_value = ssim(img_gray, out_gray, data_range=img_gray.max() - img_gray.min())
                    total_ssim += ssim_value
                total_items += images.size(0)

        average_ssim = total_ssim / total_items if total_items > 0 else 0

        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Evaluation completed"),
            loss=1 - average_ssim,  # Loss is 1 - SSIM to keep lower loss better
            num_examples=total_items,
            metrics={"ssim": average_ssim}  # Report SSIM
        )