import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

class PerformancePipline:
    def __init__(self, model_path1, model_path2, dataset_root, batch_size=1):
        self.model1 = self.load_model(model_path1)
        self.model2 = self.load_model(model_path2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.transform = transforms.Compose([ # TODO: turn this into a constructor parameter if we want to custom-define our own transforms
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.dataloader = self.create_dataloader()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model.eval()
        return model

    def create_dataloader(self, which_dataset='culane', split='val'):
        # dataset = load dataset, in same way as we do for training (see dummy train)
        # TODO: run parameter checks: which dataset, split, etc.
        dataset = []
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def compare_models(self):
        model1_results = []
        model2_results = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                #TODO: we could have random perturbations here, to see which model is more robust

                output1 = self.model1(inputs)
                output2 = self.model2(inputs)

                model1_results.append(output1.cpu().numpy())
                model2_results.append(output2.cpu().numpy())

        return model1_results, model2_results
    
    #TODO: evaluation metrics to call from above

if __name__ == "__main__":
    model_path1 = "model_checkpoint1.pth"  # Update with your checkpoint file paths
    model_path2 = "model_checkpoint2.pth"  # Update with your checkpoint file paths
    dataset_root = "/path/to/dataset"  # culane or tusimple.
    batch_size = 1  # Adjust batch size as needed
    #TODO: any other parameters to pass in?

    comparer = PerformancePipline(model_path1, model_path2, dataset_root, batch_size)
    model1_results, model2_results = comparer.compare_models()

    # Now you can analyze and compare the segmentation results as needed
    # For example, you can calculate metrics like IoU, F1-score, etc., to evaluate performance.
