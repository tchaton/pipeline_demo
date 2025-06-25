# serve.py
import os
import torch
import litserve as ls

# Import the model class to load the checkpoint
from model import LitLinearRegression

pipeline_shared_path = os.getenv('PIPELINE_SHARED_PATH', "")
checkpoint_path = os.path.join(pipeline_shared_path, "model.ckpt") if pipeline_shared_path else "model.ckpt"

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        """Load the model from the checkpoint."""
        self.model = LitLinearRegression.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        print(f"Model '{checkpoint_path}' loaded and moved to device: {device}")

    def decode_request(self, request: dict) -> torch.Tensor:
        """Convert JSON request to a tensor."""
        # We expect a request like: {"x": 5.0}
        return torch.tensor([[request["x"]]], dtype=torch.float32)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""
        return self.model(x)

    def encode_response(self, output_tensor: torch.Tensor) -> dict:
        """Convert the model's output tensor to a JSON response."""
        # We'll send back a response like: {"y_pred": 11.023}
        return {"y_pred": output_tensor.item()}


if __name__ == "__main__":
    # Check if the model checkpoint exists before starting the server
    if not os.path.exists(checkpoint_path):
        print("Error: Could not find 'model.ckpt'.")
        print("Please run 'python train.py' first to train and save the model.")
        exit(1)
        
    print("--- Starting LitServe server ---")
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)