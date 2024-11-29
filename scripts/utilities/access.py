from sklearn.metrics import classification_report
import torch
from models.dog_behavior.stgcn_model import STGCN
from models.dog_behavior.data_loader import DogBehaviorDataset
from config.config import config

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for keypoints, labels in test_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            outputs = model(keypoints)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    test_dataset = DogBehaviorDataset(config.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = STGCN(num_keypoints=config.num_joints, num_classes=6)
    model.load_state_dict(torch.load("outputs/checkpoints/model.pth"))  # 모델 경로 지정
    test_model(model, test_loader)
