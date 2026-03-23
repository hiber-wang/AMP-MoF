from detectors.deep_autoencoder import DeepAutoencoder
from dataset import DrivingCaptures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="VAE", help='autoencoder model name')
parser.add_argument('--data_dir', type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/behavior", help='image data dir')
parser.add_argument("--labels", type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/behavior/record.txt")
parser.add_argument('--model_path', type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/selforacle.pt", help='trained model path')
parser.add_argument('--thresholds_path', type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/thresholds.json", help='calculated thresholds path')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)


args = parser.parse_args()  # 获取所有参数

def get_model(args, model_name):
    if model_name == "DAE":
        return DeepAutoencoder(name="DAE",
                               args=args, hidden_layer_dim=256)
    elif model_name == "VAE":
        return DeepAutoencoder(name="VAE", args=args)

def load_or_train_model(args):
    anomaly_detector = get_model(args=args, model_name=args.model_name)
    anomaly_detector.initialize()

    dataset = DrivingCaptures(args.data_dir, args.labels)
    val_dataset = DrivingCaptures(args.data_dir, args.labels, is_train=False)

    anomaly_detector.load_or_train_model(dataset, val_dataset,is_train=True)
    
    return anomaly_detector


def main():
    load_or_train_model(args)


if __name__ == "__main__":
    main()