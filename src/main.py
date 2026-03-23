import argparse
from ads import run_ads
from training_runner import get_embedding_library, pretrain, build_and_init_model_from_pretraining, calc_hypersphere_center, test_for_single_folder, train, test, TrainRunner

def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--watch", action="store_true")
    argparser.add_argument("--record", action="store_true")
    argparser.add_argument("--normal", action="store_true")
    argparser.add_argument("--agent", default="behavior", type=str, choices=["behavior", "resnet101", "vgg16", "epoch"])
    argparser.add_argument("--oracle", default="Atten", type=str, choices=["Atten", "SelfOracle", "ThirdEye", "Atten_ae"])
    argparser.add_argument("--R", default=1, type=float)
    argparser.add_argument("--output", default="output", type=str)
    argparser.add_argument("--driving_model", default="driving_model/model", type=str)
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--num_vehicles", default=20, type=int)
    argparser.add_argument("--num_walkers", default=0, type=int)
    argparser.add_argument("--time", default=200, type=int)
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--cal_threashold", action="store_true")
    argparser.add_argument("--test", action="store_true")
    argparser.add_argument("--pretrain", action="store_true")
    argparser.add_argument("--pretrain_batch_size", default=64, type=int)
    argparser.add_argument("--pretrain_lr", default=0.001, type=float)
    argparser.add_argument("--pretrain_epochs", default=10, type=int)
    argparser.add_argument("--pretrain_model", default="output/pretrain_model_vgg16")
    argparser.add_argument("--seq_len", default=8, type=int)
    argparser.add_argument("--num_features", default=10, type=int)
    argparser.add_argument("--root_dir", default="dataset/behavior", type=str)
    argparser.add_argument("--num_frames", default=8, type=int)
    argparser.add_argument("--num_workers", default=8, type=int)
    argparser.add_argument("--img_size", default=224, type=int)
    argparser.add_argument("--batch_size", default=32, type=int)
    argparser.add_argument("--lr", default=0.01, type=float)
    argparser.add_argument("--epochs", default=200, type=int)
    argparser.add_argument("--weight_decay", default=1e-3, type=float)
    argparser.add_argument("--center_path", default="output/center")
    argparser.add_argument("--cal_center", action="store_true")
    argparser.add_argument("--model", default="output/model", type=str)
    argparser.add_argument('--model_path', type=str, default="output/selforacle.pt", help='trained model path')
    argparser.add_argument('--thresholds_path', type=str, default="output/thresholds.json", help='calculated thresholds path')


    argparser.add_argument("--labels", default="output/behavior/record.txt", type=str)
    argparser.add_argument("--embedding_dim", default=256, type=int)
    argparser.add_argument("--eps", default=1e-6, type=float)
    argparser.add_argument("--eta", default=1.0, type=float)
    argparser.add_argument("--abnormal_start", default=60, type=int)
    argparser.add_argument("--abnormal_end", default=5, type=int)
    argparser.add_argument("--retrain", action="store_true")
    argparser.add_argument("--ratio", default=1.0, type=float)
    argparser.add_argument('--imu', type=str, default="output/hd_thresholds.json", help='calculated thresholds path')






    return argparser


if __name__ == "__main__":
    argparser = set_args()
    args = argparser.parse_args()
    if args.watch:
        run_ads(args)

    if args.train:
        train_runner = TrainRunner(args)
        if args.pretrain:
            print("开始预先训练模型----------------------------")
            train_runner.pretrain(args)
        train_runner.init_model_from_pretraining(args)
        if args.cal_center:
            print("start cal center ----------------------------")
            train_runner.cal_center(args)
        train_runner.train(args)
    
    if args.cal_threashold:
        train_runner = TrainRunner(args)
        if args.oracle != "Atten_ae":
            print("开始拟合Gamma分布--------------------------------------------")
            train_runner.calc_and_store_thresholds(args)
        else:
            print("开始拟合Gamma分布--------------------------------------------")
            train_runner.init_model_from_pretraining(args)
            train_runner.cal_assert_ae(args)
    
    if args.test:
        train_runner = TrainRunner(args)
        print("开始计算评价指标---------------------------------------------")
        train_runner.test_for_agent(args)

