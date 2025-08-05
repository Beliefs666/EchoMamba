"""Functions for training and running segmentation."""

import sys
import time

import click
import scipy.signal
import skimage.draw
import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import echonet
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from models.config_setting import setting_config
from models.engine import *
from models.utils import *
import matplotlib.pyplot as plt
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
import shutil
import torchvision
from models.EchoMamba import EchoMamba



@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--num_epochs", type=int, default=50)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=4)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=42)




def run(
    data_dir=None,
    weights=None,
    num_epochs=50,
    num_workers=4,
    batch_size=4,
    device=None,
    seed=42,
):
    """Trains/val/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 42.
    """
   # print("*****************lr{}******************",lr)
    config = setting_config
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')   #  results/UlterLight...../
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')    # resume_model=results/UlterLight...../checkpoints/latest.pth
    outputs = os.path.join(config.work_dir, 'outputs')
    outputs_video = os.path.join(config.work_dir, 'outputs_video')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  #  results/UlterLight...../checkpoints/
    if not os.path.exists(outputs):
        os.makedirs(outputs)  #  results/UlterLight...../outputs/
    if not os.path.exists(outputs_video):
        os.makedirs(outputs_video)

    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)


    print('#----------set device----------#')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('#----------GPU init----------#')
    set_seed(seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()


    # Set up model

    model_cfg = config.model_config
    model=EchoMamba(num_classes=model_cfg['num_classes'],
                           input_channels=model_cfg['input_channels'],
                           c_list=model_cfg['c_list'],
                           split_att=model_cfg['split_att'],
                           bridge=model_cfg['bridge'], )

    model.to(device)


    print('#----------print flops and params----------#')
    inp = torch.randn(1, 3,112,112)
    inp=inp.to(device)
    flops = FlopCountAnalysis(model, inp)
    # print(flop_count_table(flops, max_depth=5))
    log_info = f'{flop_count_table(flops, max_depth=5)}'
    logger.info(log_info)
    print(log_info)



    if device.type == "cuda":
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])



    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    if os.path.exists(os.path.join(checkpoint_dir, 'best_dynamic.pth')):
        print('#----------loading pre----------#')
        best_weight = torch.load(config.work_dir + '/checkpoints/best_dynamic.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1



    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }



    print('#----------Preparing echenet dataset----------#')
    train_dataset = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                # batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers)
    val_dataset = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                drop_last=True)
    #用于测试评估
    test_dataset = echonet.datasets.Echo(root=data_dir, split="test", target_type=["Filename","LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"],mean=mean, std=std )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                              shuffle=False,
                                              pin_memory=(device.type == "cuda"),
                                              num_workers=num_workers)

    print('#----------Preparing camus dataset----------#')
    # train_dataset = echonet.datasets.camus.HEART_datasets(path_Data=data_dir, config=config, train=True, val=False)
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            # batch_size=config.batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=num_workers)
    # val_dataset = echonet.datasets.camus.HEART_datasets(path_Data=data_dir, config=config, train=False, val=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=num_workers,
    #                                          drop_last=True)
    # # 用于测试评估
    # test_dataset = echonet.datasets.camus.HEART_datasets(path_Data=data_dir, config=config, train=False, val=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=1, num_workers=num_workers, shuffle=False,
    #                                           pin_memory=(device.type == "cuda"))

    print('#----------Preparing hmcqu dataset----------#')

    # train_dataset = echonet.datasets.hmcqu.get_hmcqu_dataset(
    #     [ f"{data_dir}/train.csv"  ],
    #     base_path=f"{data_dir}/",
    #     file_name="records.h5"
    # )
    # test_dataset = echonet.datasets.hmcqu.get_hmcqu_dataset(
    #      [ f"{data_dir}/test.csv"  ],
    #     base_path=f"{data_dir}/",
    #     file_name="records.h5"
    # )
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    # Run training and testing loops
    with open(os.path.join(outputs, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:


            if os.path.exists(resume_model):
                print('#----------Resume Model and Other params----------#')
                checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
                model.module.load_state_dict(checkpoint['model_state_dict'],strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                saved_epoch = checkpoint['epoch']
                start_epoch += saved_epoch
                print("**********start_epoch******************:",start_epoch)
                min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

                log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
                logger.info(log_info)
                print(log_info)
        except FileNotFoundError:
            f.write("Starting run from scratch\n")




        print('#----------Training----------#')
        loss_history0 = []
        loss_history1 = []
        for epoch in range(start_epoch, num_epochs + 1):

            torch.cuda.empty_cache()

            loss_train = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
            min_loss_train=999
            loss_history0.append(loss_train)
            if loss_train< min_loss_train:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_loss_train.pth'))
                min_loss_train = loss_train
                min_loss_epoch = epoch
                shutil.copy(
                                os.path.join(checkpoint_dir, 'best_loss_train.pth'),
                                os.path.join(checkpoint_dir, f'best-epoch{min_loss_epoch}-loss{min_loss_train:.4f}-train.pth')
                            )

            loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
            loss_history1.append(loss)
            if loss < min_loss:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch
                shutil.copy(
                    os.path.join(checkpoint_dir, 'best.pth'),
                    os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
                )
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                     }, os.path.join(checkpoint_dir, 'latest.pth'))


        # 绘制损失曲线
        save_loss_plot(loss_history0, os.path.join(outputs, "loss_plot0.png"))
        save_loss_plot(loss_history1, os.path.join(outputs, "loss_plot1.png"))

        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            model.module.load_state_dict(best_weight)

            loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                mean,
                std
            )




def save_loss_plot(loss_history, save_path):
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i

