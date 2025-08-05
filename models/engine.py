import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from models.utils import save_imgs
# import SimpleITK as sitk
import torchvision.transforms.functional as TF
from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.binary import assd as medpy_assd
import os


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []
    with tqdm(total=len(train_loader)) as pbar:
        for iter_idx, (_, (large_frame, small_frame, large_trace, small_trace)) in enumerate(train_loader):
            optimizer.zero_grad()
            large_frame = large_frame.cuda(non_blocking=True).float()
            large_trace = large_trace.cuda(non_blocking=True).float()
            small_frame = small_frame.cuda(non_blocking=True).float()
            small_trace = small_trace.cuda(non_blocking=True).float()

            if config.amp:
                with autocast():
                    gt_pre_large, y_large = model(large_frame)
                    gt_pre_small, y_small = model(small_frame)
                    loss_large = criterion(gt_pre_large, y_large[:, 0, :, :], large_trace)
                    loss_small = criterion(gt_pre_small, y_small[:, 0, :, :], small_trace)
                    loss = (loss_large + loss_small) / 2
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                gt_pre_large, y_large = model(large_frame)
                gt_pre_small, y_small = model(small_frame)
                loss_large = criterion(gt_pre_large, y_large[:, 0, :, :], large_trace)
                loss_small = criterion(gt_pre_small, y_small[:, 0, :, :], small_trace)
                loss = (loss_large + loss_small) / 2
                loss.backward()
                optimizer.step()

            loss_list.append(loss.item())

            now_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if iter_idx % config.print_interval == 0:
                log_info = f'train: epoch {epoch},  loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                print(log_info)
                logger.info(log_info)
            pbar.update()
        scheduler.step()
    return np.mean(loss_list)



def val_one_epoch(val_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):

    model.eval()
    preds_large = []
    preds_small = []
    gts_large = []
    gts_small = []
    preds = []
    gts = []
    loss_list = []
    hd95_values, hd95_values_small, hd95_values_large = [], [], []
    assd_values, assd_values_large, assd_values_small = [], [], []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in val_loader:

                large_frame = large_frame.cuda(non_blocking=True).float()
                large_trace = large_trace.cuda(non_blocking=True).float()
                small_frame = small_frame.cuda(non_blocking=True).float()
                small_trace = small_trace.cuda(non_blocking=True).float()

                gt_pre_large, y_large = model(large_frame)
                gt_pre_small, y_small = model(small_frame)
                loss_large = criterion(gt_pre_large, y_large[:, 0, :, :], large_trace)
                loss_small = criterion(gt_pre_small, y_small[:, 0, :, :], small_trace)

                loss = (loss_large + loss_small) / 2
                loss_list.append(loss.item())
                gts.append(large_trace.squeeze(1).cpu().detach().numpy())
                gts.append(small_trace.squeeze(1).cpu().detach().numpy())
                gts_large.append(large_trace.squeeze(1).cpu().detach().numpy())
                gts_small.append(small_trace.squeeze(1).cpu().detach().numpy())
                if type(y_large) and type(y_small) is tuple:
                    y_large = y_large[0]
                    y_small = y_small[0]
                y_large = y_large.squeeze(1).cpu().detach().numpy()
                y_small = y_small.squeeze(1).cpu().detach().numpy()
                preds.append(y_large)
                preds.append(y_small)
                preds_large.append(y_large)
                preds_small.append(y_small)
                pbar.update()
    if epoch % config.val_interval == 0:
        # for pred in preds:

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0


        hd95_average = np.mean(hd95_values)
        hd95_std = np.std(hd95_values)

        assd_average = np.mean(assd_values)
        assd_std = np.std(assd_values)


        log_info = (f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, '
                    f'hd95_average:{hd95_average},hd95_std:{hd95_std},'
                    f'assd_average:{assd_average},assd_std:{assd_std},'
                    f'accuracy:{accuracy}, sensitivity:{sensitivity}, specificity:{specificity},      confusion_matrix:{confusion}')
        print(log_info)
        logger.info(log_info)

        # large计算
        preds_large = np.array(preds_large).reshape(-1)
        preds_small = np.array(preds_small).reshape(-1)
        gts_large = np.array(gts_large).reshape(-1)
        gts_small = np.array(gts_small).reshape(-1)

        y_pre_large = np.where(preds_large >= config.threshold, 1, 0)
        y_true_large = np.where(gts_large >= 0.5, 1, 0)
        y_pre_small = np.where(preds_small >= config.threshold, 1, 0)
        y_true_small = np.where(gts_small >= 0.5, 1, 0)

        confusion_large = confusion_matrix(y_true_large, y_pre_large)
        TN_large, FP_large, FN_large, TP_large = confusion_large[0, 0], confusion_large[0, 1], confusion_large[
            1, 0], confusion_large[1, 1]

        f1_or_dsc_large = float(2 * TP_large) / float(2 * TP_large + FP_large + FN_large) if float(
            2 * TP_large + FP_large + FN_large) != 0 else 0
        miou_large = float(TP_large) / float(TP_large + FP_large + FN_large) if float(
            TP_large + FP_large + FN_large) != 0 else 0


        hd95_average_large = np.mean(hd95_values)
        hd95_std_large = np.std(hd95_values)

        assd_average_large = np.mean(assd_values)
        assd_std_large = np.std(assd_values)

        log_info = (f'val large , miou: {miou_large}, f1_or_dsc: {f1_or_dsc_large}, '
                    f'hd95_average:{hd95_average_large},'
                    f'average_assd:{assd_average_large},confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)

        # small计算
        confusion_small = confusion_matrix(y_true_small, y_pre_small)
        TN_small, FP_small, FN_small, TP_small = confusion_small[0, 0], confusion_small[0, 1], confusion_small[
            1, 0], \
            confusion_small[1, 1]

        f1_or_dsc_small = float(2 * TP_small) / float(2 * TP_small + FP_small + FN_small) if float(
            2 * TP_small + FP_small + FN_small) != 0 else 0
        miou_small = float(TP_small) / float(TP_small + FP_small + FN_small) if float(
            TP_small + FP_small + FN_small) != 0 else 0

        hd95_average_small = np.mean(hd95_values)
        assd_average_small = np.mean(assd_values)

        log_info = (f'val small , miou: {miou_small}, f1_or_dsc: {f1_or_dsc_small}, '
                    f'hd95_average:{hd95_average_small},'
                    f' average_assd:{assd_average_small},')
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def safe_hd95_assd(pred, gt, spacing=(1.0, 1.0)):
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return np.nan, np.nan
    try:
        pred = pred[0]  # 形状变为 (H, W)
        gt = gt[0]  # 形状变为 (H, W)

        if isinstance(spacing, torch.Tensor):
            spacing = spacing.cpu().numpy()  # 从 GPU 转移到 CPU 并转换为 numpy 数组
        elif isinstance(spacing, tuple):
            spacing = np.array(spacing)  # 将 tuple 转换为 numpy 数组并反转（确保维度匹配）
        hd95 = medpy_hd95(pred, gt, voxelspacing=spacing)
        assd = medpy_assd(pred, gt, voxelspacing=spacing)
        return hd95, assd
    except Exception as e:
        print(f"计算HD95/ASSD时出错: {str(e)}")
        return np.nan, np.nan


def test_one_epoch(test_loader, model, criterion, logger, config, mean, std, test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds, preds_large, preds_small = [], [], []
    gts, gts_large, gts_small = [], [], []
    loss_list = []
    hd95_values, hd95_values_small, hd95_values_large = [], [], []
    assd_values, assd_values_large, assd_values_small = [], [], []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (_, (filenames, large_frame, small_frame, large_trace, small_trace)) in test_loader:

                large_frame = large_frame.cuda(non_blocking=True).float()
                large_trace = large_trace.cuda(non_blocking=True).float()
                small_frame = small_frame.cuda(non_blocking=True).float()
                small_trace = small_trace.cuda(non_blocking=True).float()

                gt_pre_large, y_large = model(large_frame)
                gt_pre_small, y_small = model(small_frame)
                loss_large = criterion(gt_pre_large, y_large[:, 0, :, :], large_trace)
                loss_small = criterion(gt_pre_small, y_small[:, 0, :, :], small_trace)

                loss = (loss_large + loss_small) / 2
                loss_list.append(loss.item())

                large_trace = large_trace.squeeze(1).cpu().detach().numpy()
                small_trace = small_trace.squeeze(1).cpu().detach().numpy()
                gts.append(large_trace)
                gts.append(small_trace)
                gts_large.append(large_trace)
                gts_small.append(small_trace)
                if isinstance(y_large, tuple) and isinstance(y_small, tuple):
                    y_large = y_large[0]
                    y_small = y_small[0]

                y_large = y_large.squeeze(1).cpu().detach().numpy()
                y_small = y_small.squeeze(1).cpu().detach().numpy()

                preds.append(y_large)
                preds.append(y_small)
                preds_large.append(y_large)
                preds_small.append(y_small)

                # 计算指标
                hd95_large, assd_large = safe_hd95_assd(
                    (y_large > config.threshold).astype(np.int32),
                    (large_trace > 0.5).astype(np.int32))

                hd95_small, assd_small = safe_hd95_assd(
                    (y_small > config.threshold).astype(np.int32),
                    (small_trace > 0.5).astype(np.int32))

                # 存储有效指标
                if not np.isnan(hd95_large):
                    hd95_values.append(hd95_large)
                    hd95_values_large.append(hd95_large)
                if not np.isnan(assd_large):
                    assd_values.append(assd_large)
                    assd_values_large.append(assd_large)
                if not np.isnan(hd95_small):
                    hd95_values.append(hd95_small)
                    hd95_values_small.append(hd95_small)
                if not np.isnan(assd_small):
                    assd_values.append(assd_small)
                    assd_values_small.append(assd_small)

                print("**************save_imgs***********************")
                print("***********save_path:", config.work_dir + '/outputs/')
                save_imgs(large_frame, large_trace, y_large, filenames, 'large', config.work_dir + '/outputs/',
                          config.datasets, mean, std, config.threshold, test_data_name=test_data_name)
                save_imgs(small_frame, small_trace, y_small, filenames, 'small', config.work_dir + '/outputs/',
                          config.datasets, mean, std, config.threshold, test_data_name=test_data_name)
                pbar.update()
            with open(os.path.join(config.work_dir, 'metrics.txt'), 'w') as f:
                # 写入表头
                f.write('Case_ID\tHD95(mm)\tASSD(mm)\tValid\n')

                # 逐行写入数据
                for i, (hd, assd) in enumerate(zip(hd95_values, assd_values)):
                    # 检查数据有效性
                    is_valid = not (np.isnan(hd) or np.isinf(hd) or
                                    np.isnan(assd) or np.isinf(assd))

                    # 格式化输出（NaN显示为NA，保留3位小数）
                    hd_str = f'{hd:.3f}' if not np.isnan(hd) else 'NA'
                    assd_str = f'{assd:.3f}' if not np.isnan(assd) else 'NA'

                    f.write(f'Case_{i + 1}\t{hd_str}\t{assd_str}\t{is_valid}\n')

            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)

            y_pre = np.where(preds >= config.threshold, 1, 0)
            y_true = np.where(gts >= 0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0


            hd95_average = np.mean(hd95_values)
            hd95_std = np.std(hd95_values)

            assd_average = np.mean(assd_values)
            assd_std = np.std(assd_values)

            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)

            log_info = (f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, '
                        f'hd95_average:{hd95_average},hd95_std:{hd95_std},'
                        f'assd_average:{assd_average},assd_std:{assd_std},'
                        f'accuracy:{accuracy}, sensitivity:{sensitivity}, specificity:{specificity}, ')
            print(log_info)
            logger.info(log_info)

            # large计算
            preds_large = np.array(preds_large).reshape(-1)
            preds_small = np.array(preds_small).reshape(-1)
            gts_large = np.array(gts_large).reshape(-1)
            gts_small = np.array(gts_small).reshape(-1)

            y_pre_large = np.where(preds_large >= config.threshold, 1, 0)
            y_true_large = np.where(gts_large >= 0.5, 1, 0)
            y_pre_small = np.where(preds_small >= config.threshold, 1, 0)
            y_true_small = np.where(gts_small >= 0.5, 1, 0)

            confusion_large = confusion_matrix(y_true_large, y_pre_large)
            TN_large, FP_large, FN_large, TP_large = confusion_large[0, 0], confusion_large[0, 1], confusion_large[
                1, 0], confusion_large[1, 1]

            f1_or_dsc_large = float(2 * TP_large) / float(2 * TP_large + FP_large + FN_large) if float(
                2 * TP_large + FP_large + FN_large) != 0 else 0
            miou_large = float(TP_large) / float(TP_large + FP_large + FN_large) if float(
                TP_large + FP_large + FN_large) != 0 else 0


            hd95_average_large = np.mean(hd95_values_large)
            assd_average_large = np.mean(assd_values_large)


            log_info = (f'test large , miou: {miou_large}, f1_or_dsc: {f1_or_dsc_large}, '
                        f'hd95_average:{hd95_average_large},'
                        f'average_assd:{assd_average_large},')
            print(log_info)
            logger.info(log_info)

            # small计算
            confusion_small = confusion_matrix(y_true_small, y_pre_small)
            TN_small, FP_small, FN_small, TP_small = confusion_small[0, 0], confusion_small[0, 1], confusion_small[
                1, 0], \
                confusion_small[1, 1]

            f1_or_dsc_small = float(2 * TP_small) / float(2 * TP_small + FP_small + FN_small) if float(
                2 * TP_small + FP_small + FN_small) != 0 else 0
            miou_small = float(TP_small) / float(TP_small + FP_small + FN_small) if float(
                TP_small + FP_small + FN_small) != 0 else 0


            hd95_average_small = np.mean(hd95_values_small)
            assd_average_small = np.mean(assd_values_small)

            log_info = (f'test small , miou: {miou_small}, f1_or_dsc: {f1_or_dsc_small}, '
                        f'hd95_average:{hd95_average_small},'
                        f'average_assd:{assd_average_small},confusion_matrix: {confusion}')
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)
