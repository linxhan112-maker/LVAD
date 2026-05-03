mport argparse
import datetime
import json
import os
import time
from pathlib import Path

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset_shan import AbnormalDatasetGradientsTest
from data.train_dataset_shan import AbnormalDatasetGradientsTrain
from engine_train import train_one_epoch, test_one_epoch
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc
import torch

def print_diff_params(student_sd, teacher_sd, tol=1e-6):
    diff_count = 0

    for k in student_sd.keys():
        s_param = student_sd[k].float()
        t_param = teacher_sd[k].float()
        if s_param.shape != t_param.shape:
            print(f"⚠️ Shape mismatch at {k}: student {s_param.shape}, teacher {t_param.shape}")
            continue

        # 计算差异
        diff_mask = (s_param - t_param).abs() > tol
        if diff_mask.any():
            diff_count += 1
            print(f"\n参数名: {k}")
            print(f"形状: {s_param.shape}")
            print("Student 参数样例:", s_param[diff_mask].flatten()[:10])
            print("Teacher 参数样例:", t_param[diff_mask].flatten()[:10])
            print("差异样例:", (s_param - t_param)[diff_mask].flatten()[:10])

    print(f"\n总共有 {diff_count} 个参数与 teacher 不同 (阈值 {tol})")

def count_params(state_dict):
    total = 0
    for k, v in state_dict.items():
        total += v.numel()
    return total

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    log_writer = SummaryWriter(log_dir=args.output_dir)

    device = args.device
    if args.run_type =='train':
        dataset_train = AbnormalDatasetGradientsTrain(args)
        print(dataset_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    dataset_test = AbnormalDatasetGradientsTest(args)
    print(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # define the model
    if args.dataset == 'avenue':
        model = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                                abnormal_score_func=args.abnormal_score_func,
                                                masking_method=args.masking_method,
                                                grad_weighted_loss=args.grad_weighted_rec_loss).float()
    else:
        model = mae_cvt_patch8(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                                abnormal_score_func=args.abnormal_score_func,
                                                masking_method=args.masking_method,
                                                grad_weighted_loss=args.grad_weighted_rec_loss).float()
    # 查看增强模块参数的 requires_grad 属性
    # for name, param in model.enhancer.named_parameters():
    #     print(f"参数名: {name} | 是否可训练: {param.requires_grad}")

    # 预期输出示例：
    # 参数名: enhance.in_conv.0.weight | 是否可训练: True
    # 参数名: enhance.in_conv.0.bias | 是否可训练: True
    # ...其他参数应显示为True

    model.to(device)
    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model)
    elif args.run_type == "inference":
        student = torch.load(args.output_dir + "/checkpoint-best-student.pth")['model']
        teacher = torch.load(args.output_dir + "/checkpoint-best.pth")['model']

        print(f"Student total params: {count_params(student):,}")
        print(f"Teacher total params: {count_params(teacher):,}")

        for key in student:
            if 'student' in key:
                teacher[key] = student[key]
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)




def do_training(args, data_loader_test, data_loader_train, device, log_writer, model):
    print("actual lr: %.2e" % args.lr)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_micro = 0.0
    best_micro_student = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args
        )
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        test_stats = test_one_epoch(
            model, data_loader_test, device, epoch, log_writer=log_writer, args=args
        )
        log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

        if args.output_dir:
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, latest=True)
        if test_stats['micro'] > best_micro:
            best_micro = test_stats['micro']
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, best=True)
        if args.start_TS_epoch <= epoch:
            if test_stats['micro'] > best_micro_student:
                best_micro_student = test_stats['micro']
                misc.save_model(args=args, model=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best=True, student=True)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_test) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    args = parser.parse_args()
    if args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()#
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
