import os
import datetime
from utilities import *
import time
import torch
import numpy as np
import pickle
from torch.optim.swa_utils import AveragedModel, SWALR

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    swa_sign = False
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1000000))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_trainables) / 1000000))
    trainables = audio_trainables

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))

    print('now use new scheduler')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(10, 60)), gamma=1.0)

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    result = np.zeros([args.n_epochs + 1, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        for i, (audio_input, labels) in enumerate(train_loader):
            # measure data loading time
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            audio_output = audio_model(audio_input)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))

            # original optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)
        print('validation finished')

        cum_stats = stats

        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = np.mean([stat['acc'] for stat in cum_stats])

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = np.mean([stat['acc'] for stat in stats])

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        print("ACC: {:.6f}".format(acc))
        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch, :] = [mAP, acc, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_acc, optimizer.param_groups[0]['lr']]

        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')

        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('number of params groups:' + str(len(optimizer.param_groups)))
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            loss_fn = nn.CrossEntropyLoss()
            # loss without reduction, easy to check per-sample loss
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss