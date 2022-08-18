import math
import os
import pickle
import shutil
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from scipy.stats import gmean
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision.utils import make_grid
import copy


class import_and_train_model:
    def __init__(self, initMode='default', verbose=True):
        self.best_values = [0, 0, 0]
        self.f1 = 0
        self.acc = 0
        self.initial_epoch = 0
        self.epoch = None
        self.loss = None
        self.class_weights_tensor = None
        self.initMode = initMode
        self.verbose = verbose
        self.model = None
        self.early_stopping = None
        self.lr_scheduler = None
        self.optimizer = None
        self.criterion = None
        self.checkpoint_path = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.classes = None
        return

    def import_deit_models(self, train_main, data_loader):
        classes = data_loader.classes
        self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True,
                                       num_classes=len(np.unique(classes)))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = nn.DataParallel(model) # to run on multiple GPUs
        self.model.to(device)

        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        self.criterion = nn.CrossEntropyLoss(data_loader.class_weights_tensor)

        torch.cuda.set_device(train_main.params.gpu_id)
        self.model.cuda(train_main.params.gpu_id)
        self.criterion = self.criterion.cuda(train_main.params.gpu_id)

        # Observe that all parameters are being optimized
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_main.params.lr,
                                           weight_decay=train_main.params.weight_decay)

        # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Early stopping and lr scheduler
        self.lr_scheduler = LRScheduler(self.optimizer)
        self.early_stopping = EarlyStopping()

    def import_deit_models_for_testing(self, train_main, test_main):
        classes = np.load(test_main.params.main_param_path + '/classes.npy')
        self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True,
                                       num_classes=len(np.unique(classes)))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = nn.DataParallel(model)
        self.model.to(device)

        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')
        self.criterion = nn.CrossEntropyLoss(class_weights_tensor)

        torch.cuda.set_device(train_main.params.gpu_id)
        self.model.cuda(train_main.params.gpu_id)
        self.criterion = self.criterion.cuda(train_main.params.gpu_id)

        # Observe that all parameters are being optimized
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_main.params.lr,
                                           weight_decay=train_main.params.weight_decay)

        # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Early stopping and lr scheduler
        self.lr_scheduler = LRScheduler(self.optimizer)
        self.early_stopping = EarlyStopping()

    def run_training(self, train_main, data_loader, initial_epoch, epochs, lr, name, best_values):

        best_loss, best_f1, best_acc1 = best_values[0], best_values[1], best_values[2]

        print("Initial values:- best_loss : {}, best_f1: {}, best_acc1: {} ".format(best_loss, best_f1, best_acc1))

        train_losses, test_losses, train_accuracies, test_accuracies, train_f1s, test_f1s = [], [], [], [], [], []

        print("Beginning training")
        time_begin = time()

        for epoch in range(initial_epoch, epochs):
            time_begin_epoch = time()
            print('EPOCH : {} / {}'.format(epoch + 1, epochs))

            adjust_learning_rate(self.optimizer, epoch, lr, train_main.params.warmup,
                                 train_main.params.disable_cos, epochs)

            train_acc1, train_loss, train_outputs, train_targets = cls_train(data_loader.train_dataloader, self.model,
                                                                             self.criterion,
                                                                             self.optimizer,
                                                                             train_main.params.clip_grad_norm)
            test_acc1, test_loss, test_outputs, test_targets, total_mins = cls_validate(data_loader.val_dataloader,
                                                                                        self.model, self.criterion,
                                                                                        time_begin=time_begin)

            train_f1 = f1_score(train_outputs, train_targets, average='macro')
            train_accuracy = accuracy_score(train_outputs, train_targets)

            test_f1 = f1_score(test_outputs, test_targets, average='macro')
            test_accuracy = accuracy_score(test_outputs, test_targets)

            if train_main.params.save_best_model_on_loss_or_f1_or_accuracy == 1:
                if test_loss < best_loss or epoch == 1:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                                'f1': test_f1,
                                'acc': test_acc1,
                                'epoch': epoch},
                               data_loader.checkpoint_path + '/trained_model_' + name + '.pth')

            elif train_main.params.save_best_model_on_loss_or_f1_or_accuracy == 2:

                if test_f1 > best_f1 or epoch == 1:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                                'f1': test_f1,
                                'acc': test_acc1,
                                'epoch': epoch},
                               data_loader.checkpoint_path + '/trained_model_' + name + '.pth')

            elif train_main.params.save_best_model_on_loss_or_f1_or_accuracy == 3:
                if test_acc1 > best_acc1 or epoch == 1:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                                'f1': test_f1,
                                'acc': test_acc1,
                                'epoch': epoch},
                               data_loader.checkpoint_path + '/trained_model_' + name + '.pth')
            else:
                print('Choose correct metric i.e. based on loss or acc or f1 to save the model')

            best_acc1 = max(test_acc1, best_acc1)
            best_f1 = max(test_f1, best_f1)
            best_loss = min(test_f1, best_loss)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            train_f1s.append(train_f1)
            test_f1s.append(test_f1)

            total_mins_per_epoch = (time() - time_begin_epoch) / 60

            print('[Train] Acc:{}, F1:{}, loss:{}'.format(np.round(train_accuracy, 3),
                                                          np.round(train_f1, 3),
                                                          np.round(train_loss, 3),
                                                          np.round(test_accuracy, 3)))
            print('[Test] Acc:{}, F1:{}, loss:{}, epoch time (in mins) :{}, cumulative time (in mins):{}'.format(np.round(test_accuracy, 3),
                                                                              np.round(test_f1, 3),
                                                                              np.round(test_loss, 3),
                                                                              np.round(total_mins_per_epoch, 3),
                                                                              np.round(total_mins, 3)))
            if train_main.params.run_lr_scheduler == 'yes':
                self.lr_scheduler(test_loss)

            if train_main.params.run_early_stopping == 'yes':
                self.early_stopping(test_loss)
                if self.early_stopping.early_stop:
                    break

        total_mins = (time() - time_begin) / 60

        print(f'Script finished in {total_mins:.2f} minutes, '
              f'best acc top-1: {best_acc1:.2f}, '
              f'best loss top-1: {best_loss:.2f}, '
              f'best f1 top-1: {best_f1:.2f}, ')

        Logs = [train_losses, train_accuracies, test_losses, test_accuracies, train_f1s, test_f1s]

        Log_Path = data_loader.checkpoint_path

        with open(Log_Path + '/Logs_' + name + '.pickle', 'ab') as cw:
            pickle.dump(Logs, cw)

        Logs = pd.read_pickle(Log_Path + '/Logs_' + name + '.pickle')

        train_losses = Logs[0]
        train_f1s = Logs[4]
        test_losses = Logs[2]
        test_f1s = Logs[5]

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training loss')
        plt.subplot(1, 2, 1)
        plt.plot(test_losses, label='Validation loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_f1s, label='Training F1')
        plt.subplot(1, 2, 2)
        plt.plot(test_f1s, label='Validation F1')

        plt.savefig(data_loader.checkpoint_path + '/performance_curves_' + name + '.png')

    def run_prediction(self, data_loader, name):
        # classes = np.load(train_main.params.outpath + '/classes.npy')
        classes = data_loader.classes
        PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'

        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        avg_acc1, target, output, prob = cls_predict(data_loader.test_dataloader, self.model, self.criterion,
                                                     time_begin=time())

        target = torch.cat(target)
        output = torch.cat(output)
        prob = torch.cat(prob)

        target = target.cpu().numpy()
        output = output.cpu().numpy()
        prob = prob.cpu().numpy()

        output_max = output.argmax(axis=1)

        target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
        output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

        GT_Pred_GTLabel_PredLabel_Prob = [target, output_max, target_label, output_label, prob]
        with open(data_loader.checkpoint_path + '/GT_Pred_GTLabel_PredLabel_prob_model_' + name + '.pickle', 'wb') \
                as cw:
            pickle.dump(GT_Pred_GTLabel_PredLabel_Prob, cw)

        accuracy_model = accuracy_score(target_label, output_label)
        clf_report = classification_report(target_label, output_label)
        f1 = f1_score(target_label, output_label, average='macro')

        f = open(data_loader.checkpoint_path + 'test_report_' + name + '.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                              clf_report))
        f.close()

    def resuming_training(self, train_main, data_loader, modeltype):
        # self.import_deit_models(train_main, data_loader)

        if modeltype == 0:
            PATH = data_loader.checkpoint_path + 'trained_model_original.pth'
        elif modeltype == 1:
            PATH = data_loader.checkpoint_path + 'trained_model_tuned.pth'
        elif modeltype == 2:
            PATH = data_loader.checkpoint_path + 'trained_model_finetuned.pth'

        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.f1 = checkpoint['f1']
        self.acc = checkpoint['acc']
        self.initial_epoch = checkpoint['epoch']
        self.best_values = [self.loss, self.f1, self.acc]

        if modeltype == 0:
            if self.initial_epoch < train_main.params.epochs:
                print('Original trained model exists but not completely trained. Therefore resuming the training from '
                      'previous epochs')
                self.init_train_predict(train_main, data_loader, 0)

        elif modeltype == 1:
            if self.initial_epoch < train_main.params.finetune_epochs:
                print('Tuned trained model exists but not completely trained. Therefore resuming the training from '
                      'previous epochs')
                self.init_train_predict(train_main, data_loader, 1)

        elif modeltype == 2:
            if self.initial_epoch < train_main.params.finetune_epochs:
                print('Finetuned trained model exists but not completely trained. Therefore resuming the training from '
                      'previous epochs')
                self.init_train_predict(train_main, data_loader, 2)

    def init_train_predict(self, train_main, data_loader, modeltype):
        if modeltype == 0:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.params.lr)
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.epochs,
                              train_main.params.lr, "original", self.best_values)
            self.run_prediction(data_loader, 'original')

        elif modeltype == 1:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.params.lr / 10)
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.finetune_epochs,
                              train_main.params.lr / 10, "tuned", self.best_values)
            self.run_prediction(data_loader, 'tuned')

        elif modeltype == 2:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.params.lr / 100)
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.finetune_epochs,
                              train_main.params.lr / 100, "finetuned", self.best_values)
            self.run_prediction(data_loader, 'finetuned')

    def train_predict(self, train_main, data_loader, modeltype):
        if modeltype == 0:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.epochs,
                              train_main.params.lr, "original", self.best_values)
            self.run_prediction(data_loader, 'original')

        elif modeltype == 1:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.finetune_epochs,
                              train_main.params.lr / 10, "tuned", self.best_values)
            self.run_prediction(data_loader, 'tuned')

        elif modeltype == 2:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.params.finetune_epochs,
                              train_main.params.lr / 100, "finetuned", self.best_values)
            self.run_prediction(data_loader, 'finetuned')

    def train_and_save(self, train_main, data_loader):
        model_present_path0 = data_loader.checkpoint_path + 'trained_model_original.pth'
        model_present_path1 = data_loader.checkpoint_path + 'trained_model_tuned.pth'
        model_present_path2 = data_loader.checkpoint_path + 'trained_model_finetuned.pth'

        self.import_deit_models(train_main, data_loader)
        if train_main.params.finetune == 0:
            if not os.path.exists(model_present_path0):
                self.train_predict(train_main, data_loader, 0)
            else:
                self.resuming_training(train_main, data_loader, 0)

        elif train_main.params.finetune == 1:
            if not os.path.exists(model_present_path0):
                self.train_predict(train_main, data_loader, 0)
                self.init_train_predict(train_main, data_loader, 1)
            elif not os.path.exists(model_present_path1):
                print(' I am using trained_model_original.pth as the base')
                self.resuming_training(train_main, data_loader, 0)
                print('Now training tuned model')
                self.initial_epoch = 0
                self.init_train_predict(train_main, data_loader, 1)
            else:
                print(' I am using trained_model_tuned.pth as the base')
                self.resuming_training(train_main, data_loader, 1)

        elif train_main.params.finetune == 2:

            if not os.path.exists(model_present_path0):
                self.train_predict(train_main, data_loader, 0)
                self.init_train_predict(train_main, data_loader, 1)
                self.init_train_predict(train_main, data_loader, 2)
            elif not os.path.exists(model_present_path1):
                print(' I am using trained_model_original.pth as the base')
                self.resuming_training(train_main, data_loader, 0)
                print('Now training tuned model')
                self.initial_epoch = 0
                self.init_train_predict(train_main, data_loader, 1)
                print('Now training finetuned model')
                self.initial_epoch = 0
                self.init_train_predict(train_main, data_loader, 2)

            elif not os.path.exists(model_present_path2):
                print(' I am using trained_model_tuned.pth as the base')
                self.resuming_training(train_main, data_loader, 1)
                print('Now training finetuned model')
                self.initial_epoch = 0
                self.init_train_predict(train_main, data_loader, 2)
            else:
                print(' I am using trained_model_finetuned.pth as the base')
                self.resuming_training(train_main, data_loader, 2)
        else:
            print('Choose the correct finetune label')

    def run_prediction_on_unseen(self, test_main, data_loader, name):
        classes = np.load(test_main.params.main_param_path + '/classes.npy')
        if len(test_main.params.model_path) > 1:
            print("Do you want to predict using ensemble model ? If so then set the ensemble parameter to 1 and run "
                  "again")
        else:
            checkpoint_path = test_main.params.model_path[0]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'
            # PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'
            im_names = data_loader.Filenames

            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            output, prob = cls_predict_on_unseen(data_loader.test_dataloader, self.model)

            output = torch.cat(output)
            prob = torch.cat(prob)

            output = output.cpu().numpy()
            prob = prob.cpu().numpy()

            output_max = output.argmax(axis=1)

            output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

            output_corrected_label = copy.deepcopy(output_label)

            first_indices = prob.argsort()[:, -1]
            confs = [prob[i][first_indices[i]] for i in range(len(first_indices))]
            for i in range(len(confs)):
                if confs[i] < test_main.params.threshold:
                    output_corrected_label[i] = 'unknown'

            # Pred_PredLabel_Prob = [output_max, output_label, output_corrected_label, prob]
            # with open(test_main.params.test_outpath + '/Single_model_Pred_PredLabel_Prob_' + name + '.pickle',
            # 'wb') as cw:
            #     pickle.dump(Pred_PredLabel_Prob, cw)

            output_label = output_label.tolist()

            if test_main.params.threshold > 0:
                print('I am using threshold value as : {}'.format(test_main.params.threshold))
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_corrected_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions_thresholded.txt',
                           To_write, fmt='%s')
            else:
                print('I am using default value as threshold i.e. 0')
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

    def run_ensemble_prediction_on_unseen(self, test_main, data_loader, name):
        classes = np.load(test_main.params.main_param_path + '/classes.npy')
        Ensemble_prob = []
        im_names = data_loader.Filenames

        for i in range(len(test_main.params.model_path)):
            checkpoint_path = test_main.params.model_path[i]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'

            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            output, prob = cls_predict_on_unseen(data_loader.test_dataloader, self.model)

            prob = torch.cat(prob)

            prob = prob.cpu().numpy()

            Ensemble_prob.append(prob)

        Ens_DEIT_prob_max = []
        Ens_DEIT_label = []
        Ens_DEIT = []
        name2 = []

        if test_main.params.ensemble == 1:
            Ens_DEIT = sum(Ensemble_prob) / len(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'arth_mean_'

        elif test_main.params.ensemble == 2:
            Ens_DEIT = gmean(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'geo_mean_'

        Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)

        first_indices = Ens_DEIT.argsort()[:, -1]
        Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        for i in range(len(Ens_confs)):
            if Ens_confs[i] < test_main.params.threshold:
                Ens_DEIT_corrected_label[i] = 'unknown'

        # Pred_PredLabel_Prob = [Ens_DEIT_prob_max, Ens_DEIT_label, Ens_DEIT_corrected_label, Ens_DEIT]
        # with open(test_main.params.test_outpath + '/Ensemble_models_Pred_PredLabel_Prob_' + name2 + name + '.pickle',
        #           'wb') as cw:
        #     pickle.dump(Pred_PredLabel_Prob, cw)

        Ens_DEIT_label = Ens_DEIT_label.tolist()

        if test_main.params.threshold > 0:
            print('I am using threshold value as : {}'.format(test_main.params.threshold))
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_corrected_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '_thresholded.txt', To_write, fmt='%s')
        else:
            print('I am using default value as threshold i.e. 0')
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

    def run_prediction_on_unseen_with_y(self, test_main, data_loader, name):
        classes = np.load(test_main.params.main_param_path + '/classes.npy')
        if len(test_main.params.model_path) > 1:
            print("Do you want to predict using ensemble model ? If so then set the ensemble parameter to 1 and run "
                  "again")
        else:
            checkpoint_path = test_main.params.model_path[0]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'
            # PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'
            im_names = data_loader.Filenames

            # print('im_names : {}'.format(im_names))

            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            avg_acc1, target, output, prob = cls_predict_on_unseen_with_y(data_loader.test_dataloader, self.model,
                                                                          self.criterion,
                                                                          time_begin=time())

            target = torch.cat(target)
            output = torch.cat(output)
            prob = torch.cat(prob)

            target = target.cpu().numpy()
            output = output.cpu().numpy()
            prob = prob.cpu().numpy()

            output_max = output.argmax(axis=1)

            target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
            output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

            output_corrected_label = copy.deepcopy(output_label)

            first_indices = prob.argsort()[:, -1]
            confs = [prob[i][first_indices[i]] for i in range(len(first_indices))]
            for i in range(len(confs)):
                if confs[i] < test_main.params.threshold:
                    output_corrected_label[i] = 'unknown'

            GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob = [target, output_max, target_label, output_label,
                                                                 output_corrected_label, prob]
            with open(
                    test_main.params.test_outpath + '/Single_GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_' + name + '.pickle',
                    'wb') \
                    as cw:
                pickle.dump(GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob, cw)

            output_label = output_label.tolist()

            if test_main.params.threshold > 0:
                print('I am using threshold value as : {}'.format(test_main.params.threshold))
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_corrected_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions_thresholded.txt',
                           To_write, fmt='%s')

            else:
                print('I am using default value as threshold i.e. 0')
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.params.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

            accuracy_model = accuracy_score(target_label, output_label)
            clf_report = classification_report(target_label, output_label)
            f1 = f1_score(target_label, output_label, average='macro')

            f = open(test_main.params.test_outpath + 'Single_test_report_' + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

    def run_ensemble_prediction_on_unseen_with_y(self, test_main, data_loader, name):
        classes = np.load(test_main.params.main_param_path + '/classes.npy')
        Ensemble_prob = []
        Ensemble_GT = []
        im_names = data_loader.Filenames

        for i in range(len(test_main.params.model_path)):
            checkpoint_path = test_main.params.model_path[i]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'

            checkpoint = torch.load(PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            # output, prob = cls_predict_on_unseen(data_loader.test_dataloader, self.model)
            avg_acc1, target, output, prob = cls_predict_on_unseen_with_y(data_loader.test_dataloader, self.model,
                                                                          self.criterion,
                                                                          time_begin=time())

            target = torch.cat(target)
            prob = torch.cat(prob)

            prob = prob.cpu().numpy()
            target = target.cpu().numpy()

            Ensemble_prob.append(prob)
            Ensemble_GT.append(target)

        Ens_DEIT_prob_max = []
        Ens_DEIT_label = []
        Ens_DEIT = []
        name2 = []
        GT = Ensemble_GT[0]
        GT_label = np.array([classes[GT[i]] for i in range(len(GT))], dtype=object)

        if test_main.params.ensemble == 1:
            Ens_DEIT = sum(Ensemble_prob) / len(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'arth_mean_'

        elif test_main.params.ensemble == 2:
            Ens_DEIT = gmean(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'geo_mean_'

        Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)

        first_indices = Ens_DEIT.argsort()[:, -1]
        Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        for i in range(len(Ens_confs)):
            if Ens_confs[i] < test_main.params.threshold:
                Ens_DEIT_corrected_label[i] = 'unknown'

        GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob = [GT, Ens_DEIT_prob_max, GT_label, Ens_DEIT_label,
                                                             Ens_DEIT_corrected_label, Ens_DEIT]
        with open(
                test_main.params.test_outpath + '/GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_' + name2 + name + '.pickle',
                'wb') \
                as cw:
            pickle.dump(GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob, cw)

        Ens_DEIT_label = Ens_DEIT_label.tolist()

        if test_main.params.threshold > 0:
            print('I am using threshold value as : {}'.format(test_main.params.threshold))

            ## Original
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_label)
            clf_report = classification_report(GT_label, Ens_DEIT_label)
            f1 = f1_score(GT_label, Ens_DEIT_label, average='macro')

            f = open(test_main.params.test_outpath + 'Ensemble_test_report_' + name2 + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            filenames_out = im_names[0]
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_label[jj]:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

            ## Thresholded

            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_corrected_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '_thresholded_' + str(test_main.params.threshold) + '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_corrected_label)
            clf_report = classification_report(GT_label, Ens_DEIT_corrected_label)
            f1 = f1_score(GT_label, Ens_DEIT_corrected_label, average='macro')

            f = open(test_main.params.test_outpath + 'Ensemble_test_report_' + name2 + name + '_thresholded_' + str(
                test_main.params.threshold) + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            filenames_out = im_names[0]
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_corrected_label[jj]:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '_thresholded_' + str(
                        test_main.params.threshold) + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '_thresholded_' + str(
                        test_main.params.threshold) + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_corrected_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

        else:
            print('I am using default value as threshold i.e. 0')
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.params.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_label)
            clf_report = classification_report(GT_label, Ens_DEIT_label)
            f1 = f1_score(GT_label, Ens_DEIT_label, average='macro')

            f = open(test_main.params.test_outpath + 'Ensemble_test_report_' + name2 + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            filenames_out = im_names[0]
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_label[jj]:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.params.test_outpath + '/' + name2 + name + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

    def initialize_model(self, train_main, test_main, data_loader, lr):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if data_loader.class_weights_tensor is not None:
            self.criterion = nn.CrossEntropyLoss(data_loader.class_weights_tensor)
        elif test_main is None:
            class_weights_tensor = torch.load(train_main.params.main_param_path + '/class_weights_tensor.pt')
            self.criterion = nn.CrossEntropyLoss(class_weights_tensor)
        else:
            class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')
            self.criterion = nn.CrossEntropyLoss(class_weights_tensor)

        torch.cuda.set_device(train_main.params.gpu_id)
        self.model.cuda(train_main.params.gpu_id)
        self.criterion = self.criterion.cuda(train_main.params.gpu_id)
        # Observe that all parameters are being optimized
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=train_main.params.weight_decay)

    def load_model_and_run_prediction(self, train_main, test_main, data_loader):
        self.import_deit_models_for_testing(train_main, test_main)
        if test_main.params.finetuned == 0:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen(test_main, data_loader, 'original')
            else:
                self.run_ensemble_prediction_on_unseen(test_main, data_loader, 'original')

        elif test_main.params.finetuned == 1:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen(test_main, data_loader, 'tuned')
            else:
                self.run_ensemble_prediction_on_unseen(test_main, data_loader, 'tuned')

        elif test_main.params.finetuned == 2:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen(test_main, data_loader, 'finetuned')
            else:
                self.run_ensemble_prediction_on_unseen(test_main, data_loader, 'finetuned')
        else:
            print('Choose the correct finetune label')

    def load_model_and_run_prediction_with_y(self, train_main, test_main, data_loader):
        self.import_deit_models_for_testing(train_main, test_main)
        if test_main.params.finetuned == 0:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen_with_y(test_main, data_loader, 'original')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(test_main, data_loader, 'original')

        elif test_main.params.finetuned == 1:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen_with_y(test_main, data_loader, 'tuned')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(test_main, data_loader, 'tuned')

        elif test_main.params.finetuned == 2:
            self.initialize_model(train_main, test_main, data_loader, train_main.params.lr)
            if test_main.params.ensemble == 0:
                self.run_prediction_on_unseen_with_y(test_main, data_loader, 'finetuned')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(test_main, data_loader, 'finetuned')
        else:
            print('Choose the correct finetune label')


def adjust_learning_rate(optimizer, epoch, lr, warmup, disable_cos, epochs):
    lr = lr
    if epoch < warmup:
        lr = lr / (warmup - epoch)
    elif not disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_images(data, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    print(data[1])
    ax.imshow(make_grid((data[0].detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=10, min_lr=1e-10, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# class LRScheduler:
#     """
#     Learning rate scheduler. If the validation loss does not decrease for the
#     given number of `patience` epochs, then the learning rate will decrease by
#     by given `factor`.
#     """
#
#     def __init__(
#             self, optimizer, patience=10, min_lr=1e-10, factor=0.5
#     ):
#         """
#         new_lr = old_lr * factor
#         :param optimizer: the optimizer we are using
#         :param patience: how many epochs to wait before updating the lr
#         :param min_lr: least lr value to reduce to while updating
#         :param factor: factor by which the lr should be updated
#         """
#         self.optimizer = optimizer
#         self.patience = patience
#         self.min_lr = min_lr
#         self.factor = factor
#         self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode='min',
#             patience=self.patience,
#             factor=self.factor,
#             min_lr=self.min_lr,
#             verbose=True
#         )
#
#     def __call__(self, val_loss):
#         self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=40, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def cls_train(train_loader, model, criterion, optimizer, clip_grad_norm):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    targets = []
    outputs = []

    for i, (images, target) in enumerate(train_loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images, target = images.to(device), target.to(device)

        output, x = model(images)
        # output = model(images)  # to run it on CSCS and colab

        loss = criterion(output, target.long())

        acc1 = accuracy(output, target)

        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2)

        optimizer.step()

        outputs.append(output)
        targets.append(target)

    outputs = torch.cat(outputs)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.argmax(outputs, axis=1)

    targets = torch.cat(targets)
    targets = targets.cpu().detach().numpy()

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    return avg_acc1, avg_loss, outputs, targets


def cls_validate(val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    targets = []
    outputs = []

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images, target = images.to(device), target.to(device)

            output = model(images)
            # loss = criterion(output, target)
            loss = criterion(output, target.long())
            acc1 = accuracy(output, target)

            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            outputs.append(output)
            targets.append(target)

        outputs = torch.cat(outputs)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = torch.cat(targets)
        targets = targets.cpu().detach().numpy()

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    return avg_acc1, avg_loss, outputs, targets, total_mins


def cls_predict(val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    outputs = []
    targets = []
    probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images, target = images.to(device), target.to(device)
            targets.append(target)

            output = model(images)
            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return avg_acc1, targets, outputs, probs


def cls_predict_on_unseen(test_loader, model):
    model.eval()
    outputs = []
    probs = []
    time_begin = time()
    with torch.no_grad():
        for i, (images) in enumerate(test_loader):
            device = torch.device("cuda")
            # images = torch.stack(images).to(device)
            images = images.to(device)

            output = model(images)
            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)

    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return outputs, probs


def cls_predict_on_unseen_with_y(val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    outputs = []
    targets = []
    probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images, target = images.to(device), target.to(device)
            targets.append(target)

            output = model(images)
            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return avg_acc1, targets, outputs, probs
