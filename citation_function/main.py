import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel
import os
from config import get_config
from data import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import seaborn as sns
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention, Transformer_CNN_RNN_Attention




class bert_Classification:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        
        # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.input_size = 768
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('../roberta-base',local_files_only=True)
        elif args.model_name == 'scibert':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        else:
            raise ValueError('unknown model')
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_RNN_Attention(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, labels, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        all_preds, all_targets = [], []
        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = torch.argmax(predicts, dim=1)
            n_correct += (preds == targets).sum().item()
            n_train += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        print(classification_report(all_targets, all_preds, target_names=labels))
        print(confusion_matrix(all_targets, all_preds))

        acc = n_correct / n_train
        p, r, f1, _ =  precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)


        return train_loss / n_train,acc, p, r, f1

    def _test(self, dataloader, labels,criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        all_preds, all_targets = [], []
        confusion_mat = None
        
        # Turn on the eval mode
        self.Mymodel.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                preds = torch.argmax(predicts, dim=1)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        report = classification_report(all_targets, all_preds, target_names=labels, output_dict=True)
        confusion_mat = confusion_matrix(all_targets, all_preds)

        acc = n_correct / n_test
        p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

        return test_loss / n_test,acc, p, r, f1, report, confusion_mat
      
    def run(self):
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader,labels = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        patience_step = 0
        l_tracc,l_teacc, l_trloss, l_teloss, l_epo = [], [], [], [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        classes = []  
        precision = [] 
        confusion_mat = None
        for epoch in range(self.args.num_epoch):
            
            train_loss, train_acc,train_pre,train_recall,train_f1 = self._train(train_dataloader, labels,criterion, optimizer)
            test_loss, test_acc, test_pre,test_recall,test_f1, report, matrix = self._test(test_dataloader,labels, criterion)
            
            # # 10-fold cross validation
            # avg_results = self._cross_validation(train_dataloader, test_dataloader, criterion, optimizer)

            l_epo.append(epoch), l_tracc.append(train_acc), l_teacc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)

            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                best_model_path = os.path.join(self.args.output, '{}_{}_best_model_mix.pt'.format(args.model_name, args.method_name))  
                torch.save(self.Mymodel.state_dict(), best_model_path)
                patience_step = 0
                classes = list(report.keys())[:-3]  
                precision = [report[cls]['precision'] for cls in classes] 
                confusion_mat = matrix
                
            else:
                patience_step += 1

            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f},p: {:.2f},r: {:.2f},f1: {:.2f}'.format(train_loss, train_acc * 100,train_pre*100,train_recall*100,train_f1*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f},p: {:.2f},r: {:.2f},f1: {:.2f}'.format(test_loss, test_acc * 100,test_pre*100,test_recall*100,test_f1*100))

            # self.logger.info('Average test loss: {:.4f}, accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, F1 score: {:.2f}'.format(
            #                     avg_results[0], avg_results[1] * 100, avg_results[2] * 100, avg_results[3] * 100, avg_results[4] * 100))
            
            if patience_step > self.args.patience_step:
                break

        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('best model saved at: {}'.format(best_model_path))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        
        plt.figure()
        bar_width = 0.2
        index = np.arange(len(labels))
        plt.bar(index, precision, bar_width, color='b', label='Precision')
        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Scores by class')
        plt.xticks(index + bar_width, labels, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig('figs/label_Precision.png')
       
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('figs/Matrix.png')
        
        # Draw the training process
        plt.figure()
        plt.plot(l_epo, l_teacc, label='test-accuracy')
        plt.plot(l_epo, l_tracc, label='train-accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('figs/acc.png')

        plt.figure()
        plt.plot(l_epo, l_teloss, label='test-loss')
        plt.plot(l_epo, l_trloss, label='train-loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('figs/loss.png')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    be_cls = bert_Classification(args, logger)
    be_cls.run()