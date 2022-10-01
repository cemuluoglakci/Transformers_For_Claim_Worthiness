from utils import custom_models, wandb_wrapper

import os
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import requests
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, log_loss,
matthews_corrcoef, average_precision_score)

class WorthinessChecker(wandb_wrapper.WandbWrapper):
    def __init__(self, best_run, constants):
        self.constants = constants
        self.config = self.get_optimized_config(best_run)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)

    def get_optimized_config(self, best_run):
        optimized_config_dict = best_run.config
        optimized_config = Dict2Class(optimized_config_dict)
        optimized_config.epochs = self.get_real_epoch(best_run)
        return optimized_config

    def get_real_epoch(self, best_run):
        print('''Epoch configuration of the best run:''')
        epochs=best_run.config['epochs']
        print(epochs)

        early_stop_list = best_run.history()[best_run.history()['early_stopped_at']>0]['early_stopped_at']
        early_stop_list.name='cumulative_epoch'
        epoch_of_fold=early_stop_list%best_run.config['epochs']
        epoch_of_fold.name='epoch_of_fold'
        fold_index=early_stop_list//best_run.config['epochs']+1
        fold_index.name='fold_index'

        print('''Early stopped at:''')
        print(pd.DataFrame([fold_index, early_stop_list,epoch_of_fold]))

        total_fold_count = len(self.constants.seed_list)*self.constants.fold_count
        non_stopped_fold_count = total_fold_count-len(early_stop_list)
        average_early_stop = (epoch_of_fold.sum()+non_stopped_fold_count*epochs)//fold_index.count()
        real_epoch = int(average_early_stop - self.constants.patience+1)

        print ('\nAverage epoch used as a reference for early stopping: ', real_epoch)

        return real_epoch

    def predict(self, sentence: str):
        if not hasattr(self, 'model'):
            print('Training the model...')
            self.train_full_model()
        #preprocess text
        input_ids = self.tokenizer.encode_plus(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]' or equivalent
                            max_length = self.config.max_token_length,           # 64? 4-128? Pad & truncate all sentences.
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = False,   # Do not Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       ).input_ids

        self.model.eval()
        probability = self.model(input_ids.to(self.constants.device)).item()
        isCheckWorthy = (probability> .5)

        print(sentence)
        if isCheckWorthy:
            print('This expression contains a check-worthy claim with a {:.2%} conficency '.format(probability))
        else:
            print('This expression DOES NOT contain a check-worthy claim with a {:.2%} conficency '.format(1-probability))
        return probability

    def get_embedding(self, sentence: str):
        if not hasattr(self, 'model'):
            print('Training the model...')
            self.train_full_model()
        #preprocess text
        input_ids = self.tokenizer.encode_plus(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]' or equivalent
                            #max_length = self.config.max_token_length,           # 64? 4-128? Pad & truncate all sentences.
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = False,   # Do not Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       ).input_ids

        self.model.eval()
        last_hidden_states = self.model.transformer(input_ids.to(self.constants.device)).last_hidden_state
        return last_hidden_states[0][0]

    def prediction_expression(self, sentence: str):
        probability = self.predict(sentence)
        isCheckWorthy = (probability> .5)
        reply = ""
        if isCheckWorthy:
            reply = 'This expression contains a check-worthy claim with a {:.2%} confidency '.format(probability)
        else:
            reply = 'This expression DOES NOT contain a check-worthy claim with a {:.2%} confidency '.format(1-probability)
        return reply

    def batch_predict(self, test_df: pd.DataFrame):
        if not hasattr(self, 'model'):
            print('Training the model...')
            self.train_full_model()

        self.model.eval()

        test_dataset = self.create_dataset(test_df)
        _, test_dataloader = self.ret_dataloader(pd.DataFrame(), test_dataset)

        probability_list = torch.Tensor(0)

        for batch in test_dataloader:
            probability, _ = self.evaluate_one_batch(self, batch)
            probability_list = torch.cat((probability_list, probability), axis=0)

        predictions =  [int(i > .5) for i in probability_list]
        test_df['predictions'] = predictions
        test_df['probability'] = probability_list

        return test_df

    def train_full_model(self):
        # clean gpu memory in any case if previous wandb run was crashed.
        torch.cuda.empty_cache()
        epochs = self.config.epochs

        train_df, test_df = self.get_data_from_file()

        train_dataset = self.create_dataset(train_df)
        test_dataset = self.create_dataset(test_df)

        train_dataloader, test_dataloader = self.ret_dataloader(train_dataset, test_dataset)
        model = custom_models.TransformerClassifier(self.config).to(self.constants.device)

        optimizer = self.ret_optim(model)
        scheduler = self.ret_scheduler(train_dataloader, optimizer)

        epoch_train_metrics_list = []
        epoch_test_metrics_list = []

        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            self.train_one_epoch(model, self.constants.device, train_dataloader, self.constants.loss_function, optimizer, scheduler)

            epoch_train_metrics = self.evaluate_one_epoch(model, train_dataloader)
            epoch_test_metrics = self.evaluate_one_epoch(model, test_dataloader)
            epoch_train_metrics_list.append(epoch_train_metrics)
            epoch_test_metrics_list.append(epoch_test_metrics)

            test_mAP = epoch_test_metrics['mAP'].loc[0]
            train_mAP = epoch_train_metrics['mAP'].loc[0]
            print("  Training mAP: {:.3f} - Test mAP: {:.3f}".format(train_mAP,test_mAP ))

        train_metrics = pd.concat(epoch_train_metrics_list)
        test_metrics = pd.concat(epoch_test_metrics_list)

        print('*** Training Metrics ***')
        print(train_metrics)

        print('*** Test Metrics ***')
        print(test_metrics)

        self.model = model

        return model

    def load_model(self, PATH: str):
        device = self.constants.device
        
        model = custom_models.TransformerClassifier(self.config).to(device)
        model.load_state_dict(torch.load(PATH, map_location=device))

        self.model = model

    def load_model_online(self, LINK: str):
        device = self.constants.device
        
        model = custom_models.TransformerClassifier(self.config).to(device)
        drive_response = requests.get(LINK)
        model.load_state_dict(torch.load(BytesIO(drive_response.content), map_location=device))

        self.model = model

    def load_raw_model(self):
        device = self.constants.device       
        model = custom_models.TransformerClassifier(self.config).to(device)
        self.model = model

    def get_data_from_file(self):
        data_version = self.config.data_version
        train_df = pd.read_csv(os.path.join(self.constants.parent_dir, 'Data','train_english_{}.tsv'.format(data_version)), delimiter='\t')
        test_df = pd.read_csv(os.path.join(self.constants.parent_dir, 'Data','test_english_{}.tsv'.format(data_version)), delimiter='\t')
        return train_df, test_df

    def create_dataset(self, df):
        max_token_length = self.config.max_token_length

        sentences = df.tweet_text.values
        labels = None
        try:
            labels = df.check_worthiness.values
        except:
            labels = df.claim_worthiness.values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]' or equivalent
                                max_length = max_token_length,           # 64? 4-128? Pad & truncate all sentences.
                                truncation=True,
                                padding = 'max_length',
                                return_attention_mask = False,   # Do not Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )

            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.tensor(labels).float()
        dataset = TensorDataset(input_ids, labels)
        return dataset


    def ret_dataloader(self, train_dataset, validation_dataset):

        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = self.config.batch_size # Trains with this batch size.
                )

        validation_dataloader = DataLoader(
                    validation_dataset, # The validation samples.
                    sampler = SequentialSampler(validation_dataset), # Pull out batches sequentially.
                    batch_size = self.config.batch_size # Evaluate with this batch size.
                )
        return train_dataloader, validation_dataloader

    def ret_optim(self, model):
        #print('Learning_rate = ',wandb.config.learning_rate )
        optimizer = torch.optim.AdamW(model.parameters(),
                          lr = self.config.learning_rate, 
                          eps = 1e-8 
                        )
        return optimizer

    def ret_scheduler(self,train_dataloader,optimizer):
        epochs = self.config.epochs
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        return scheduler

    def evaluate_one_epoch(self, model, dataloader):

        model.eval()

        total_eval_loss = 0

        probability_list = torch.Tensor(0)
        label_list = np.empty(0)

        # Evaluate data for one epoch
        for batch in dataloader:
            
            probability, loss = self.evaluate_one_batch(model, batch)

            # Accumulate the validation loss, probability and labels.
            total_eval_loss += loss.item()
            probability_list = torch.cat((probability_list, probability), axis=0)
            label_list = np.concatenate((label_list, batch[1]), axis=0)

        # Calculate and log metrics and loss.
        metrics_df = self.get_metrics(probability_list, label_list)
        metrics_df.loc[0,'loss'] = total_eval_loss / len(dataloader)

        return metrics_df

    def evaluate_one_batch(self, model, batch):
        b_input_ids = batch[0].to(self.constants.device)
        b_labels = batch[1].to(self.constants.device)

        with torch.no_grad():        
            probability = model(b_input_ids).flatten()
            loss = self.constants.loss_function(probability, b_labels)

        probability = probability.detach().cpu()
        return probability, loss
 

    def train_one_epoch(self,model, device, dataloader, loss_function, optimizer, scheduler):
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            model.zero_grad()

            probability = model(b_input_ids)
            loss = loss_function(probability.flatten(), b_labels)
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # del b_input_ids
            # del b_labels

        return total_train_loss / len(dataloader)

    def get_metrics(self,probability, label_list):
        metrics_dictionary = {}
        # predictions = np.argmax(probability.detach().cpu().numpy(), axis=0)
        predictions =  [int(i > .5) for i in probability]

        accuracy = accuracy_score(label_list, predictions)
        precision = precision_score(label_list, predictions, zero_division=0)
        recall = recall_score(label_list, predictions, zero_division=0)
        f1 = f1_score(label_list, predictions, zero_division=0)
        log_loss = metrics.log_loss(label_list, predictions)
        mcc = matthews_corrcoef(label_list, predictions)
        auc = roc_auc_score(label_list, probability)

        mAP = average_precision_score(label_list, probability)

        metric_df = pd.DataFrame(np.empty(0, dtype=self.constants.metric_types))
        metric_df.loc[0] = [mAP, auc, accuracy, precision, recall, f1, mcc, log_loss, 0]

        return metric_df

class Predictor(WorthinessChecker):
    def __init__(self, best_run, constants):
        self.constants = constants
        self.config = best_run
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])