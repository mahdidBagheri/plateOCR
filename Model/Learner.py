from tqdm import tqdm
from collections import OrderedDict
import torch
import numpy as np
from matplotlib import pyplot as plt

from Utils.Eval import Eval
from Utils.Utils import onehot_to_classlable, class_lable_to_text
from Config.ModelConfig import device

class Learner():
    def __init__(self, model, loss, optimizer, lr_scheduler, train_loader, test_loader):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader

    def run_epoch(self, epoch, val):
        self.epoch = epoch

        if not val:
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"train epoch {epoch}")

        else:
            self.model.eval()
            pbar = tqdm(self.test_loader, desc=f"val epoch {epoch}")


        outputs = []
        losses = []
        accs = []
        for i , batch in enumerate(pbar):
            if not val:
                loss, acc = self.train_step(batch)
            else:
                loss, acc = self.test_step(batch)

            losses.append(loss.item())
            accs.append(acc)
            output = OrderedDict({'loss': abs(np.mean(losses)), 'acc': np.mean(accs), 'lr':self.lr_scheduler.get_last_lr()})
            pbar.set_postfix(output)
            outputs.append(output)

        if not val:
            result = self.train_end(outputs)
        else:
            result = self.test_end(outputs)

        return result


    def train_step(self, batch):
        loss, acc = self.run_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer_step()
        return loss, np.mean(acc)

    def train_end(self, outputs):
        self.lr_scheduler.step()
        return outputs

    def test_step(self, batch):
        loss, acc = self.run_batch(batch, val=True)
        return loss, acc


    def test_end(self, outputs):
        return outputs

    def optimizer_step(self):
        self.optimizer.step()

    def run_batch(self, batch, val=False):
        input = batch[0].to(device)
        target = batch[1].to(device)

        if(val):
            with torch.no_grad():
                output = self.model(input)
        else:
            output = self.model(input)

        acc = self.calc_accuracy(output, target)

        loss = self.loss(output, target)
        return loss, acc

    def calc_accuracy(self, output, target):
        evaluator = Eval()

        char_accs = []
        for i, batch in enumerate(target):
            labels = onehot_to_classlable(target[i])
            predictions = onehot_to_classlable(output[i])
            truth_word = class_lable_to_text(labels)
            predicted_word = class_lable_to_text(predictions)
            acc = evaluator.char_accuracy(predicted_word, truth_word)
            char_accs.append(acc)
        return char_accs

    def draw_results(self, outputs):
        losses = []
        accs = []
        for output in outputs:
            losses.append(output["loss"])
            accs.append(output["acc"])

        plt.figure()
        plt.plot(losses)
        plt.savefig()

        plt.figure()
        plt.plot(losses)
        plt.savefig()

        pass