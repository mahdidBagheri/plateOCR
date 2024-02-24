import torch
from Config.DatasetConfig import batch_size, prediction_head_num, prediction_length

class CTCLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
        a=0


    def forward(self, prediction, lable):
        T, B, H = prediction.size()
        Tl, Bl, Hl = lable.size()

        T = prediction_head_num
        N = 8
        S = prediction_length
        S_min = 1

        # pred_sizes = torch.LongTensor([[39 for i in range(64)]])

        input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long)
        target_lengths = torch.full(size=(batch_size,), fill_value=S, dtype=torch.long)

        synth_lbl = self.synthesize_lable(lable) # size -> 8 * 7
        synth_pred = prediction.transpose(0,1) # size -> 64 * 8 * 39
        loss = self.ctc_loss(synth_pred, synth_lbl, input_lengths, target_lengths)
        return loss


    def synthesize_lable(self, input):
        batch_size, seq_length, num_classes = input.size()
        output = torch.zeros((batch_size, seq_length))
        for b, batch in enumerate(input):
            for s, seq in enumerate(batch):
                output[b,s] = torch.argmax(seq)
        return output

    def synthesize_pred(self, input):
        batch_size, seq_length, num_classes = input.size()
        output = torch.zeros((batch_size, seq_length, num_classes))
        for b, batch in enumerate(input):
            for s, seq in enumerate(batch):
                output[b,s, torch.argmax(seq)] = 1.0
        return output