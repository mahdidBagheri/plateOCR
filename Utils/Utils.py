import torch

def onehot_to_classlable(onehot):
    seq_length, class_size = onehot.size()
    class_lables = torch.zeros((seq_length))

    for i, seq in enumerate(onehot):
        class_lables[i] = torch.argmax(onehot[i])

    return class_lables

def class_lable_to_text(class_lable):
    s = ""
    for i, idx in enumerate(class_lable):
        if(idx == 0):
            s += " "
        elif (idx > 0 and idx < 10):
            s += f"{int(idx)}"
            a = 0
        else:
            s += f"{chr(int(idx)+55)}"
            a = 0
    return s

def shrink_text(text):
    i = 0
    while (i < len(text)):
        if (text[i] == " "):
            text = text[:i] + text[i+1:]
        else:
            i += 1
    return text

def extend_text(text, size):
    repeat = round(size / (len(text)+2))

    extended = " "*repeat
    for c in text:
        for i in range(repeat):
            extended += c

    extended += " "*(size - len(extended))
    return extended
