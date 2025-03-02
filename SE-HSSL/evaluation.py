import torch
import torch.nn.functional as F
from logreg import LogReg

def masked_accuracy(logits, labels):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()

def accuracy(logits, labels, masks):
    accs = []
    for mask in masks:
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs

def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100,lr_weight=0.0):
    z=z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=lr_weight)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        accs = accuracy(logits, labels, masks)

    return accs

def node_classification_eval(model,data,num_splits=20,lr=0.005,max_epoch=100,lr_weight=0.0):
    
    model.eval()
    z,_ = model(data.features, data.hyperedge_index) 
    
    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(z, data.labels, masks, lr=lr, max_epoch=max_epoch,lr_weight=lr_weight))
    return accs 