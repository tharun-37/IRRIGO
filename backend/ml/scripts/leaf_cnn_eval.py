import os
import argparse
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from train import LeafDataset, create_model, evaluate


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def eval_model(dataset_root, model_path, batch_size, device, save_cm=None):
    test_dir = os.path.join(dataset_root, 'test')
    transform = get_val_transform()

    test_dataset = LeafDataset(test_dir, transform=transform)
    class_names = test_dataset.class_cols
    num_classes = len(class_names)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

    checkpoint = load_checkpoint(model_path, device)
    model = create_model(num_classes, model_name='efficientnet_b0')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print('\nFinal Test Set Results:')
    print(f'Test Accuracy  : {test_acc*100:.2f}%')
    print(f'Test Precision : {test_prec*100:.2f}%')
    print(f'Test Recall    : {test_rec*100:.2f}%')
    print(f'Test F1-Score  : {test_f1*100:.2f}%')

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print('\nDetailed Classification Report:')
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(cm)
        if save_cm:
            np.savetxt(save_cm, cm, fmt='%d', delimiter=',')
            print(f'Confusion matrix saved to {save_cm}')
    except Exception:
        pass


def infer_image(image_path, dataset_root, model_path, device, topk=5):
    transform = get_val_transform()
    checkpoint = load_checkpoint(model_path, device)

    # determine classes
    # prefer class_names in checkpoint, otherwise read from test folder csv
    class_names = checkpoint.get('class_names', None)
    if class_names is None:
        # try reading from test folder
        test_csv = os.path.join(dataset_root, 'test', '_classes.csv')
        if os.path.exists(test_csv):
            import pandas as pd
            df = pd.read_csv(test_csv)
            cols = [c.strip() for c in df.columns]
            class_names = cols[1:]
        else:
            raise RuntimeError('Could not determine class names')

    num_classes = len(class_names)
    model = create_model(num_classes, model_name='efficientnet_b0')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    topk_idx = np.argsort(probs)[-topk:][::-1]
    topk_mapping = {class_names[i]: float(probs[i]) for i in topk_idx}
    # top-1 prediction
    top1_idx = int(np.argmax(probs))
    prediction = class_names[top1_idx]
    probability = float(probs[top1_idx])

    return {
        'prediction': prediction,
        'probability': probability,
        'topk': topk_mapping,
        'all_probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Test / inference script for leaf disease model')
    parser.add_argument('--dataset-root', default=os.path.dirname(__file__), help='Path to dataset root (default: script folder)')
    parser.add_argument('--model', default=os.path.join(os.path.dirname(__file__), 'best_leaf_disease_model.pth'), help='Path to saved model checkpoint')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default=None, choices=['cpu','cuda'], help='Device to run on (auto if omitted)')
    parser.add_argument('--single', help='Path to a single image for inference')
    parser.add_argument('--save-cm', help='Optional path to save confusion matrix CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')

    if args.single:
        res = infer_image(args.single, args.dataset_root, args.model, device)
        print(json.dumps(res, indent=2))
        return

    # Interactive prompt: let user supply an image path, or run full evaluation
    try:
        user_input = input('Enter path to an image to classify (leave empty to evaluate the test set): ').strip()
    except Exception:
        user_input = ''

    if user_input:
        # allow users to paste quoted paths; strip surrounding quotes and common URL prefixes
        user_input = user_input.strip().strip('"\'"')
        lowered = user_input.lower()
        if lowered.startswith('file:///'):
            # file:///C:/path or file:///home/user/path
            user_input = user_input[8:]
        elif lowered.startswith('file://'):
            user_input = user_input[7:]

        if not os.path.isabs(user_input):
            user_input = os.path.join(os.getcwd(), user_input)

        res = infer_image(user_input, args.dataset_root, args.model, device)
        print(json.dumps(res, indent=2))
    else:
        eval_model(args.dataset_root, args.model, args.batch_size, device, save_cm=args.save_cm)


if __name__ == '__main__':
    main()
