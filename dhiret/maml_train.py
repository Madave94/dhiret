import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from learn2learn.algorithms import MAML
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gc
import argparse
import timm
from pathlib import Path

from dhiret.models.maml import SiameseNet, ContrastiveLoss
from dhiret.common.dataset import VLPSiameseDataset, TaskSelector
from dhiret.common.evaluation import evaluate

def training_loop(task_selector_dataloader, maml, criterion, device, meta_optimizer, num_support_shots, num_shots, writer, epoch):
    shot_counter = 0
    meta_loss = None
    with tqdm(task_selector_dataloader, desc=f"Epoch {epoch + 1}", unit="iteration") as progress_bar:
        for batch_idx, task_data in enumerate(progress_bar):  # Iterate over the whole dataset with respect to num_shots
            if shot_counter == 0:
                task_selector.random_task()  # Randomly select a task
                task_losses = []
                learner = maml.clone()

                # Perform inner loop adaptation
            img1, img2, label = task_data["image1"].to(device), task_data["image2"].to(device), task_data[
                "label"].to(device)
            output1, output2 = learner(img1, img2)
            loss = criterion(output1, output2, label)

            # If the current shot is in the support set, adapt the learner
            if shot_counter < num_support_shots:
                learner.adapt(loss)
            # If the current shot is in the query set, evaluate the learner and accumulate the loss
            else:
                task_losses.append(loss)

            del img1, img2, label, output1, output2, loss
            gc.collect()
            shot_counter += 1  # Increment the shot counter

            if shot_counter == num_shots:  # If the shot counter reaches num_shots, update the meta-parameters
                meta_optimizer.zero_grad()
                meta_loss = torch.stack(task_losses).mean()
                meta_loss.backward()
                meta_optimizer.step()

                torch.cuda.empty_cache()  # Add this line to clear the cache after each iteration
                shot_counter = 0  # Reset the shot counter
                writer.add_scalar("Training Loss", meta_loss, (batch_idx + (epoch + 1) * len(task_selector_dataloader)))
            progress_bar.set_postfix(meta_loss=meta_loss.item() if meta_loss is not None else "N/A")

def validation_loop(val_task_selector_dataloader, maml, criterion, device, num_support_shots, num_shots):
    val_shot_counter = 0
    val_losses = []

    with tqdm(val_task_selector_dataloader, desc=f"Validation",
              unit="iteration", leave=False) as val_progress_bar:
        for val_task_data in val_progress_bar:
            if val_shot_counter == 0:
                val_task_selector.random_task()
                learner = maml.clone()

            # Perform inner loop adaptation
            img1, img2, label = val_task_data["image1"].to(device), val_task_data["image2"].to(device), \
                                val_task_data["label"].to(device)
            output1, output2 = learner(img1, img2)
            val_loss = criterion(output1, output2, label)

            if val_shot_counter < num_support_shots:
                learner.adapt(val_loss)
            else:
                val_losses.append(val_loss.item())

            # Free up memory
            del img1, img2, label, output1, output2, val_loss
            torch.cuda.empty_cache()
            gc.collect()

            val_shot_counter += 1
            if val_shot_counter == num_shots:
                val_shot_counter = 0

    # Calculate the mean validation loss
    mean_val_loss = np.mean(val_losses)
    return mean_val_loss

def run_evaluation(args, epoch, writer):
    csv_evaluate_file = Path("data/dhreaal/dhreaal.csv")
    test_image_folder = Path("data/dhreaal/test")
    model_name = "siamese"
    model_version = args.backbone
    weights_path = Path("{}_os{}_tep{}_shots{}_supshots{}_epoch{}.pth".format(args.backbone, args.output_shape, args.num_epochs,
                                                                     args.num_shots, args.num_support_shots, epoch+1))
    pi_map, sc_map = evaluate(csv_evaluate_file, test_image_folder, model_name, model_version, args.output_shape,
                              weights_path, image_resize=args.image_resize)
    writer.add_scalar("L1+L2 mAP", (pi_map + sc_map) / 2, epoch)
    writer.add_scalar('Primary Instance mAP', pi_map, epoch)
    writer.add_scalar("Secondary Category mAP", sc_map, epoch)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for maml.")
    parser.add_argument("backbone", type=str, help="Name of the used backbone.")
    parser.add_argument("output_shape", type=int, help="Output shape used for adaptive pooling.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--num_shots", type=int, default=5, help="Number of shots to run on each task model. This includes query and support set.")
    parser.add_argument("--num_support_shots", type=int, default=3, help="Number of shots in the support set")
    parser.add_argument("--lr_inner", type=float, default=1e-4, help="Set the learning rate for the inner loop.")
    parser.add_argument("--lr_outer", type=float, default=1e-2, help="Set the learning rate for the outer loop.")
    parser.add_argument("--image_resize", type=int, default=256, help="Resize images to this size (optional).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader (optional).")
    parser.add_argument("--debug", action="store_true", help="Activate to set datasets in debug mode.")

    args = parser.parse_args()

    assert args.num_support_shots < args.num_shots, \
        "The number of support shots {} needs to be smaller then the number of shots {}.".format(args.num_support_shots, args.num_shots)
    assert args.num_shots >= 2, "Need at least two shots."
    assert args.num_support_shots >= 1, "Need at least one support shot."

    model_list = timm.list_models()
    assert args.backbone in model_list, "{} not in model list.".format(args.backbone)

    return args

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_model = SiameseNet(args.backbone, args.output_shape).to(device)
    Path("saved_models", args.backbone).mkdir(exist_ok=True)
    #siamese_model = SiameseEfficientNet("mobilenetv2_050").to(device)
    #siamese_model = SiameseEfficientNet("efficientnet_b0").to(device)
    criterion = ContrastiveLoss(2.0).to(device)
    maml = MAML(siamese_model, lr=args.lr_inner, first_order=False)
    meta_optimizer = torch.optim.Adam(maml.parameters(), lr=args.lr_outer)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=333, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    transform = transforms.Compose([
        transforms.Resize((args.image_resize, args.image_resize)),
        transforms.ToTensor()
    ])

    train_dataset_content_type = VLPSiameseDataset("data/dhmtic", csv_file="train.csv",
                                                   task="content_type" ,transform=transform, debug=args.debug)
    train_dataset_media_type = VLPSiameseDataset("data/dhmtic", csv_file="train.csv",
                                                   task="media_type" ,transform=transform, debug=args.debug)
    task_selector = TaskSelector(train_dataset_media_type, train_dataset_content_type)
    task_selector_dataloader = DataLoader(task_selector, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    val_dataset_content_type = VLPSiameseDataset("data/dhmtic", csv_file="val.csv",
                                                 task="content_type", transform=transform, debug=args.debug)
    val_dataset_media_type = VLPSiameseDataset("data/dhmtic", csv_file="val.csv",
                                               task="media_type", transform=transform, debug=args.debug)
    val_task_selector = TaskSelector(val_dataset_media_type, val_dataset_content_type)
    val_task_selector_dataloader = DataLoader(val_task_selector, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    num_epochs = args.num_epochs
    num_shots = args.num_shots
    num_support_shots = args.num_support_shots
    writer = SummaryWriter("runs/{}_os{}_ep{}_shots{}_supshots{}".format(
        args.backbone, args.output_shape, args.num_epochs, args.num_shots, args.num_support_shots))

    best_val_loss = None

    for epoch in range(num_epochs):
        training_loop(task_selector_dataloader, maml, criterion, device, meta_optimizer,
                      num_support_shots, num_shots, writer, epoch)
        mean_val_loss = validation_loop(val_task_selector_dataloader, maml, criterion, device, num_support_shots, num_shots)
        writer.add_scalar('Validation Loss', mean_val_loss, epoch+1)
        if best_val_loss == None or mean_val_loss < best_val_loss:
            print("New best model in epoch {} with loss {}.".format(epoch+1, mean_val_loss))
            best_val_loss = mean_val_loss
        torch.save(maml.module.state_dict(), Path("saved_models", args.backbone, "{}_os{}_tep{}_shots{}_supshots{}_epoch{}.pth".format(
        args.backbone, args.output_shape, args.num_epochs, args.num_shots, args.num_support_shots, epoch+1) ))
        run_evaluation(args, epoch, writer)
