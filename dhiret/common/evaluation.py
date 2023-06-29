import argparse
import csv
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from annoy import AnnoyIndex
import torch
from tqdm import tqdm

from dhiret.common.utils import load_model, add_embedding_to_annoy_index
from dhiret.common.dataset import RetrievalEvaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image retrieval performance.")
    parser.add_argument("csv_file", type=Path, help="Path to the .csv file containing image metadata.")
    parser.add_argument("image_folder", type=Path, help="Path to the folder containing images.")
    parser.add_argument("model_name", type=str, help="Name of the model to use.")
    parser.add_argument("model_version", type=str, help="Model version (architecture modification).")
    parser.add_argument("--output_size", type=int, default=None, help="Model output size.")
    parser.add_argument("--weights_path", type=Path, default=None, help="Path to the model weights (optional).")
    parser.add_argument("--image_resize", type=int, default=256, help="Resize images to this size (optional).")
    parser.add_argument("--clip_dataset_and_epoch", type=str, default=None, help="Which clip dataset was used for training plus the number of epochs.")
    parser.add_argument("--print_results", type=Path, default=None, help="Print results to csv")

    args = parser.parse_args()
    return args

def main(args):
    evaluate(
        csv_file=args.csv_file,
        image_folder=args.image_folder,
        model_name=args.model_name,
        model_version=args.model_version,
        weights_path=args.weights_path,
        output_size=args.output_size,
        image_resize=args.image_resize,
        clip_dataset_and_epoch=args.clip_dataset_and_epoch,
        print_results=args.print_results
    )

def evaluate(csv_file, image_folder, model_name, model_version, output_size=None, weights_path=None, image_resize=256, clip_dataset_and_epoch=None, print_results=None):
    # check gpu availability
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Load the appropriate model
    model = load_model(model_name, model_version, output_size, weights_path, clip_dataset_and_epoch)
    model.to(device)
    model.eval()

    # Create the RetrievalEvaluation dataset
    transform = transforms.Compose([
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor()])
    dataset = RetrievalEvaluation(csv_file=csv_file, img_dir=image_folder, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize the Annoy index and image name list
    embedding_size = model.embed_dim
    index = AnnoyIndex(embedding_size, 'angular')
    image_name_list = []

    # Create embeddings for images using the DataLoader and add them to the Annoy index
    for batch in tqdm(dataloader):
        images = batch["image"].to(device=device)
        img_names = batch["image_name"]

        embeddings = model.forward_single(images)
        embeddings.to("cpu")
        for i, embedding in enumerate(embeddings):
            add_embedding_to_annoy_index(index, embedding, img_names[i], image_name_list)

    index.build(10)  # Adjust the number of trees as needed

    # Evaluate the retrieval performance
    pi_map, sc_map = dataset.evaluate_retrieval(annoy_index=index, image_name_list=image_name_list)
    print(f"Mean Average Precision for L1+L2: {(pi_map+sc_map)/2:.4f}")
    print(f"Mean Average Precision for Primary Instance: {pi_map:.4f}")
    print(f"Mean Average Precision for Secondary Categories: {sc_map:.4f}")
    if print_results is not None:
        with open(print_results, "a") as f:
            writer = csv.writer(f)
            if clip_dataset_and_epoch:
                writer.writerow([
                    model_name, model_version, clip_dataset_and_epoch, (pi_map+sc_map)/2, pi_map, sc_map
                ])
            else:
                writer.writerow([
                    model_name, model_version, weights_path, (pi_map+sc_map)/2, pi_map, sc_map
                ])
    return pi_map, sc_map

if __name__ == "__main__":
    args = parse_args()
    main(args)