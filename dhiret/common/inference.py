import argparse
from pathlib import Path
import json
import torch
from torchvision import transforms
from tqdm import tqdm

from dhiret.common.utils import load_model, build_annoy_index, load_image, add_embedding_to_annoy_index, save_annoy_index

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images.")
    parser.add_argument("image_folder", type=Path, help="Path to the folder containing images.")
    parser.add_argument("embeddings_folder", type=Path, help="Path to the embedding folder.")
    parser.add_argument("model_name", type=str, help="Name of the model to use.")
    parser.add_argument("model_version", type=str, help="Model version (architecture modification).")
    parser.add_argument("--output_size", type=int, default=None, help="Model output size.")
    parser.add_argument("--weights_path", type=Path, default=None, help="Path to the model weights (optional).")
    parser.add_argument("--image_resize", type=int, default=256, help="Resize images to this size (optional).")
    parser.add_argument("--clip_dataset_and_epoch", type=str, default=None,
                        help="Which clip dataset was used for training plus the number of epochs.")

    args = parser.parse_args()
    return args

def main(args):
    run_inference(
        image_folder=args.image_folder,
        embeddings_folder=args.embeddings_folder,
        model_name=args.model_name,
        model_version=args.model_version,
        output_size=args.output_size,
        weights_path=args.weights_path,
        image_resize=args.image_resize,
        clip_dataset_and_epoch=args.clip_dataset_and_epoch
    )

def run_inference(image_folder, embeddings_folder, model_name, model_version, output_size=None, weights_path=None, image_resize=None, clip_dataset_and_epoch=None):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the appropriate model
    model = load_model(model_name, model_version, output_size, weights_path, clip_dataset_and_epoch)
    model.to(device)

    # Load the index and image_name list -> the name of these files is based on model_name + model_version
    index_file_path = embeddings_folder / f"{model_name}_{model_version}_index.ann"
    image_name_list_file_path = embeddings_folder / f"{model_name}_{model_version}_image_name_list.json"

    embedding_size = model.embed_dim
    index = build_annoy_index(embedding_size, index_file_path)

    if image_name_list_file_path.exists():
        with open(image_name_list_file_path, "r") as f:
            image_name_list = json.load(f)
    else:
        image_name_list = []

    # Process images from the folder
    image_folder = Path(image_folder)
    # '.jpg', '.jpeg', '.png', '.tif'
    # image_paths = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.tif')
    image_paths = [p for ext in image_extensions for p in image_folder.glob(f'**/{ext}')]
    convert_to_tensor = transforms.Compose([transforms.Resize((image_resize, image_resize)),transforms.ToTensor()])

    for image_path in tqdm(image_paths):
        # Check if image was already inferenced
        if str(image_path) not in image_name_list:
            image = load_image(image_path, image_resize)
            image = convert_to_tensor(image)
            image = image.to(device)
            embedding = model.embed(image)
            embedding.cpu()
            add_embedding_to_annoy_index(index, embedding, str(image_path), image_name_list)

    # Save the annoy index and the image name list
    save_annoy_index(index, index_file_path)

    with open(image_name_list_file_path, "w") as f:
        json.dump(image_name_list, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)



