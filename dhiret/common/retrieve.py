import argparse
import json
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

from dhiret.common.utils import load_model, build_annoy_index, load_image

def parse_args():
    parser = argparse.ArgumentParser(description="Find similar images using precomputed embeddings.")
    parser.add_argument("input_image", type=Path, help="Path to the input image.")
    parser.add_argument("embeddings_folder", type=Path, help="Path to the embedding folder.")
    parser.add_argument("model_name", type=str, help="Name of the model to use.")
    parser.add_argument("model_version", type=str, help="Model version (architecture modification).")
    parser.add_argument("--k", type=int, default=5, help="Number of similar images to retrieve.")
    parser.add_argument("--output_size", type=int, default=None, help="Model output size.")
    parser.add_argument("--weights_path", type=Path, default=None, help="Path to the model weights (optional).")
    parser.add_argument("--image_resize", type=int, default=256, help="Resize images to this size (optional).")
    parser.add_argument("--visualize", action="store_true", help="Visualize the input and similar images (up to 5).")
    parser.add_argument("--clip_dataset_and_epoch", type=str, default=None,
                        help="Which clip dataset was used for training plus the number of epochs.")

    args = parser.parse_args()
    return args

def main(args):
    find_similar_images(
        input_image=args.input_image,
        embeddings_folder=args.embeddings_folder,
        model_name=args.model_name,
        model_version=args.model_version,
        output_size=args.output_size,
        weights_path=args.weights_path,
        image_resize=args.image_resize,
        k=args.k,
        visualize=args.visualize,
        clip_dataset_and_epoch=args.clip_dataset_and_epoch
    )

def find_similar_images(input_image, embeddings_folder, model_name, model_version, output_size, weights_path, image_resize, k, visualize, clip_dataset_and_epoch=None):
    if visualize and k > 5:
        print("Visualization is only supported for k <= 5. Disabling visualization.")
        visualize = False

    # Load the appropriate model
    model = load_model(model_name, model_version, output_size, weights_path, clip_dataset_and_epoch)

    # Load the index and image_name list -> the name of these files is based on model_name + model_version
    index_file_path = embeddings_folder / f"{model_name}_{model_version}_index.ann"
    image_name_list_file_path = embeddings_folder / f"{model_name}_{model_version}_image_name_list.json"

    embedding_size = model.embed_dim
    index = build_annoy_index(embedding_size, index_file_path)

    with open(image_name_list_file_path, "r") as f:
        image_name_list = json.load(f)

    image = load_image(input_image)
    convert_to_tensor = transforms.Compose([transforms.Resize((image_resize, image_resize)),transforms.PILToTensor()])
    image_tensor = convert_to_tensor(image)

    embedding = model.embed(image_tensor)

    similar_image_indices = index.get_nns_by_vector(embedding, k)
    similar_image_paths = [image_name_list[idx] for idx in similar_image_indices]

    print("Input image:", input_image)
    print(f"Top {k} similar images:")
    for i, img_path in enumerate(similar_image_paths, 1):
        print(f"{i}. {img_path}")
    if visualize:
        visualize_images(input_image, similar_image_paths)

def visualize_images(input_image_path, similar_image_paths):
    plt.figure(figsize=(15, 3))

    # Display the input image
    input_image_path = Path(input_image_path)
    input_image = Image.open(input_image_path).convert("RGB")
    input_image_np = np.array(input_image)
    input_image_base_name = input_image_path.name
    plt.subplot(1, len(similar_image_paths) + 1, 1)
    plt.imshow(input_image_np)
    plt.title(f"Input Image\n{input_image_base_name}")
    plt.axis("off")

    # Display the similar images
    for i, img_path_str in enumerate(similar_image_paths, 2):
        img_path = Path(img_path_str)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        image_base_name = img_path.name
        plt.subplot(1, len(similar_image_paths) + 1, i)
        plt.imshow(image_np)
        plt.title(f"Similar Image {i - 1}\n{image_base_name}")
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)