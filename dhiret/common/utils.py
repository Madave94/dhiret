from PIL import Image
from annoy import AnnoyIndex
from pathlib import Path
import torch

from dhiret.models.clipzeroshot import get_clip
from dhiret.models.maml import SiameseNet
import dhiret.models.dino_architectures as vits

Image.MAX_IMAGE_PIXELS = None


def load_model(model_name, model_version, output_size=None, weights_path=None, clip_dataset_and_epoch=None):
    """
    Load a specified model based on the model_name, model_version, output_size, and weights_path.

    Args:
        model_name (str): The name of the model, e.g. 'clip', 'siamese', or 'dino'.
        model_version (str): The version of the model.
        output_size (int, optional): The output size for the model, if applicable. Defaults to None.
        weights_path (Path, optional): The path to the saved weights file, if applicable. Defaults to None.

    Returns:
        torch.nn.Module: The loaded model instance.

    Raises:
        ValueError: If the model_name or model_version are invalid.
    """
    assert isinstance(model_name, str), f"Expected model_name to be a string, got {type(model_name)}"
    assert isinstance(model_version, str), f"Expected model_version to be a string, got {type(model_version)}"
    assert output_size is None or isinstance(output_size,
                                             int), f"Expected output_size to be None or int, got {type(output_size)}"
    assert weights_path is None or isinstance(weights_path,
                                              Path), f"Expected weights_path to be None or Path, got {type(weights_path)}"

    if model_name == 'clip':
        model = get_clip(model_version, clip_dataset_and_epoch, 224)
    elif model_name == "siamese":
        model = SiameseNet(model_version, output_size=output_size)
        model_weight_full_path = Path("saved_models", model_version) / weights_path
        model.load_state_dict(torch.load(model_weight_full_path))
        print("Loaded pretrained weights from: {}".format( model_weight_full_path ))
    elif model_name == "dino":
        model = vits.__dict__[model_version]()
        model_weight_full_path = Path("saved_models", model_version) / weights_path
        model.load_state_dict(torch.load(model_weight_full_path))
        print("Loaded pretrained weights from: {}".format(model_weight_full_path))
    else:
        raise ValueError('Invalid model_name or model_version')

    return model

def build_annoy_index(embedding_size, index_file_path):
    index = AnnoyIndex(embedding_size, 'angular')
    if index_file_path.exists():
        index.load(str(index_file_path))
    return index

def save_annoy_index(index, index_file_path):
    index.build(10)  # Adjust the number of trees as needed
    index.save(str(index_file_path))

def add_embedding_to_annoy_index(index, embedding, image_name, image_name_list):
    if image_name not in image_name_list:
        index.add_item(len(image_name_list), embedding)
        image_name_list.append(image_name)

def load_image(image_path, image_resize=None):
    image = Image.open(image_path).convert("RGB")
    if image_resize is not None:
        image = image.resize((image_resize, image_resize))
    return image