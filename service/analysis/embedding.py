import torch

def get_image_embedding(images, processor, model):
    if not isinstance(images, list):
        images = [images]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings
