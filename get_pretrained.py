# get pretrained weights for fine-tuning
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth',
            map_location='cpu')
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
torch.save(checkpoint,"/content/detr-r101_no-class-head.pth") # File to resume