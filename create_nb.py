import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# XAl-ChangeNet Visualization\nInspect predictions and explanations."),
    nbf.v4.new_code_cell("import json\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport torch\nfrom PIL import Image\n\nfrom models.siamese_unet import SiameseResNet18UNet\nfrom scripts.explain import grad_cam as grad_cam_fn"),
    nbf.v4.new_code_cell("pairs_file = Path('data/xbd/pairs_example.json')  # update\nckpt_path = Path('checkpoints/latest.pth')\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
    nbf.v4.new_code_cell("with pairs_file.open('r', encoding='utf-8') as f:\n    record = json.load(f)[0]\nroot = pairs_file.parent\npre_path = root / record['pre_image']\npost_path = root / record['post_image']\nmask_path = root / record['mask']\n\npre_img = np.array(Image.open(pre_path).convert('RGB'))\npost_img = np.array(Image.open(post_path).convert('RGB'))\nmask_img = np.array(Image.open(mask_path).convert('L')) / 255.0"),
    nbf.v4.new_code_cell("from albumentations import Compose, Resize, Normalize\ntransform = Compose([Resize(512, 512), Normalize()])\npre_tensor = torch.from_numpy(transform(image=pre_img)['image'].transpose(2,0,1)).unsqueeze(0).float().to(device)\npost_tensor = torch.from_numpy(transform(image=post_img)['image'].transpose(2,0,1)).unsqueeze(0).float().to(device)\n\nmodel = SiameseResNet18UNet().to(device)\nckpt = torch.load(ckpt_path, map_location=device)\nmodel.load_state_dict(ckpt['model'])\nmodel.eval()\nwith torch.no_grad():\n    logits = model(pre_tensor, post_tensor)\n    probs = torch.sigmoid(logits)\npred_mask = probs.squeeze().cpu().numpy()"),
    nbf.v4.new_code_cell("fig, axes = plt.subplots(1, 4, figsize=(18, 6))\naxes[0].imshow(pre_img); axes[0].set_title('Pre'); axes[0].axis('off')\naxes[1].imshow(post_img); axes[1].set_title('Post'); axes[1].axis('off')\naxes[2].imshow(mask_img, cmap='gray'); axes[2].set_title('Ground Truth'); axes[2].axis('off')\naxes[3].imshow(pred_mask, cmap='magma'); axes[3].set_title('Prediction'); axes[3].axis('off')\nplt.show()"),
    nbf.v4.new_code_cell("grad_cam_fn(model, pre_tensor, post_tensor, device, Path('outputs') / 'viz_grad_cam.png', post_img)\nImage.open(Path('outputs') / 'viz_grad_cam.png')")
]
nbf.write(nb, 'notebooks/visualize.ipynb')
