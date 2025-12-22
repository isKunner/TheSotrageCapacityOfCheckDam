<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />

<div>
  <h3 align="center">TheSotrageCapacityOfCheckDam</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <a href="https://github.com/isKunner">Kevin Chen</a>
  </p>
</div>

## The model construction
#### 2025.12.8
1. Most check dams can still be identified, even if the resolution is reduced.
2. Overly details textures may be invalid information.
3. The area occupied by check dams is very small, so directly identifying the check dams consumes computing power.
4. So we propose the first model, as bellow:

    | Model                 | Structure & Components                                                                                                                                                                                                                                                                                                                                           | Function & Purpose                                                                                                                                                                                                       |
    |:----------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | **ExtractCheckDam**   | *   **Downsampling**: `Conv2d(K=7, S=4, P=3)` applied 3 times.<br>*   **Feature Extraction**: Shared `nn.Sequential` of 4 `ResidualBlock`s (`3->64->64->32->1`).<br>*   **Upsampling/Fusion**: Gradually upsamples (x4) and adds features from coarser to finer scales, then interpolates to input size.<br>*   **Output**: `Sigmoid` activation.                | Generates a prior probability map highlighting likely check dam locations. Uses multi-scale context via downsampling and fuses features hierarchically to improve localization accuracy.                                 |
    | **ExtractSiltedLand** | *   **Feature Extraction**: Direct `nn.Sequential` of 4 `ResidualBlock`s (`3->64->64->32->1`).<br>*   **Output**: `Sigmoid` activation.                                                                                                                                                                                                                          | Generates a prior probability map highlighting likely silted land areas. Applies feature extraction directly to the full-resolution input to capture detailed spatial patterns associated with siltation.                |
    | **CheckDamNet**       | *   **Prior Generation**: Calls `ExtractCheckDam` and `ExtractSiltedLand`.<br>*   **Input Fusion**: Concatenates original input (3 channels), check dam prior (1 channel), and silted land prior (1 channel) -> 5 channels.<br>*   **Main Network**: Standard `UNet` takes the 5-channel fused input.<br>*   **Output**: `UNet`'s final logits (before Softmax). | Combines learned priors with the original image data. Feeds this enriched representation into a U-Net architecture for the final semantic segmentation task, aiming to accurately delineate check dams and silted areas. |
5. Due to the messy annotated data, a better loss function is needed at present.
## The Another Content

### Focal Loss

```
torch.manual_seed(42)

B, C, H, W = 2, 3, 2, 3
ignore_idx = 255

pred_logits = torch.randn(B, C, H, W, requires_grad=True)
target_labels = torch.randint(0, C, (B, H, W))
target_labels[0, 0, 0] = ignore_idx
target_labels[1, 1, 2] = ignore_idx

alpha_weights_list = [0.5, 1.5, 2.5]
alpha_weights_tensor = torch.tensor(alpha_weights_list, dtype=torch.float32)
print("pred_logis: ")
print(pred_logits)
print("target_labels: ")
print(target_labels)
print(f"Input logits shape: {pred_logits.shape}")
print(f"Target labels shape: {target_labels.shape}")
print(f"Ignore index: {ignore_idx}")
print(f"Alpha weights (list): {alpha_weights_list}")
print(f"Alpha weights (tensor): {alpha_weights_tensor}")
print("-" * 20)

'''
pred_logis: 
tensor([[[[ 1.9269,  1.4873,  0.9007],
          [-2.1055,  0.6784, -1.2345]],

         [[-0.0431, -1.6047, -0.7521],
          [ 1.6487, -0.3925, -1.4036]],

         [[-0.7279, -0.5594, -0.7688],
          [ 0.7624,  1.6423, -0.1596]]],


        [[[-0.4974,  0.4396,  0.3189],
          [-0.4245,  0.3057, -0.7746]],

         [[ 0.0349,  0.3211,  1.5736],
          [-0.8455, -1.2742,  2.1228]],

         [[-1.2347, -0.4879, -1.4181],
          [ 0.8963,  0.0499,  2.2667]]]], requires_grad=True)
target_labels: 
tensor([[[255,   0,   1],
         [  1,   1,   2]],

        [[  2,   0,   2],
         [  0,   2, 255]]])
Input logits shape: torch.Size([2, 3, 2, 3])
Target labels shape: torch.Size([2, 2, 3])
Ignore index: 255
Alpha weights (list): [0.5, 1.5, 2.5]
Alpha weights (tensor): tensor([0.5000, 1.5000, 2.5000])
--------------------
'''

# When an element in Targets (i.e., a pixel's true label) equals the ignore_index you specify, F.cross_entropy doesn't calculate the regular cross-entropy loss (i.e., -log(p_t)) for that pixel.
# Instead, it returns a loss value of 0 directly for that pixel. 
# pred_logits[0, :, 0, 1] = [1.4873, -1.6047, -0.5594]
# Softmax:
# exp(1.4873)≈4.4253
# exp(−1.6047)≈0.2010
# exp(−0.5594)≈0.5716
# Z=4.4253+0.2010+0.5716≈5.1979
# p0 = 4.4253/Z = 0.8541
# p1 = 0.2010/Z = 0.0387
# p2 = 0.5716/Z = 0.1100
# 真实类别是0，所以计算：
# CE=−log(0.8514)≈−(−0.1609)≈0.1609

ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

'''
cross_entropy
torch.Size([2, 2, 3])
tensor([[[0.0000, 0.1609, 1.9748],
         [0.3616, 2.4483, 0.4883]],

        [[1.8945, 0.8258, 3.2809],
         [1.6869, 0.9391, 0.0000]]], grad_fn=<NllLoss2DBackward0>)
'''

pt = torch.exp(-ce_loss)

'''
tensor([[[1.0000, 0.8514, 0.1388],
         [0.6966, 0.0864, 0.6137]],

        [[0.1504, 0.4379, 0.0376],
         [0.1851, 0.3910, 1.0000]]], grad_fn=<ExpBackward0>)
'''

focal_loss_base = (1 - pt) ** self.gamma * ce_loss  # Base focal component

'''
tensor([[[0.0000, 0.0036, 1.4647],
         [0.0333, 2.0433, 0.0729]],

        [[1.3676, 0.2610, 3.0388],
         [1.1202, 0.3483, 0.0000]]], grad_fn=<MulBackward0>)
'''

if self.alpha is not None:
    # --- Key difference: Manual alpha application ---
    flat_targets = targets.view(-1)  # Shape: [N*H*W]

    # --- FIX: Handle ignore_index in index_select ---
    # 1. Identify locations of ignore_index
    ignore_mask = (flat_targets == self.ignore_index)  # Shape: [N*H*W], Bool tensor

    # 2. Create safe targets by replacing ignore_index with a valid index (e.g., 0)
    safe_targets = flat_targets.clone()
    safe_targets[ignore_mask] = 0  # Any valid class index works here

    # 3. Perform index_select with safe targets
    alpha_per_pixel_flat = self.alpha.index_select(0, safe_targets)  # Shape: [N*H*W]

    # 4. (Optional but robust) Set alpha to 0 for ignored pixels
    #     Although ce_loss is 0 for these, setting alpha to 0 makes intent clearer
    alpha_per_pixel_flat = alpha_per_pixel_flat * (~ignore_mask).float()

    # Reshape back to original target shape
    alpha_per_pixel = alpha_per_pixel_flat.view(targets.shape)  # Shape: [N, H, W]

    # Apply alpha weight to the base focal loss
    focal_loss = alpha_per_pixel * focal_loss_base
else:
    focal_loss = focal_loss_base
    
'''
tensor([[[0.0000e+00, 1.7772e-03, 2.1970e+00],
         [4.9937e-02, 3.0650e+00, 1.8220e-01]],

        [[3.4189e+00, 1.3048e-01, 7.5970e+00],
         [5.6012e-01, 8.7071e-01, 0.0000e+00]]], grad_fn=<MulBackward0>)
'''    

```

### [pytorch lighting](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#)
#### [Lightning Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training)
1. When you convert to use Lightning, the code IS NOT abstracted - just organized. All the other code that’s not in the LightningModule has been automated for you by the Trainer.
    ```
    net = MyLightningModuleNet()
    trainer = Trainer()
    trainer.fit(net)
    ```
2. There are no .cuda() or .to(device) calls required. Lightning does these for you.
    ```
   # don't do in Lightning
    x = torch.Tensor(2, 3)
    x = x.cuda()
    x = x.to(device)
    
    # do this instead
    x = x  # leave it alone!
    
    # or to init a new tensor
    # move a newly created tensor relative to an existing one
    new_x = torch.Tensor(2, 3)
    new_x = new_x.to(x)
   ```
3. A LightningModule is a torch.nn.Module but with added functionality. Use it as such!
    ```
    net = Net.load_from_checkpoint(PATH)
    net.freeze()
    out = net(x)
    ```
   
4. The LightningModule has many convenient methods, but the core ones you need to know about are

    | Name                   | Description                                                       |
    |------------------------|-------------------------------------------------------------------|
    | __init__ and setup()   | Define initialization here                                        |
    | forward()              | To run data through your model only (separate from training_step) |
    | training_step()        | The complete training step                                        |
    | validation_step()      | The complete validation step                                      |
    | test_step()            | The complete test step                                            |
    | predict_step()         | The complete predict step                                         |
    | configure_optimizers() | To define your optimizer and learning rate scheduler              |

    4.1 Training Loop
    
    To activate the training loop, override the [training_step()](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step) method.
    ```
    class LightningTransformer(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss
   ```
   
    If you want to calculate epoch-level metrics and log them, use [log()](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log)

#### [TQDMProgressBar](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.TQDMProgressBar.html#tqdmprogressbar)
`classlightning.pytorch.callbacks.TQDMProgressBar(refresh_rate=1, process_position=0, leave=False)`

Bases: ProgressBar

This is the default progress bar used by Lightning. It prints to stdout using the tqdm package and shows up to four different bars:
- sanity check progress: the progress during the sanity check run
- train progress: shows the training progress. It will pause if validation starts and will resume when it ends, and also accounts for multiple validation runs during training when [val_check_interval](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer.params.val_check_interval) is used.
- validation progress: only visible during validation; shows total progress over all validation datasets.
- test progress: only active when testing; shows total progress over all test datasets.

Parameters:

| Name             | Type | Description                                                                                                        |
|------------------|------|--------------------------------------------------------------------------------------------------------------------|
| refresh_rate     | int  | Determines at which rate (in number of batches) the progress bars get updated. Set it to 0 to disable the display. |
| process_position | int  | Determines the position of the progress bar in the terminal.                                                       |Set this to a value greater than 0 to offset the progress bars by this many lines. This is useful when you have progress bars defined elsewhere and want to show all of them together.|
| leave            | bool | If set to True, leaves the finished progress bar in the terminal at the end of the epoch. Default: False           |

#### [EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping)

`classlightning.pytorch.callbacks.EarlyStopping(monitor, min_delta=0.0, patience=3, verbose=False, mode='min', strict=True, check_finite=True, stopping_threshold=None, divergence_threshold=None, check_on_train_epoch_end=None, log_rank_zero_only=False)`

Bases: [Callback](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback)

It must be noted that the patience parameter counts the number of validation checks with no improvement, and not the number of training epochs. Therefore, with parameters check_val_every_n_epoch=10 and patience=3, the trainer will perform at least 40 training epochs before being stopped.

Parameters:

| Name    | Type | Description                                                                                                                                                                                                                         |
|---------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| monitor | str  | Quantity to be monitored. The quantity **must be logged** in the training_step(), validation_step(), or test_step()methods of theLightningModule. <br> e.g. self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True) |
| mode    | str  | One of 'min', 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing and in 'max' mode it will stop when the quantity monitored has stopped increasing.                                        |


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

Here's the description of the project:

*

Here's why:

*

Here's summary:

*

How to start:

*

<p align="right">(<a href="#top">back to top</a>)</p>


This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for
the acknowledgements section. Here are a few examples.

* [Next.js](https://nextjs.org/)
* [React.js](https://reactjs.org/)
* [Vue.js](https://vuejs.org/)
* [Angular](https://angular.io/)
* [Svelte](https://svelte.dev/)
* [Laravel](https://laravel.com)
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

This is an example of how to list things you need to use the software and how to install them.

* npm
  ```sh
  npm install npm@latest -g
  ```

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't
rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos
work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (
and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites
to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge

[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members

[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge

[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers

[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/othneildrew

[product-screenshot]: images/screenshot.png