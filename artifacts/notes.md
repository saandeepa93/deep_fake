## Hourglass architecture for motion transfer

### **After keypoint detection**


$$
x : \mathit{Source\;image\;frame}\\
x^\prime: Driving\;image\;frame\\
\Delta x: keypoints\;of\;source\;image\\
\Delta x^\prime: keypoints\;of\;driving\;image\\
\mathcal{H}: Heatmap\;image\;output\;of\;x\\
\mathcal{H}^\prime: Heatmap\;image\;output\;of\;x^\prime\\
\mathcal{F}: Optical\;flow\;output\;of\;x^\prime
$$


* Encode $\mathbf{x} \rightarrow \mathcal{\epsilon}$ encoded feature maps.
* Use  $\textbf{dense\_motion\_module}$ to generate $\mathcal{F}$ for optic flow using $\mathbf{x}$ , $\Delta x$ and $\Delta x^{\prime}$.

  + Run $\textbf{mask\_embedding}$ to create mask of the source image $\mathbf{x}$ using the keypoints $\Delta x$ and $\Delta x^{\prime}$.

    -  Run both $\mathcal{H}$ and $\mathcal{H}^{'}$ via a Gaussian method $(\textbf{kp2gaussian})$ and normalize
    - Take the difference $\mathcal{H} - \mathcal{H}^\prime$ as the final heatmap
    - Take the difference of mean of source and driving images $\Delta x - \Delta x^{\prime}$ as the $\textit{kp\_video\_diff}$
    - Add another dimension of zeros to the 10 channel heatmaps as background
    - Get the squished coordinates $\textit{grid}$ and add the delta values $\textit{kp\_video\_diff}$.
    - Run $\textbf{grid\_sample}(\mathbf{x}, \textit{kp\_video\_diff})$ to get the deformed input image as $\textit{prediction}$.

  + Run the keypoint detector $\Delta$ on $\textit{prediction}$ and retreive the heatmap values for the new deformed image as $\textit{mask}$.




*  Performs grid_sampling on the encoded image from (1) and $\mathcal{F}$ grid from (2) i.e.

$$
deformed = \textbf{grid\_sample}(\mathcal{\epsilon}, \mathcal{F})
$$


```
(encoder): Encoder(
      (down_blocks): ModuleList(
        (0): DownBlock3D(
          (conv): Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pool): AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        )
        (1): DownBlock3D(
          (conv): Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pool): AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        )
        (2): DownBlock3D(
          (conv): Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pool): AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        )
        (3): DownBlock3D(
          (conv): Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pool): AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        )
        (4): DownBlock3D(
          (conv): Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pool): AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        )
      )
    )
    (decoder): Decoder(
      (up_blocks): ModuleList(
        (0): UpBlock3D(
          (conv): Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): UpBlock3D(
          (conv): Conv3d(1024, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): UpBlock3D(
          (conv): Conv3d(512, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): UpBlock3D(
          (conv): Conv3d(256, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (4): UpBlock3D(
          (conv): Conv3d(128, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
          (norm): SynchronizedBatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv): Conv3d(35, 10, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
    )
  )

```

## Encoder-Decoder feature size
```
encoder
input features:  3
output features:  64


input features:  64
output features:  128


input features:  128
output features:  256


input features:  256
output features:  512


input features:  512
output features:  512


decoder
input features:  512
output features:  512


input features:  1024
output features:  256


input features:  512
output features:  128


input features:  256
output features:  64


input features:  128
output features:  32

```

