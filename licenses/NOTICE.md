# Implementation attribution

The frozen encoder uses the official RiNALMo implementation and pretrained
RiNALMo-Mega weights supplied separately by the user. The local adapter converts
packed attention weights to the official non-flash layout and uses PyTorch
scaled dot product attention. RiNALMo is distributed under Apache 2.0; the
license copy is retained alongside this notice.

The MAC block in `src/m6a/models/m6a_mac.py` derives from the project's CircMAC
reference implementation. It combines attention, native Mamba and convolution;
the sequence prediction head uses aligned downsampling and upsampling with
U-Net skip connections. `mamba-ssm` is an external dependency and retains its own
license. Distribution rights for the project code remain with its authors.
