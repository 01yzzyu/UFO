"""UniCoT (Fr0zencr4nE/UniCoT-7B-MoT) local adapter.

Uni-CoT **extends Bagel-7B-MoT** and ships the SAME inference stack: its
`inference_mdp_self_reflection.py` loads the model with the identical
`modeling.bagel` + `InterleaveInferencer` code as Bagel's app.py
(verified: https://github.com/Fr0zenCrane/UniCoT/blob/main/inference_mdp_self_reflection.py).

Therefore this adapter reuses the Bagel adapter verbatim, only the model_path
differs. Make the UniCoT repo importable (it provides `modeling`, `inferencer`,
`data`) and point model_path at the UniCoT weights.

Prerequisites:
    git clone https://github.com/Fr0zenCrane/UniCoT.git && cd UniCoT
    conda create -n unicot python=3.10 -y && conda activate unicot
    pip install -r requirements.txt
    pip install flash_attn==2.5.8 --no-build-isolation
    export PYTHONPATH=/path/to/UniCoT:$PYTHONPATH
    # weights: Fr0zencr4nE/UniCoT-7B-MoT  (or v0.2)

models.yaml entry:
    - {name: UniCoT, group: unified, provider: local, local_adapter: unicot,
       model_path: models/UniCoT-7B-MoT}

NOTE: UniCoT adds a self-reflection generation loop on top of Bagel. This adapter
uses the plain Bagel interleave inference (understanding + single-pass image
generation). If you need the full self-reflection pipeline, port the loop from
`inference_mdp_self_reflection.py`.
"""

from .bagel import BagelAdapter


class UniCoTAdapter(BagelAdapter):
    # Identical loading and inference to Bagel; only the weights differ.
    pass
