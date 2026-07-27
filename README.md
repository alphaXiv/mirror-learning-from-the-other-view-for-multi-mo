# MIRROR claim reproduction

This repository contains a bounded, public-data reproduction of
“MIRROR: Learning from the Other View for Multi-Modal Reasoning” (arXiv:2607.21552).
The experiment uses the MIT-licensed Geometry3K/InterGPS validation split and
Qwen3-VL-4B-Instruct. Publication results will be added after fresh Kubernetes
runs complete.

Formal evidence is produced only by the fixed command:

```bash
bash scripts/run_k8s.sh
```

The dataset is downloaded at run time and is not redistributed here.
