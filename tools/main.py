import tyro

from vggt_rerun.api.inference_vggt import VGGTInferenceConfig, run_inference

if __name__ == "__main__":
    run_inference(tyro.cli(VGGTInferenceConfig))
