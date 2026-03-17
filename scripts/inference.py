"""
Thin inference entry: argparse, optional TripoSG whole mesh, URDFInference run.
Run from repo root: python scripts/inference.py --image_path ... --model_path ...
"""
import os
import sys
import argparse
import random
import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# TripoSG as top-level triposg when needed
_triposg_root = os.path.join(_root, "TripoSG")
if _triposg_root not in sys.path:
    sys.path.insert(0, _triposg_root)

from urdf_anything.inference import URDFInference, get_3d_triposg


def main():
    parser = argparse.ArgumentParser(description="URDF inference script")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dino_path", type=str, default="DINOv3")
    parser.add_argument(
        "--in_the_wild",
        action="store_true",
        help="use TripoSG to generate whole mesh from image",
    )
    parser.add_argument(
        "--whole_mesh_path",
        type=str,
        default=None,
        help="Whole mesh file (required when not --in_the_wild)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--no_save_meshes", action="store_true")
    parser.add_argument("--no_save_urdf", action="store_true")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="urdf_anything/model/URDFModel_config.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_tokens", type=int, default=512)
    parser.add_argument("--triposg_weights_dir", type=str, default=None)
    parser.add_argument("--rmbg_weights_dir", type=str, default=None)
    parser.add_argument("--eot_threshold", type=float, default=0.5)
    parser.add_argument("--max_reconstruction_attempts", type=int, default=3)
    parser.add_argument("--overlap_chamfer_threshold", type=float, default=3e-2)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for TripoSG and PyTorch",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.in_the_wild and not args.whole_mesh_path:
        raise ValueError(
            "When not using --in_the_wild, must provide --whole_mesh_path"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.in_the_wild:
        print("mode: in the wild (use TripoSG to generate 3D)")
        whole_mesh = None
        whole_mesh_path = None
    else:
        print("mode: use provided whole mesh")
        whole_mesh_path = args.whole_mesh_path
        if not os.path.exists(whole_mesh_path):
            raise FileNotFoundError(
                f"whole mesh file not found: {whole_mesh_path}"
            )
        whole_mesh = None

    inference = URDFInference(
        model_path=args.model_path,
        model_config_path=args.model_config_path,
        dino_path=args.dino_path,
        device=args.device,
        num_tokens=args.num_tokens,
        eot_threshold=args.eot_threshold,
        max_reconstruction_attempts=args.max_reconstruction_attempts,
        overlap_chamfer_threshold=args.overlap_chamfer_threshold,
    )

    if args.in_the_wild:
        print("use TripoSG to generate whole mesh from image...")
        whole_mesh = get_3d_triposg(
            image_path=args.image_path,
            device=args.device,
            seed=args.seed,
            triposg_weights_dir=args.triposg_weights_dir,
            rmbg_weights_dir=args.rmbg_weights_dir,
            # target_faces=10000, 
            output_dir=args.output_dir,
        )
        whole_mesh.export(os.path.join(args.output_dir, "whole.obj"))
        whole_mesh_path = None
    else:
        whole_mesh = None

    result = inference.infer_from_image(
        image_path=args.image_path,
        whole_mesh=whole_mesh,
        whole_mesh_path=whole_mesh_path,
        output_dir=args.output_dir,
        save_meshes=not args.no_save_meshes,
        save_urdf=not args.no_save_urdf,
    )
    print(f"\ninference completed! generated {result['num_links']} links")
    print(f"results saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
