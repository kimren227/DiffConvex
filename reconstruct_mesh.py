import os
import argparse
import torch
import logging
import datetime
import numpy as np
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from scripts.load_xml import load_scene
from scripts.render import NVDRenderer
from scripts.geometry import compute_vertex_normals, compute_face_normals
from convexes import Convexes


def main(target_mesh_path, steps, num_convexes, num_planes, convex_size, std, densify_iter, save_iter, density_stop):
    assert os.path.exists(target_mesh_path), f"Mesh file not found: {target_mesh_path}"
    assert target_mesh_path.endswith(".ply"), f"Invalid mesh file type: {target_mesh_path}, only .ply files are supported"

    # Prepare folder for saving results
    now = datetime.datetime.now()
    scene_name = os.path.basename(target_mesh_path).split(".")[0]
    exp_name = f"{scene_name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_folder = os.path.join("results", exp_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Load scene
    SCENE_PATH = os.path.join(os.getcwd(), "scenes", "default", f"default.xml")
    scene_params = load_scene(SCENE_PATH, target_mesh_path)
    v_ref = scene_params["mesh-target"]["vertices"]
    n_ref = scene_params["mesh-target"]["normals"]
    f_ref = scene_params["mesh-target"]["faces"]

    # Render target images
    renderer = NVDRenderer(scene_params, shading=True, boost=3)
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)

    # Initialize convexes
    convex = Convexes(init_num_convex=num_convexes, 
                      init_num_planes_per_convex=num_planes,
                      std=std, convex_size=convex_size, lr=1e-4).cuda()

    with logging_redirect_tqdm():
        for it in (pbar:=trange(steps, desc="Optimizing convexes")):
            convex.optimizer.zero_grad()
            mesh = convex()
            v, f = mesh.get_mesh_verts_faces(0)
            face_normals = compute_face_normals(v, f)
            n = compute_vertex_normals(v, f, face_normals)
            opt_imgs = renderer.render(v, n, f.type(torch.int32))
            # L1 loss between rendered images and reference images
            loss = (opt_imgs - ref_imgs).abs().mean()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            loss.backward()
            convex.optimizer.step()

            # Densify, spawn and purge the convexes
            if it % save_iter == 0:
                convex.save_obj(
                    f"{experiment_folder}/convexes_{scene_name}_{it:04d}.obj")
                        
            if it % densify_iter == 0 and it < steps * density_stop:

                convex.densify_convexes_from_mesh(v, f)
                convex.purge_redundant_planes(convex.optimizer)
                convex.remove_small_convexes(1e-5, convex.optimizer)
                # Spawn new convexes
                current_num_convex = np.unique(
                    convex.convex_indices.cpu().detach().numpy()).shape[0]
                logging.info(
                    f"Spawning {num_convexes - current_num_convex} new convexes")
                convex.random_spawn(num_convexes - current_num_convex)

            if it % 10 == 0:
                convex.purge_redundant_planes(convex.optimizer)
                convex.remove_small_convexes(1e-5, convex.optimizer)

    convex.save_obj(f"{experiment_folder}/{scene_name}_final.obj")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mesh", type=str, required=True)
    argparser.add_argument("--steps", type=int, default=20000)
    argparser.add_argument("--num_convexes", type=int, default=32)
    argparser.add_argument("--num_planes", type=int, default=32)
    argparser.add_argument("--convex_size", type=float, default=0.1)
    argparser.add_argument("--std", type=float, default=0.1)
    argparser.add_argument("--densify_iter", type=int, default=1000)
    argparser.add_argument("--save_iter", type=int, default=100)
    argparser.add_argument("--density_stop", type=float, default=0.75)
    args = argparser.parse_args()

    main(args.mesh, 
         steps=args.steps, 
         num_convexes=args.num_convexes, 
         num_planes=args.num_planes, 
         convex_size=args.convex_size, 
         std=args.std,
         densify_iter=args.densify_iter, 
         save_iter=args.save_iter, 
         density_stop=args.density_stop)
