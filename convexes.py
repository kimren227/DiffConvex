import torch
import trimesh
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import HalfspaceIntersection, ConvexHull
from pytorch3d.structures import Meshes

import diff_convex


class Convexes(nn.Module):
    def __init__(self, init_num_convex=1, init_num_planes_per_convex=128, convex_size=0.1, std=0.01, init_translation=None, init_box_params=None, lr=1e-3):
        super(Convexes, self).__init__()
        self.init_num_planes_per_convex = init_num_planes_per_convex
        self.init_num_convex = init_num_convex
        if init_box_params is not None:
            planes, convex_indices, translations = self.init_from_boxes(
                init_box_params)

        else:
            planes, convex_indices, translations = self.init_convexes(
                init_num_convex, init_num_planes_per_convex, convex_size, std)
            if init_translation is not None:
                translations = torch.from_numpy(init_translation)
        self.planes = nn.Parameter(planes)
        self.translations = nn.Parameter(translations)
        self.convex_indices = convex_indices
        self.step_size = lr
        params = [{'params': [self.planes], 'lr': self.step_size, "name": "planes"},
                  {'params': [self.translations], 'lr': 1e-3, "name": "translations"}]

        self.optimizer = torch.optim.Adam(params, lr=self.step_size, eps=1e-15)

    def init_from_boxes(self, boxes):
        num_boxes = boxes.shape[0]
        planes = []
        convex_indices = []
        translations = []

        for i in range(num_boxes):
            vertices = boxes[i]
            translation = vertices.mean(dim=0)
            translations.append(translation.cpu().detach().numpy())
            vertices = vertices - translation
            vertices = vertices.cpu().detach().numpy()
            plane_equations = diff_convex.get_hull_equations(vertices.astype(np.float64),
                                                             np.asarray([0, 0, 0]).astype(np.float64)).reshape(-1, 4)
            good_equations = []
            for plane in plane_equations:
                if len(good_equations) == 0:
                    good_equations.append(plane)
                else:
                    good = True
                    for good_plane in good_equations:
                        if np.dot(plane[:3], good_plane[:3]) > 0.99:
                            good = False
                            break
                    if good:
                        good_equations.append(plane)
            good_equations = np.stack(good_equations, axis=0)
            planes.append(good_equations)
            convex_indices.append(np.ones(good_equations.shape[0]) * i)
        planes = np.concatenate(planes, axis=0)
        convex_indices = np.concatenate(convex_indices, axis=0)
        translations = np.stack(translations, axis=0)

        planes = torch.from_numpy(planes).float().cuda()
        translations = torch.from_numpy(translations).float().cuda()
        convex_indices = torch.from_numpy(convex_indices).long().cuda()
        return planes, convex_indices, translations

    def init_convexes(self, num_convex, num_planes, convex_size, std=0.1):
        convex_planes = []
        convex_plane_indices = []
        convex_translations = []
        for i in range(num_convex):
            planes = self.init_convex(num_planes, convex_size)
            translation = torch.randn(1, 3) * std
            convex_planes.append(planes)
            convex_translations.append(translation)
            convex_plane_indices.append(torch.ones(num_planes) * i)
        convex_planes = torch.cat(convex_planes, dim=0)
        convex_plane_indices = torch.cat(convex_plane_indices, dim=0)
        convex_translations = torch.cat(convex_translations, dim=0)

        return convex_planes, convex_plane_indices, convex_translations

    def init_convex(self, num_planes, convex_size):
        plane_normals = torch.randn(num_planes, 3)
        plane_normals = plane_normals / \
            plane_normals.norm(dim=-1, keepdim=True)
        plane_offsets = torch.ones((num_planes, 1)) * convex_size
        planes = torch.cat([plane_normals, plane_offsets], dim=-1)
        return planes

    def get_transforme_halfspaces(self, normalized_planes):
        plane_convex_indices = self.convex_indices  # [P]
        # [P, 3]
        plane_translations = self.translations[plane_convex_indices.long()]
        plane_offsets = torch.einsum(
            "pd,pd->p", normalized_planes[:, :-1], plane_translations)  # [P]
        plane_offsets = normalized_planes[:, -1] - plane_offsets  # [P]
        return torch.cat([normalized_planes[:, :-1], plane_offsets.unsqueeze(-1)], dim=-1)

    def get_mesh(self):
        planes = self.planes
        normals = F.normalize(planes[:, :-1], dim=-1)
        offsets = -torch.abs(planes[:, -1:]) - 1e-4
        planes = torch.cat([normals, offsets], dim=-1).cuda()

        transformed_planes = self.get_transforme_halfspaces(planes)

        convex = diff_convex.convex_to_mesh(np.ascontiguousarray(transformed_planes.cpu().detach().numpy().astype(np.float64)),
                                            np.ascontiguousarray(
                                                self.convex_indices.cpu().detach().numpy().astype(np.int32)),
                                            np.ascontiguousarray(self.translations.cpu().detach().numpy().astype(np.float64)))
        vertex_planes = transformed_planes[convex[0]].reshape(-1, 3, 4)
        faces = convex[1].reshape(-1, 3)
        faces = np.ascontiguousarray(faces)
        vertices = torch.linalg.solve(
            vertex_planes[:, :, :-1], -vertex_planes[:, :, -1])
        vertices = torch.stack(
            [vertices[:, 0], vertices[:, 1], vertices[:, 2]], dim=-1)
        faces = torch.from_numpy(faces).long().cuda()
        mesh = Meshes(vertices[None, ...], faces[None, ...])
        return mesh

    def random_spawn(self, num_convexes, bbox_min=None, bbox_max=None, optimizer=None):
        if num_convexes == 0:
            return
        if bbox_min is None or bbox_max is None:
            mesh = self.get_mesh()
            v, f = mesh.get_mesh_verts_faces(0)

            bbox_min = v.min(dim=0)[0]
            bbox_max = v.max(dim=0)[0]
        else:
            bbox_min = torch.tensor(bbox_min).float().cuda()
            bbox_max = torch.tensor(bbox_max).float().cuda()

        bbox_size = bbox_max - bbox_min
        random_positions = np.random.rand(
            num_convexes, 3) * bbox_size.cpu().detach().numpy() + bbox_min.cpu().detach().numpy()
        planes, convex_indices, translations = self.init_convexes(
            num_convexes, self.init_num_planes_per_convex, 0.1)
        translations = torch.from_numpy(random_positions).float().cuda()
        planes = planes.cuda()
        planes = torch.cat([self.planes, planes], dim=0)
        convex_indices = convex_indices.to(self.convex_indices.device)
        convex_indices = torch.cat(
            [self.convex_indices, convex_indices+self.convex_indices.max()+1], dim=0)
        translations = torch.cat([self.translations, translations], dim=0)

        optimizer = self.optimizer if optimizer is None else optimizer

        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            planes, name="planes")
        self.planes = optimizable_tensors["planes"]

        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            translations, name="translations")
        self.translations = optimizable_tensors["translations"]

        self.convex_indices = convex_indices

    def densify_convexes_from_mesh(self, vertices, faces, optimizer=None):

        vertices, faces = trimesh.remesh.subdivide_loop(
            vertices.cpu().detach().numpy(), faces.cpu().detach().numpy(), iterations=1)

        G = nx.Graph()
        for face in faces:
            for i in range(3):
                G.add_edge(face[i], face[(i+1) % 3])
                G.add_edge(face[i], face[(i+2) % 3])

        # find connected components
        connected_components = list(nx.connected_components(G))
        convex_vertices = []
        counter = 0
        for component in connected_components:
            convex_vertices.append(vertices[list(component)])
            counter += 1

        new_planes = []
        new_translations = []
        new_convex_indices = []
        counter = 0
        for i, convex in enumerate(convex_vertices):
            center = np.mean(convex, axis=0)
            convex -= center
            plane_equations = diff_convex.get_hull_equations(convex.astype(
                np.float64), np.asarray([0, 0, 0]).astype(np.float64)).reshape(-1, 4)

            good_equations = []
            for plane in plane_equations:
                if len(good_equations) == 0:
                    good_equations.append(plane)
                else:
                    good = True
                    for good_plane in good_equations:
                        if np.dot(plane[:3], good_plane[:3]) > 0.999:
                            good = False
                            break
                    if good:
                        good_equations.append(plane)
            if len(good_equations) < 4:
                print("not enough planes")
                continue
            good_equations = np.stack(good_equations, axis=0)
            good_equations[:, -1] = -np.abs(good_equations[:, -1]) - 1e-4
            new_planes.append(good_equations)
            new_translations.append(center)
            new_convex_indices.append(
                np.ones(good_equations.shape[0]) * counter)
            counter += 1

        new_planes = np.concatenate(new_planes, axis=0)
        new_translations = np.stack(new_translations, axis=0)
        new_convex_indices = np.concatenate(new_convex_indices, axis=0)

        new_planes = torch.from_numpy(new_planes).float().cuda()
        new_translations = torch.from_numpy(new_translations).float().cuda()
        new_convex_indices = torch.from_numpy(new_convex_indices).long().cuda()

        optimizer = self.optimizer if optimizer is None else optimizer

        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            new_planes, name="planes")
        self.planes = optimizable_tensors["planes"]

        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            new_translations, name="translations")
        self.translations = optimizable_tensors["translations"]

        self.convex_indices = new_convex_indices

    def purge_redundant_planes(self, optimizer=None):

        planes = self.planes
        normals = F.normalize(planes[:, :-1], dim=-1)
        offsets = -torch.abs(planes[:, -1:]) - 1e-5
        planes = torch.cat([normals, offsets], dim=-1).cuda()

        bounding_box_size = 1000
        bounding_planes = torch.tensor([[0, 0, 1, -bounding_box_size],
                                        [0, 0, -1, -bounding_box_size],
                                        [0, 1, 0, -bounding_box_size],
                                        [0, -1, 0, -bounding_box_size],
                                        [1, 0, 0, -bounding_box_size],
                                        [-1, 0, 0, -bounding_box_size]]).float().cuda()

        transformed_planes = self.get_transforme_halfspaces(planes)

        convex_indices = self.convex_indices.cpu().detach().numpy()
        tranlation = self.translations.cpu().detach().numpy()
        tranlation = np.concatenate([tranlation, np.zeros((1, 3))], axis=0)

        removing_convex_index = []
        for i, convex_id in enumerate(np.unique(convex_indices)):
            num_planes = (convex_indices == convex_id).sum()
            if num_planes < 4:
                convex_indices[convex_indices == convex_id] = -1
                print("Removing redundant convex!!!")
                removing_convex_index.append(convex_id)

        transformed_planes = transformed_planes[convex_indices != -1]
        convex_indices = convex_indices[convex_indices != -1]

        convex = diff_convex.convex_to_mesh(transformed_planes.cpu().detach().numpy().astype(np.float64),
                                            convex_indices,
                                            tranlation.astype(np.float64))
        active_plane_indices = np.unique(convex[0])
        planes = []
        convex_indices = []
        for i in range(self.planes.shape[0]):
            if i in active_plane_indices:
                planes.append(self.planes[i])
                convex_indices.append(self.convex_indices[i])

        planes = torch.stack(planes, dim=0)
        convex_indices = torch.stack(convex_indices, dim=0)

        optimizer = self.optimizer if optimizer is None else optimizer

        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            planes, name="planes")
        self.planes = optimizable_tensors["planes"]

        self.convex_indices = convex_indices

    def remove_small_convexes(self, threshold, optimizer=None):
        planes = self.planes
        normals = F.normalize(planes[:, :-1], dim=-1)
        offsets = -torch.abs(planes[:, -1:]) - 1e-5
        planes = torch.cat([normals, offsets], dim=-1).cuda()
        convex_indices = self.convex_indices.cpu().detach().numpy()
        for i, convex_id in enumerate(np.unique(convex_indices)):
            if (convex_indices == convex_id).sum() < 4:
                convex_indices[convex_indices == convex_id] = -1
                print("Removing small convex due to insufficient planes!!!")
                continue
            hs = HalfspaceIntersection(
                planes[convex_indices == convex_id].cpu().detach().numpy(), np.zeros(3))
            intersection_points = hs.intersections
            bbox_min = intersection_points.min(axis=0)
            bbox_max = intersection_points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
            if volume < threshold/3:
                convex_indices[convex_indices == convex_id] = -1
            volume = ConvexHull(points=intersection_points).volume
            translation = self.translations[int(convex_id)]
            if translation.norm() > 10:
                volume = 0
            if volume < threshold:
                convex_indices[convex_indices == convex_id] = -1

        planes = planes[convex_indices != -1]
        planes = torch.from_numpy(planes.cpu().detach().numpy()).float().cuda()
        convex_indices = torch.from_numpy(
            convex_indices[convex_indices != -1]).long().cuda()
        optimizer = self.optimizer if optimizer is None else optimizer
        optimizable_tensors = self.replace_tensor_to_optimizer(
            optimizer,
            planes, name="planes")
        self.planes = optimizable_tensors["planes"]

        self.convex_indices = convex_indices

    def replace_tensor_to_optimizer(self, optimizer, tensor, name):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(
                    group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def save_obj(self, path):
        mesh = self.get_mesh()
        v, f = mesh.get_mesh_verts_faces(0)
        with open(path, "w") as objfile:
            for i in range(v.shape[0]):
                objfile.write(
                    f"v {v[i, 0].item()} {v[i, 1].item()} {v[i, 2].item()}\n")
            for i in range(f.shape[0]):
                objfile.write(
                    f"f {f[i, 0].item()+1} {f[i, 1].item()+1} {f[i, 2].item()+1}\n")

    def forward(self):
        return self.get_mesh()
