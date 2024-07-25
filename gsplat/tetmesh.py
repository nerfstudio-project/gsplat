### copy from https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/utils/tetmesh.py

# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

__all__ = ['marching_tetrahedra']

triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


def _unbatched_marching_tetrahedra(vertices, tets, sdf, scales):
    """unbatched marching tetrahedra.

    Refer to :func:`marching_tetrahedra`.
    """
    device = vertices.device
    
    # call by chunk
    chunk_size = 32 * 1024 * 1024
    if tets.shape[0] > chunk_size:
        merged_verts = None
        merged_scales = None
        merged_faces = None
        merged_verts_ids = None
        for tet_chunk in torch.chunk(tets, tets.shape[0] // chunk_size + 1):
            torch.cuda.empty_cache()
            verts, verts_scales, faces, verts_ids = _unbatched_marching_tetrahedra(vertices, tet_chunk, sdf, scales)
            
            if merged_verts is None:
                merged_verts = verts
                merged_scales = verts_scales
                merged_faces = faces
                merged_verts_ids = verts_ids
            else:
                all_edges = torch.cat([merged_verts_ids, verts_ids], dim=0)
                unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
                # merge vertices
                unique_verts_0 = torch.zeros((unique_edges.shape[0], 2, 3), dtype=torch.float, device=device)
                unique_verts_1 = torch.zeros((unique_edges.shape[0], 2, 1), dtype=torch.float, device=device)
                unique_verts_0[idx_map[:merged_verts[0].shape[0]]] = merged_verts[0]  
                unique_verts_0[idx_map[merged_verts[0].shape[0]:]] = verts[0]         
                unique_verts_1[idx_map[:merged_verts[1].shape[0]]] = merged_verts[1]  
                unique_verts_1[idx_map[merged_verts[1].shape[0]:]] = verts[1]         
                # merge scales
                unique_scales = torch.zeros((unique_edges.shape[0], 2, 1), dtype=torch.float, device=device)
                unique_scales[idx_map[:merged_verts[0].shape[0]]] = merged_scales     
                unique_scales[idx_map[merged_verts[0].shape[0]:]] = verts_scales      
                
                # merge faces
                unique_faces_0 = idx_map[merged_faces.reshape(-1)].reshape(-1, 3)
                unique_faces_1 = idx_map[faces.reshape(-1) + merged_verts[0].shape[0]].reshape(-1, 3)

                merged_faces = torch.cat([unique_faces_0, unique_faces_1], dim=0)
                merged_verts = (unique_verts_0, unique_verts_1)
                merged_scales = unique_scales
                merged_verts_ids = unique_edges
                torch.cuda.empty_cache()
                
        return merged_verts, merged_scales, merged_faces, merged_verts_ids
        
    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        
        # find all vertices
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        
        order = (all_edges[:, 0] > all_edges[:, 1]).bool()
        all_edges[order] = all_edges[order][:, [1, 0]]
        
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]
    edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    verts_scales = scales[interp_v.reshape(-1)].reshape(-1, 2, 1)
    
    verts = (edges_to_interp, edges_to_interp_sdf)
    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    return verts, verts_scales, faces, interp_v


def marching_tetrahedra(vertices, tets, sdf, scales):
    r"""Convert discrete signed distance fields encoded on tetrahedral grids to triangle 
    meshes using marching tetrahedra algorithm as described in `An efficient method of 
    triangulating equi-valued surfaces by using tetrahedral cells`_. The output surface is differentiable with respect to
    input vertex positions and the SDF values. For more details and example usage in learning, see 
    `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.


    Args:
        vertices (torch.tensor): batched vertices of tetrahedral meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        tets (torch.tensor): unbatched tetrahedral mesh topology, of shape
                             :math:`(\text{num_tetrahedrons}, 4)`.
        sdf (torch.tensor): batched SDFs which specify the SDF value of each vertex, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.

    Returns:
        (list[torch.Tensor], list[torch.LongTensor], (optional) list[torch.LongTensor]): 

            - the list of vertices for mesh converted from each tetrahedral grid.
            - the list of faces for mesh converted from each tetrahedral grid.

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]]], dtype=torch.float)
        >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([[-1., -1., 0.5, 0.5]], dtype=torch.float)
        >>> verts_list, faces_list, tet_idx_list = marching_tetrahedra(vertices, tets, sdf, True)
        >>> verts_list[0]
        tensor([[0.0000, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.6667],
                [0.3333, 0.6667, 0.0000],
                [0.3333, 0.0000, 0.6667]])
        >>> faces_list[0]
        tensor([[3, 0, 1],
                [3, 2, 0]])
        >>> tet_idx_list[0]
        tensor([0, 0])

    .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
        https://search.ieice.org/bin/summary.php?id=e74-d_1_214

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """

    list_of_outputs = [_unbatched_marching_tetrahedra(vertices[b], tets, sdf[b], scales[b]) for b in range(vertices.shape[0])]
    return list(zip(*list_of_outputs))