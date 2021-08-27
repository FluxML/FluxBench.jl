
function trimesh_benchmark(mesh::TriMesh)
    get_edges_packed(mesh; refresh = true)
    get_faces_to_edges_packed(mesh; refresh = true)
    get_laplacian_packed(mesh; refresh = true)
    compute_verts_normals_packed(mesh)
    compute_verts_normals_packed(mesh)
    compute_faces_areas_packed(mesh)
    compute_faces_normals_packed(mesh)
    l1 = laplacian_loss(mesh)
    l2 = edge_loss(mesh)
    l3 = chamfer_distance(mesh,mesh)
    return l1+l2+l3
end

function flux3d_add_trimesh(f3d_grp)
    
    download("https://raw.githubusercontent.com/McNopper/OpenGL/master/Binaries/teapot.obj",
        "teapot.obj")
    mesh = load_trimesh("teapot.obj")

    f3d_grp["Flux3D_TriMesh_Forward_Pass"] = @benchmarkable(
        fw(m, mesh),
        setup = (
            mesh = TriMesh($mesh);
            m = $trimesh_benchmark
        ),
        teardown = (GC.fc(); CUDA.reclaim())
    )

    f3d_grp["Flux3D_TriMesh_Backward_Pass"] = @benchmarkable(
        bw(m, mesh),
        setup = (
            mesh = TriMesh($mesh);
            m = $trimesh_benchmark
        ),
        teardown = (GC.fc(); CUDA.reclaim())
    )
   
    # f3d_grp["Flux3D_TriMesh_Forward_Pass_CUDA"] = @benchmarkable(
    #     fw(m, mesh),
    #     setup = (
    #         mesh = TriMesh($mesh) |> gpu;
    #         m = $trimesh_benchmark
    #     ),
    #     teardown = (GC.fc(); CUDA.reclaim())
    # )

    # f3d_grp["Flux3D_TriMesh_Backward_Pass_CUDA"] = @benchmarkable(
    #     bw(m, mesh),
    #     setup = (
    #         mesh = TriMesh($mesh) |> gpu;
    #         m = $trimesh_benchmark
    #     ),
    #     teardown = (GC.fc(); CUDA.reclaim())
    # )

end
