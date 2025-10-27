import json

def load_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_vertex(line):
    try:
        _, x, y, z = line.split()
        return float(x), float(y), float(z)
    except ValueError:
        raise ValueError(f"Cannot parse line: {line}")

def are_vertices_close(v1, v2, epsilon=1e-6):  # 使用更小的 epsilon
    return all(abs(a - b) <= epsilon for a, b in zip(v1, v2))

def find_matching_vertices(A_file, B_file):
    A_lines = load_obj(A_file)
    B_lines = load_obj(B_file)

    A_vertices = [parse_vertex(line) for line in A_lines if line.startswith('v ')]
    B_vertices = [parse_vertex(line) for line in B_lines if line.startswith('v ')]

    matching_indices = []

    for a_vertex in A_vertices:
	    matched = False
	    for i, b_vertex in enumerate(B_vertices):
	        if are_vertices_close(a_vertex, b_vertex):
		        matching_indices.append(i+1)
		        matched = True
		        break
                
    return matching_indices

def find_matching_faces(B_file, matching_indices):
    B_lines = load_obj(B_file)
    face_list = []

    for line in B_lines:
        if line.startswith('f '):
            indices = [int(i) for i in line.split()[1:]]
            if all(index in matching_indices for index in indices):
                face_list.append(line.strip())
    return face_list

def save_to_json(vertex_indices, face_list, output_file):
    data = {
        "vertex_indices": vertex_indices,
        "faces": face_list
    }
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# 示例用法
A_file = 'face.obj'
B_file = 'test_0004_smpl_opt.obj'
output_json_file = 'face.json'

matching_indices = find_matching_vertices(A_file, B_file)
face_list = find_matching_faces(B_file, matching_indices)

save_to_json(matching_indices, face_list, output_json_file)

print(f"Data saved to {output_json_file}")

