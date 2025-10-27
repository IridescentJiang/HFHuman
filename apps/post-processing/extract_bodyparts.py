import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['vertex_indices'], data['faces']

def load_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extract_vertices(obj_lines, vertex_indices):
    vertices = []
    index_mapping = {}
    valid_vertices = [line for line in obj_lines if line.startswith('v ')]
    for new_index, obj_index in enumerate(vertex_indices):
        # Adjust by subtracting 1 because vertex_indices are 1-based
        line = valid_vertices[obj_index - 1]
        if line.startswith('v '):
            # Track the new index mapping
            index_mapping[obj_index] = new_index + 1  # OBJ format requires 1-based indexing
            vertices.append(line.strip())
    return vertices, index_mapping

def extract_faces(face_lines, index_mapping):
    extracted_faces = []
    for face_str in face_lines:
        if face_str.startswith('f '):
            # Parse the face indices
            original_indices = [int(i) for i in face_str.split()[1:]]
            
            # Remap them to the new indices
            try:
                new_indices = [index_mapping[idx] for idx in original_indices]
                extracted_faces.append(f"f {' '.join(map(str, new_indices))}")
            except KeyError:
                # Skip faces that reference vertices not in index_mapping
                continue
    return extracted_faces

def save_obj(vertices, faces, output_file):
    with open(output_file, 'w') as file:
        for vertex in vertices:
            file.write(vertex + '\n')
        for face in faces:
            file.write(face + '\n')

# 示例用法
obj_file = 'whole_body_0522.obj'

json_file = 'right_hand.json'
output_obj_file = 'right_hand_0522.obj'

vertex_indices, faces = load_json(json_file)
obj_lines = load_obj(obj_file)

vertices, index_mapping = extract_vertices(obj_lines, vertex_indices)
extracted_faces = extract_faces(faces, index_mapping)

save_obj(vertices, extracted_faces, output_obj_file)

print(f"Extracted OBJ data saved to {output_obj_file}")


json_file = 'left_hand.json'
output_obj_file = 'left_hand_0522.obj'

vertex_indices, faces = load_json(json_file)
obj_lines = load_obj(obj_file)

vertices, index_mapping = extract_vertices(obj_lines, vertex_indices)
extracted_faces = extract_faces(faces, index_mapping)

save_obj(vertices, extracted_faces, output_obj_file)

print(f"Extracted OBJ data saved to {output_obj_file}")

