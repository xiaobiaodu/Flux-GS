import lzma
import dahuffman
import pickle
import numpy as np
import torch
import json
import struct

import torch
import json
import struct
import numpy as np

def load_comp_web(filename):
    with open(filename, "rb") as f:
        # 1. Read the 4-byte header length (Little-endian unsigned int)
        header_size_bytes = f.read(4)
        if len(header_size_bytes) < 4:
            raise ValueError("File is too short or corrupted.")
        
        header_size = struct.unpack('<I', header_size_bytes)[0]

        # 2. Read and decode the JSON metadata
        json_bytes = f.read(header_size)
        web_metadata = json.loads(json_bytes.decode('utf-8'))

        # 3. Read the remaining binary payload
        binary_payload = f.read()

    # 4. Recursive function to rebuild the Python objects
    def reconstruct_node(node):
        if isinstance(node, dict):
            # Check if this dict is actually a metadata marker for binary data
            if "_type" in node:
                offset = node["offset"]
                length = node["length"]
                
                if node["_type"] == "ndarray":
                    dtype = np.dtype(node["dtype"])
                    shape = tuple(node["shape"])
                    
                    # Extract EXACTLY the data bytes (ignoring the 4-byte padding we added for JS)
                    blob = binary_payload[offset:offset + length]
                    
                    # Reconstruct and reshape
                    arr = np.frombuffer(blob, dtype=dtype).copy()
                    return arr.reshape(shape)
                
                elif node["_type"] == "bytes":
                    return binary_payload[offset:offset + length]
            
            # Standard dictionary processing
            reconstructed_dict = {}
            for k, v in node.items():
                if k.lstrip('-').isdigit(): 
                    k = int(k)
                reconstructed_dict[k] = reconstruct_node(v)
            return reconstructed_dict

        elif isinstance(node, list):
            return [reconstruct_node(item) for item in node]
        else:
            return node

    return reconstruct_node(web_metadata)


def save_comp_web(filename, save_dict):
    binary_blobs = []
    current_offset = [0]

    def process_node(val):
        # 1. Handle PyTorch Tensors
        if isinstance(val, torch.Tensor):
            # .contiguous() is MANDATORY here to ensure JS arrays don't read memory out of order
            val = val.detach().cpu().contiguous().numpy()
        # 2. Handle NumPy Arrays
        if isinstance(val, np.ndarray):
            # Ensure Little-Endian (JS standard)
            if val.dtype.byteorder == '>':
                val = val.astype(val.dtype.newbyteorder('<'))                
            flat_bytes = val.tobytes()
            actual_length = len(flat_bytes)
            
            # JS-OPTIMIZATION: 4-Byte padding alignment
            padding_len = (4 - (actual_length % 4)) % 4
            padded_bytes = flat_bytes + (b'\x00' * padding_len)
            
            meta = {
                "_type": "ndarray",
                "dtype": str(val.dtype),
                "shape": val.shape,
                "offset": current_offset[0],
                "length": actual_length # Keep actual length for loader, offset accounts for padding
            }
            binary_blobs.append(padded_bytes)
            current_offset[0] += len(padded_bytes)
            return meta

        # 3. Handle Raw Bytes
        elif isinstance(val, (bytes, bytearray)):
            actual_length = len(val)
            
            # JS-OPTIMIZATION: 4-Byte padding alignment
            padding_len = (4 - (actual_length % 4)) % 4
            padded_bytes = val + (b'\x00' * padding_len)
            
            meta = {
                "_type": "bytes",
                "offset": current_offset[0],
                "length": actual_length
            }
            binary_blobs.append(padded_bytes)
            current_offset[0] += len(padded_bytes)
            return meta

        # 4. Handle Lists
        elif isinstance(val, list):
            return [process_node(v) for v in val]

        # 5. Handle Dictionaries
        elif isinstance(val, dict):
            cleaned_dict = {}
            for k, v in val.items():
                if isinstance(k, np.generic):
                    k = k.item()
                if not isinstance(k, (str, int, float, bool, type(None))):
                    k = str(k)
                cleaned_dict[k] = process_node(v)
            return cleaned_dict

        # 6. JSON Primitives
        else:
            if isinstance(val, np.generic):
                return val.item()
            return val

    web_metadata = process_node(save_dict)
    json_bytes = json.dumps(web_metadata, separators=(',', ':')).encode('utf-8')

    with open(filename, "wb") as f:
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_bytes)
        for blob in binary_blobs:
            f.write(blob)
            
    print(f"Exported to {filename}")
    print(f"-> Header: {len(json_bytes)} bytes | Binary Data: {current_offset[0]} bytes (Padded for JS)")



def huffman_encode(data):
    codec = dahuffman.HuffmanCodec.from_data(data)
    encoded_bytes = codec.encode(data)
    huffman_table = codec.get_code_table()
    return encoded_bytes, huffman_table

def huffman_decode(encoded_bytes, huffman_table):
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    decoded_data = codec.decode(encoded_bytes)
    return np.array(decoded_data, dtype=np.uint16)

def save_comp(filename, save_dict):
    with lzma.open(filename, "wb") as f:
        pickle.dump(save_dict, f)

def load_comp(filename):
    with lzma.open(filename, "rb") as f:
        save_dict = pickle.load(f)
    return save_dict

def write_storage(save_dict, byte, numG):
    for name in save_dict:
        if name == 'xyz':
            byte['xyz'] = len(save_dict['xyz'])
        elif "offset" in name:
            num_params = sum(v.size for v in save_dict[name].values())
            byte['MLPs'] += num_params*16/8
        elif 'MLP' in name:
            byte['MLPs'] += save_dict[name].shape[0]*16/8
        else:
            attr, comp = name.split('_')
            if 'code' in comp:
                for i in range(len(save_dict[name])):
                    byte[attr] += save_dict[name][i].shape[0]*save_dict[name][i].shape[1]*16/8
            else:
                for i in range(len(save_dict[name])):
                    byte[attr] += len(save_dict[name][i])
    byte['total'] = byte['xyz'] + byte['scale'] + byte['rotation'] + byte['app'] + byte['MLPs'] + byte['opacity']
    return "#G: " + str(numG) + "\nPosition: " + str(byte['xyz']) + "\nScale: " + str(byte['scale']) + "\nRotation: " + str(byte['rotation']) + "\nAppearance: " + str(byte['app']) + "\nMLPs: " + str(byte['MLPs'])+  "\nopacity: " + str(byte['opacity'])+ "\nTotal: " + str(byte['total']) + "\n"

def splitBy3(a):
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer