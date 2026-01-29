import h5py
import random

def filter_coords(coords_dir, slide_level, patch_size, tissue_type, random_sample, stride=None):
    with h5py.File(coords_dir, 'r') as f:
        coords = f['coords'][:]
        slide_name = coords_dir.split('/')[-1].split('.')[0]
        if 'tissue_type' in f:
            labels = f['tissue_type'][:]
        else:
            labels = [55] * len(coords)
            
        if len(coords) == 0:
            print(f"Coords is empty in file: {coords_dir}")

        coords = coords.tolist()
        indices = list(range(len(coords)))
        real_patch_size = (slide_level + 1) * patch_size
        if tissue_type is not None:
            valid_indices = [i for i, lbl in enumerate(labels) if lbl == tissue_type]
            indices = valid_indices
            coords = [coords[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

        if real_patch_size > 256:
            valid_coords = []
            valid_labels = []
            v_indices = []
            pn = real_patch_size / 256
            try:
                x = 256
                y = 256
                for c in coords:
                    c_x, c_y = c
                    if (c_x != 0 and c_y != 0) and (c_x % (x * pn) == 0 and c_y % (y * pn) == 0):
                        valid_coords.append(c)
                        valid_labels.append(labels[coords.index(c)])
                        v_indices.append(indices[coords.index(c)])
                coords = valid_coords
                labels = valid_labels
                indices = v_indices
            except:
                print(ValueError, ' ', slide_name, ".h5 can not extract coords=============")

        if random_sample is not None:
            random.seed(43)
            num_samples = len(coords)
            sample_indices = random.sample(range(num_samples), int(num_samples / random_sample))
            coords = [coords[i] for i in sample_indices]
            labels = [labels[i] for i in sample_indices]
            indices = [indices[i] for i in sample_indices]

        if stride is None:
            slide_names = [slide_name] * len(coords)
            return slide_names, coords, labels, indices

        if stride is not None and stride < patch_size:
            if patch_size == 256 and slide_level == 0:
                valid_coords = []
                valid_labels = []
                v_indices = []
                for coords in coords:
                    x, y = coords
                    y256_coords = (x, y + 256)
                    x256_coords = (x + 256, y)
                    xy256_coords = (x + 256, y + 256)

                    if y256_coords in coords and x256_coords in coords and xy256_coords in coords:
                        y256_label = labels[coords.index(y256_coords)]
                        x256_label = labels[coords.index(x256_coords)]
                        xy256_label = labels[coords.index(xy256_coords)]

                        if y256_label == x256_label == xy256_label == label:
                            aug_coord = (c_x + stride, c_y + stride)
                            aug_label = label
                            indice = indices[coords.index(c)] + 0.5

                            valid_coords.append(aug_coord)
                            valid_labels.append(aug_label)
                            v_indices.append(indice)

                coords += valid_coords
                labels += valid_labels
                indices += v_indices

            if patch_size != 256 or slide_level != 0:

                valid_coords = []
                valid_labels = []
                v_indices = []

                for coord, label, idx in zip(coords, labels, indices):
                    aug_coord = (coord[0] + stride, coord[1] + stride)
                    valid_coords.append(aug_coord)
                    valid_labels.append(label)
                    v_indices.append(idx + 0.5)

                coords += valid_coords
                labels += valid_labels
                indices += v_indices

            slide_names = [slide_name] * len(coords)
            return slide_names, coords, labels, indices

