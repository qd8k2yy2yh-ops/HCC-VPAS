# 步骤1：计算标签类别的数目
all_tumor_patch = 0
all_normal_patch = 0
all_tissue_patch = 0

for patches in dataset:
    for patch in patches:
        _, _, _, _, label = patch
        if label == 0:
            all_normal_patch += 1
        elif label == 1:
            all_tumor_patch += 1
        all_tissue_patch += 1

print("Number of normal patches:", all_normal_patch)
print("Number of tumor patches:", all_tumor_patch)
print("Total number of patches:", all_tissue_patch)

torch.manual_seed(42)

normal_patches = []
tumor_patches = []

for patches in dataset:
    for patch in patches:
        _, _, _, _, label = patch
        if label == 0:
            normal_patches.append(patch)
        elif label == 1:
            tumor_patches.append(patch)

torch.random.shuffle(normal_patches)
torch.random.shuffle(tumor_patches)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

total_normal_patches = len(normal_patches)
total_tumor_patches = len(tumor_patches)

train_normal_patches = int(train_ratio * total_normal_patches)
val_normal_patches = int(val_ratio * total_normal_patches)
test_normal_patches = total_normal_patches - train_normal_patches - val_normal_patches

train_tumor_patches = int(train_ratio * total_tumor_patches)
val_tumor_patches = int(val_ratio * total_tumor_patches)
test_tumor_patches = total_tumor_patches - train_tumor_patches - val_tumor_patches

train_data = normal_patches[:train_normal_patches] + tumor_patches[:train_tumor_patches]
val_data = normal_patches[train_normal_patches:train_normal_patches+val_normal_patches] + tumor_patches[train_tumor_patches:train_tumor_patches+val_tumor_patches]
test_data = normal_patches[train_normal_patches+val_normal_patches:] + tumor_patches[train_tumor_patches+val_tumor_patches:]

torch.manual_seed(43)

torch.random.shuffle(train_data)
torch.random.shuffle(val_data)
torch.random.shuffle(test_data)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
