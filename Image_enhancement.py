import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import random
import torch.nn.functional as F

# 随机旋转
class rotation_transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, angle_range=30):
        batch_size, channels, height, width = image_tensor.size()
        random_angle = torch.FloatTensor(batch_size).uniform_(-angle_range, angle_range)
        rotated_batch = []
        for i in range(batch_size):
            single_image = image_tensor[i]
            rotated_image = torchvision.transforms.functional.rotate(single_image, random_angle[i].item())
            rotated_batch.append(rotated_image)

        rotated_batch = torch.stack(rotated_batch, dim=0)
        return rotated_batch


# 随机裁剪后插值到原图大小
class center_crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, target_size):
        batch_size, channels, height, width = image_tensor.size()

        cropped_images = []

        for i in range(batch_size):
            single_image = image_tensor[i]
            top = (height - target_size) // 2
            left = (width - target_size) // 2
            cropped_image = single_image[:, top:top + target_size, left:left + target_size]
            cropped_images.append(cropped_image)

        cropped_batch = torch.stack(cropped_images, dim=0)
        return cropped_batch


class random_resized_crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, scale_range=(0.5, 1.0)):
        batch_size, channels, height, width = image_tensor.size()

        # 随机生成裁剪框尺寸和位置
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1])
        scaled_height = int(height * scale_factor)
        scaled_width = int(width * scale_factor)

        cropped_images = []

        for i in range(batch_size):
            single_image = image_tensor[i]
            top = (height - scaled_height) // 2
            left = (width - scaled_width) // 2

            cropped_image = single_image[:, top:top + scaled_height, left:left + scaled_width]
            cropped_resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(height, width),
                                                  mode='bicubic', align_corners=False)
            cropped_resized_image = cropped_resized_image.squeeze(0)
            cropped_images.append(cropped_resized_image)

        cropped_batch = torch.stack(cropped_images, dim=0)
        return cropped_batch


# 水平翻转
class random_horizontal_flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, p=0.5):
        batch_size, channels, height, width = image_tensor.size()
        flip_mask = torch.FloatTensor(batch_size).uniform_() < p

        flipped_images = []

        for i in range(batch_size):
            single_image = image_tensor[i]
            flipped_image = torch.flip(single_image, [2]) if flip_mask[i] else single_image
            flipped_images.append(flipped_image)

        flipped_batch = torch.stack(flipped_images, dim=0)
        return flipped_batch


class Bevel_flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, scale_factor=0, p=0.5):
        batch_size, channels, height, width = image_tensor.size()
        deform_range = int(min(height, width) * scale_factor)

        deform_x = torch.arange(height).unsqueeze(0).expand(height, width)
        deform_y = torch.arange(width).unsqueeze(1).expand(height, width)

        random_flip_mask = (torch.rand(height, width) < p).long()
        deform_x_clone = deform_x.clone()
        deform_y_clone = deform_y.clone()

        deform_x_clone += (random_flip_mask * torch.randint(-deform_range, deform_range + 1, (height, width))).long()
        deform_y_clone += (random_flip_mask * torch.randint(-deform_range, deform_range + 1, (height, width))).long()

        deform_x_clone = deform_x_clone.clamp(0, height - 1)
        deform_y_clone = deform_y_clone.clamp(0, width - 1)

        deformed_images = torch.stack([image_tensor[:, c, deform_x_clone, deform_y_clone] for c in range(channels)],
                                      dim=1)
        return deformed_images


# 垂直翻转
class random_vertical_flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, p=0.5):
        batch_size, channels, height, width = image_tensor.size()
        flip_mask = torch.FloatTensor(batch_size).uniform_() < p

        flipped_images = []

        for i in range(batch_size):
            single_image = image_tensor[i]
            flipped_image = torch.flip(single_image, [1]) if flip_mask[i] else single_image
            flipped_images.append(flipped_image)

        flipped_batch = torch.stack(flipped_images, dim=0)
        return flipped_batch


# 微小形变
class SmallScaleDeformation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, scale_factor = 0.1, deform_prob=0.2, p=0.5):
        batch_size, channels, height, width = image_tensor.size()
        deform_range = int(min(height, width) * scale_factor)
        flip_mask = torch.FloatTensor(batch_size).uniform_() < p

        deformed_images = []

        for i in range(batch_size):
            single_image = image_tensor[i]
            deformed_channel_images = []

            for c in range(channels):
                deform_x = torch.arange(height).unsqueeze(0).expand(height, width).long()
                deform_y = torch.arange(width).unsqueeze(1).expand(height, width).long()

                random_shift_x = (torch.rand(height, width) < deform_prob).long() * torch.randint(-deform_range,
                                                                                                  deform_range + 1, (
                                                                                                  height, width)).long()
                random_shift_y = (torch.rand(height, width) < deform_prob).long() * torch.randint(-deform_range,
                                                                                                  deform_range + 1, (
                                                                                                  height, width)).long()

                deform_x = (deform_x + random_shift_x).clamp(0, height - 1)
                deform_y = (deform_y + random_shift_y).clamp(0, width - 1)

                # Perform interpolation to ensure consistent dimensions across channels
                deformed_channel_image = single_image[c, deform_y, deform_x]
                deformed_channel_image = torch.nn.functional.interpolate(
                    deformed_channel_image.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bicubic',
                    align_corners=False)
                deformed_channel_image = deformed_channel_image.squeeze(0).squeeze(0)

                deformed_channel_images.append(deformed_channel_image)

            deformed_images.append(torch.stack(deformed_channel_images, dim=0))

        deformed_batch = torch.stack(deformed_images, dim=0)
        return deformed_batch



class random_noise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor, noise_std=0.01):
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_images = image_tensor + noise
        return noisy_images


class standardization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hyperspectral_data):
        mean = torch.mean(hyperspectral_data, dim=(0, 2, 3), keepdim=True)
        std = torch.std(hyperspectral_data, dim=(0, 2, 3), keepdim=True)

        # Normalize the data
        normalized_data = (hyperspectral_data - mean) / std

        return normalized_data


class CustomCompose(nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, image_tensor):
        for transform in self.transforms:
            image_tensor = transform(image_tensor)
        return image_tensor


class RandomChoice(nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, image_tensor):
        chosen_transform = random.choice(self.transforms)
        return chosen_transform(image_tensor)


class mask(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        D, C, height, width, = x.size()
        x = x.reshape(D, C, height*width)
        x = x.permute(0, 2, 1)
        len_keep = int(height*width * (1 - self.mask_ratio))

        noise = torch.rand(D, height*width, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        mask_token = torch.zeros([1, 1, C], device=x.device)
        mask_tokens = mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
        x_ = torch.cat([x_masked, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        assert height * width == x.shape[1]
        x = x.permute(0, 2, 1).view(D, C, height, width)
        return x

class figure_rand(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_tensor):
        # 使用示例
        transform_options = RandomChoice(
            rotation_transform(),
            random_resized_crop(),
            random_horizontal_flip(),
            random_vertical_flip(),
            Bevel_flip(),
            # SmallScaleDeformation(),
            random_noise()
        )

        # 应用变换
        transformed_images = transform_options(image_tensor)
        # transformed_images = normalize_hyperspectral_data(transformed_images)
        return transformed_images


