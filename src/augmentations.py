import tensorflow as tf

def random_gaussian_blur(image, max_sigma=1.5):
    sigma = tf.random.uniform([], 0.0, max_sigma)
    kernel_size = 5
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-6))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    image = tf.expand_dims(image, axis=0)
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1,1,1,1], padding="SAME")
    image = tf.squeeze(image, axis=0)
    return image

def random_gaussian_noise(image, max_stddev=0.06):
    stddev = tf.random.uniform([], 0.0, max_stddev)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev)
    image = image + noise
    return tf.clip_by_value(image, 0.0, 1.0)

def random_jpeg_compression(image, min_quality=50, max_quality=95):
    """Simulates JPEG compression artifacts from saved/shared photos."""
    quality = tf.random.uniform([], min_quality, max_quality, dtype=tf.int32)
    image = tf.cast(image * 255.0, tf.uint8)
    image = tf.image.adjust_jpeg_quality(image, jpeg_quality=quality)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def random_sharpen(image, max_strength=0.4):
    """Simulates over-sharpened images from phone post-processing."""
    strength = tf.random.uniform([], 0.0, max_strength)
    kernel = tf.constant([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    sharpened = tf.nn.depthwise_conv2d(
        tf.expand_dims(image, 0), kernel, strides=[1, 1, 1, 1], padding="SAME"
    )
    sharpened = tf.squeeze(sharpened, axis=0)
    image = image * (1.0 - strength) + sharpened * strength
    return tf.clip_by_value(image, 0.0, 1.0)

def random_saturation(image, lower=0.7, upper=1.4):
    """Simulates different color intensities from phone cameras."""
    return tf.image.random_saturation(image, lower=lower, upper=upper)


def random_hue(image, max_delta=0.05):
    """Simulates slight color tint differences between devices."""
    return tf.image.random_hue(image, max_delta=max_delta)

def random_cutout(image, max_pct=0.15):
    """
    Randomly occludes a rectangular patch — simulates fingers,
    hair, bandages, or shadows partially covering the mole.
    """
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    cut_h = tf.cast(tf.cast(h, tf.float32) * max_pct, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * max_pct, tf.int32)
    y = tf.random.uniform([], 0, h - cut_h, dtype=tf.int32)
    x = tf.random.uniform([], 0, w - cut_w, dtype=tf.int32)

    mean_pixel = tf.reduce_mean(image)
    top    = image[:y, :, :]
    mid_l  = image[y:y+cut_h, :x, :]
    mid_c  = tf.fill([cut_h, cut_w, 3], mean_pixel)
    mid_r  = image[y:y+cut_h, x+cut_w:, :]
    bottom = image[y+cut_h:, :, :]
    mid    = tf.concat([mid_l, mid_c, mid_r], axis=1)
    image  = tf.concat([top, mid, bottom], axis=0)
    return image

# Add the rest of your augmentation functions here...

def apply_custom_augmentations(image):
    """Apply each custom augmentation with independent probability."""

    # 40% — slight blur (out-of-focus shot)
    if tf.random.uniform([]) < 0.4:
        image = random_gaussian_blur(image, max_sigma=1.5)

    # 40% — sensor noise (low-light photo)
    if tf.random.uniform([]) < 0.4:
        image = random_gaussian_noise(image, max_stddev=0.06)

    # 30% — JPEG artifacts (screenshot / compressed image)
    if tf.random.uniform([]) < 0.3:
        image = random_jpeg_compression(image, min_quality=50, max_quality=95)

    # 25% — over-sharpening (phone AI post-processing)
    if tf.random.uniform([]) < 0.25:
        image = random_sharpen(image, max_strength=0.4)

    # 50% — color saturation shift
    if tf.random.uniform([]) < 0.5:
        image = random_saturation(image, lower=0.7, upper=1.4)

    # 40% — hue shift (warm/cool lighting tint)
    if tf.random.uniform([]) < 0.4:
        image = random_hue(image, max_delta=0.05)

        # 20% — partial occlusion (hair, finger, bandage)
    if tf.random.uniform([]) < 0.2:
        image = random_cutout(image, max_pct=0.15)

    return image
